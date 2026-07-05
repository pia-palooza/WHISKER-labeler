import fnmatch
import pickle
import queue
import selectors
import socket
import struct
import time
from typing import cast, Any, Optional, Set

from ...base import topics as task_topics
from ..node import Node
from ..messaging import Message

_SelfTopics = task_topics.socket_router

class SocketRouterNode(Node):
    def __init__(self, label: str):
        super().__init__(
            label=label,
            subscriptions=set(_SelfTopics.Request.__members__.values())
        )
        self._selector = selectors.DefaultSelector()
        self._server_socks: Set[socket.socket] = set()
        self._peer_socks: Set[socket.socket] = set()
        self._forward_patterns: Set[str] = set()
        self._recv_buffers: dict[int, bytearray] = {}

    def setup(self) -> None:
        """Lifecycle hook: Called before the main loop starts."""
        super().setup()

    def wakeup(self) -> None:
        """Lifecycle hook: Called periodically by the task runner."""
        super().wakeup()
        self._run_loop_iteration()

    def handle_message(self, message: Message) -> bool:
        """Lifecycle hook: Intercepts messages to handle base routing first."""
        if super().handle_message(message):
            return True

        topic = message.header.topic
        
        if message.header.routing_path and self.uuid in message.header.routing_path:
            return True
        message.header.routing_path.append(self.uuid)

        message_id = message.header.message_id
        if message.header.target_node_id == self._uuid:
            if topic == _SelfTopics.Request.BIND:
                self._handle_bind(message.payload.address)
                self.send_outgoing_reply(message_id, _SelfTopics.BindSocketSuccessReply(address=message.payload.address))
                return True
            elif topic == _SelfTopics.Request.CONNECT:
                self._handle_connect(message.payload.address)
                self.send_outgoing_reply(message_id, _SelfTopics.ConnectSocketSuccessReply(address=message.payload.address))
                return True
            elif topic == _SelfTopics.Request.FORWARD_START:
                for pattern in message.payload.patterns:
                    self._forward_patterns.add(pattern)
                    self.add_subscription(pattern)
                self.send_outgoing_reply(message_id, _SelfTopics.ForwardingStartedReply(patterns=message.payload.patterns))
                return True
        elif any(fnmatch.fnmatch(topic, pat) for pat in self._forward_patterns):
            self.logger.debug(f"Forwarding '{topic}' message from {message.header.sender_id} to network peers.")
            self._broadcast_to_network(message)
            return True
        return False

    def _run_loop_iteration(self) -> None:
        """Overriden by child classes to implement a function which is called on each run loop iteration."""
        # Windows Fix: select() throws WinError 10022 if the selector has zero registered targets.
        # Short-circuit and sleep the tick interval away if we have no sockets to multiplex yet.
        if self._selector.get_map():
            events = self._selector.select(timeout=0)
            for key, mask in events:
                if key.data is True:
                    self._accept_peer(key.fileobj)
                else:
                    self._read_peer(key.fileobj)

    def _handle_bind(self, address: str) -> None:
        host, port = address.split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        sock.listen(5)
        sock.setblocking(False)
        
        self._server_socks.add(sock)
        self._selector.register(sock, selectors.EVENT_READ, data=True)

    def _handle_connect(self, address: str) -> None:
        """Connects directly to a remote socket bridge node."""
        host, port = address.split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(True)
        try:
            sock.connect((host, int(port)))
        except ConnectionRefusedError as e:
            self.logger.error(f"{self._uuid} Failed to connect: {repr(e)}")
            return

        sock.setblocking(False)
        
        self._peer_socks.add(sock)
        self._recv_buffers[sock.fileno()] = bytearray()
        self._selector.register(sock, selectors.EVENT_READ, data=False)

    def _accept_peer(self, server_sock: socket.socket) -> None:
        """Accepts incoming connection handshakes."""
        conn, addr = server_sock.accept()
        conn.setblocking(False)
        self._peer_socks.add(conn)
        self._recv_buffers[conn.fileno()] = bytearray()
        self._selector.register(conn, selectors.EVENT_READ, data=False)

    def _read_peer(self, sock: socket.socket) -> None:
        """Drains data from physical sockets, framing wire payloads back to Messages."""
        fd = sock.fileno()
        try:
            data = sock.recv(4096)
            if not data:
                self._disconnect_peer(sock)
                return
            
            self._recv_buffers[fd].extend(data)
            buf = self._recv_buffers[fd]

            while len(buf) >= 4:
                payload_len = struct.unpack("!I", buf[:4])[0]
                if len(buf) < 4 + payload_len:
                    break
                
                msg_bytes = buf[4 : 4 + payload_len]
                del buf[: 4 + payload_len]
                
                # Push back directly to the local bus send queue
                self.message_queue.send(pickle.loads(msg_bytes))

        except (ConnectionResetError, OSError):
            self._disconnect_peer(sock)

    def _broadcast_to_network(self, message: Message) -> None:
        """Pushes structured data out to all registered network peers."""
        serialized = pickle.dumps(message)
        frame = struct.pack("!I", len(serialized)) + serialized
        
        dead_peers = set()
        for sock in self._peer_socks:
            try:
                sock.sendall(frame)
            except OSError:
                dead_peers.add(sock)
                
        for dead in dead_peers:
            self._disconnect_peer(dead)

    def _disconnect_peer(self, sock: socket.socket) -> None:
        """Cleans up internal allocations for disconnected sockets."""
        if sock in self._peer_socks:
            self._peer_socks.remove(sock)
        fd = sock.fileno()
        self._recv_buffers.pop(fd, None)
        try:
            self._selector.unregister(sock)
            sock.close()
        except KeyError:
            pass

    def shutdown(self) -> None:
        self.reset()
        super().shutdown()

    def reset(self) -> None:
        """Clean teardown hook."""
        
        for sock in list(self._peer_socks):
            self._disconnect_peer(sock)
            
        for sock in list(self._server_socks):
            try:
                self._selector.unregister(sock)
                sock.close()
            except Exception:
                pass
        self._server_socks.clear()
        self._selector.close()
