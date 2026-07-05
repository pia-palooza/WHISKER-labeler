import dataclasses
import fnmatch
from typing import Dict

from ...base import topics
from ..messaging import Message
from ..node import Node

class QueueRouterNode(Node):
    def __init__(self, label: str):
        super().__init__(
            label,
            subscriptions=set([
                topics.queue_router.Request.NODE_LIST,
                topics.queue_router.Request.NODE_INFO,
            ])
        )
        self._nodes: Dict[str, Node] = {}
        self._exact_routing_table: Dict[str, set[str]] = {}
        self._wildcard_routing_table: Dict[str, set[str]] = {}
        
        # Ensure the router is a fully participating node on its own bus
        # This populates routing tables with the router's own subscriptions
        self.register_node(self)

    def register_node(self, node: Node) -> None:
        if node.uuid in self._nodes:
            return
        self._nodes[node.uuid] = node
        self._sync_node_subscriptions(node)
        self.send_outgoing_message(
            topics.queue_router.NodeRegisteredTelemetry(
                uuid=node.uuid,
                label=node.label,
                subscriptions=node.subscriptions
            )
        )

    def unregister_node(self, node_uuid: str) -> None:
        if node := self._nodes.pop(node_uuid, None):
            for topic in node.subscriptions:
                table = self._wildcard_routing_table if ("*" in topic or "?" in topic) else self._exact_routing_table
                if topic in table:
                    table[topic].discard(node_uuid)
                    if not table[topic]:
                        del table[topic]

    def wakeup(self) -> None:
        super().wakeup()

        for node in list(self._nodes.values()):
            self._sync_node_subscriptions(node)
            while message := node.message_queue.pull_nowait():
                self._route_message(message)
                node.message_queue.outgoing_message_handled()


    def _sync_node_subscriptions(self, node: Node) -> None:
        for topic in node.subscriptions:
            table = self._wildcard_routing_table if ("*" in topic or "?" in topic) else self._exact_routing_table
            table.setdefault(topic, set()).add(node.uuid)

    def _route_message(self, message: Message) -> None:
        if message.header.routing_path and self.uuid in message.header.routing_path:
            return

        targets: set[str] = set()
        topic = message.header.topic
        if topic in self._exact_routing_table:
            targets.update(self._exact_routing_table[topic])

        for pattern, uuids in self._wildcard_routing_table.items():
            if fnmatch.fnmatch(topic, pattern):
                targets.update(uuids)

        self.log_bus_message(message)

        if not targets:
            self.logger.debug(f"No subscribers for {message.header.topic} message.")
        else:
            self.logger.debug(f"Routing {message.header.topic} to targets: {targets}")

        message.header.routing_path.append(self.uuid)
        for node_uuid in targets:
            if node := self._nodes.get(node_uuid):
                node.message_queue.put(message)

    def log_bus_message(self, message: Message) -> None:
        self.logger.debug(dataclasses.asdict(message))

    def handle_message(self, message: Message) -> bool:
        """
        Intercepts incoming messages to ensure the router processes 
        base node requests (like SHUTDOWN) and router-specific bus commands.
        """
        # Allow the base Node to process common commands like PING and SHUTDOWN
        if super().handle_message(message):
            return True
            
        # If the base class didn't claim it, process specialized router commands
        if message.header.target_node_id == self._uuid:
            self._handle_bus_command(message)
            return True
            
        return False

    def _handle_bus_command(self, message: Message) -> None:
        topic = message.header.topic
        if topic == topics.queue_router.Request.NODE_LIST:
            self.send_outgoing_reply(
                message.header.message_id,
                topics.queue_router.NodeListResponse(nodes=list(self._nodes.keys())),
            )
        elif topic == topics.queue_router.Request.NODE_INFO:
            node_uuid = message.payload.uuid
            if node := self._nodes.get(node_uuid):
                self.send_outgoing_reply(
                    message.header.message_id,
                    topics.queue_router.NodeInfoResponse(
                        uuid=node.uuid,
                        label=node.label,
                        subscriptions=node.subscriptions
                    )
                )
            else:
                self.send_outgoing_reply(
                    message.header.message_id,
                    topics.queue_router.ErrorResponse(
                        failed_topic=topic,
                        error_code=topics.queue_router.ErrorCodes.INVALID_NODE_UUID,
                        message=f"Node {node_uuid} not found."
                    )
                )
