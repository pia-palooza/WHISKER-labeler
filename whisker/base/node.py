import logging
import uuid
import threading
from typing import Any

from .messaging import MessageQueue, Message, MessageHeader
from . import topics

_UUID_LABEL_SEPARATOR = "@"

class InvalidLabelError(ValueError):
    pass

def validate_label(label: str) -> None:
    """Raises InvalidLabelError if the label is invalid."""
    if _UUID_LABEL_SEPARATOR in label:
        raise InvalidLabelError(
            f"Invalid label '{label}' due to containing "
            f"'{_UUID_LABEL_SEPARATOR}' character."
        )

def make_uuid(label: str) -> str:
    """Generates a UUID by combining the label with a random UUID4 string."""
    validate_label(label)
    return label + _UUID_LABEL_SEPARATOR + str(uuid.uuid4())[:8]

class Node:
    def __init__(self, label: str, subscriptions: set[str] | None = None):
        self._label = label
        self._uuid = make_uuid(label)
        self._subscriptions = (subscriptions or set()).union(
            topics.node.Request.__members__.values()
        )
        self._queue = MessageQueue()
        self._shutdown_event = threading.Event()

        # Configure instance-isolated logging to avoid cross-talk between nodes
        self.logger = logging.getLogger(f"{__name__}.{self._uuid}")

    ############################################################################

    @property
    def label(self) -> str:
        return self._label
    
    @property
    def uuid(self) -> str:
        return self._uuid
    
    @property
    def subscriptions(self) -> set[str]:
        return self._subscriptions
    
    @property
    def message_queue(self) -> MessageQueue:
        return self._queue
    
    @property
    def received_shutdown(self) -> bool:
        return self._shutdown_event.is_set()
    
    ############################################################################

    def add_subscription(self, topic: str) -> None:
        self._subscriptions.add(topic)
    
    def add_subscriptions(self, topics: set[str]) -> None:
        self._subscriptions.update(topics)

    def remove_subscription(self, topic: str) -> None:
        self._subscriptions.discard(topic)
    
    def remove_subscriptions(self, topics: set[str]) -> None:
        self._subscriptions.difference_update(topics)
        
    ############################################################################

    def create_message(
        self,
        payload: Any,
        **kwargs
    ) -> Message:
        header = MessageHeader(
            topic=payload.TOPIC_ID,
            sender_id=self._uuid,
            **kwargs
        )
        return Message(header=header, payload=payload)

    def send_outgoing_message(
        self,
        payload: Any,
        **kwargs
    ) -> None:
        """Constructs a message and places it in the outgoing message queue."""
        self._queue.send(self.create_message(payload, **kwargs))

    def send_outgoing_reply(
        self,
        request_message_id: str,
        payload: Any,
        **kwargs
    ):
        self.send_outgoing_message(payload, correlation_id=request_message_id, **kwargs)

    def process_incoming_messages(self):
        """Processes all messages in the incoming message queue."""
        while message := self._queue.get_nowait():
            self.handle_message(message)
            self._queue.incoming_message_handled()

    ############################################################################

    def setup(self):
        """Override in subclasses to implement any setup logic before the main loop starts."""
        self._shutdown_event.clear()

    def wakeup(self):
        """Override this method in subclasses to implement periodic custom logic."""
        self.process_incoming_messages()

    def handle_message(self, message: Message) -> bool:
        """Override this method in subclasses to handle incoming messages."""
        topic = message.header.topic

        topic_names = topics.node.Request.__members__.values()
        if topic in topic_names and message.header.target_node_id == self._uuid:
            if topic == topics.node.Request.PING:
                self.send_outgoing_reply(message.header.message_id,  topics.node.PongReply())
            elif topic == topics.node.Request.SHUTDOWN:
                self.send_outgoing_reply(message.header.sender_id, topics.node.ShutdownAck())
                self.request_shutdown()
            return True
        return False
    
    def shutdown(self):
        """Override in subclasses to implement any cleanup logic after the main loop ends."""
        pass

    def request_shutdown(self):
        self.logger.info(f"{self._uuid} triggering shutdown request")
        self._shutdown_event.set()