from dataclasses import dataclass, field
from typing import Type, Dict, Optional, Generic, Type, TypeVar
from enum import Enum
import queue
import time
import uuid

_TOPIC_REGISTRY: Dict[str, Type] = {}

@dataclass(slots=True)
class MessageHeader:
    topic: str
    sender_id: str
    target_node_id: Optional[str] = None
    correlation_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    routing_path: list[str] = field(default_factory=list)

P = TypeVar('P')

@dataclass(slots=True)
class Message(Generic[P]):
    header: MessageHeader
    payload: P

class InvalidPayloadError(ValueError):
    pass

def check_payload_type(payload: P, expected_type: Type[P]) -> P:
    if type(payload) is not expected_type:
        raise InvalidPayloadError(
            f"Payload type {type(payload).__name__} "
            f"does not match expected type {expected_type.__name__}"
        )
    return payload

def register_topic(topic: Enum | str):
    """Registers a payload class to a topic string or Enum value."""
    def wrapper(cls: Type) -> Type:
        topic_str = topic.value if isinstance(topic, Enum) else topic
        cls.TOPIC_ID = topic_str
        _TOPIC_REGISTRY[topic_str] = cls
        return cls
    return wrapper

def get_type_for_topic(topic: str) -> Optional[Type]:
    return _TOPIC_REGISTRY.get(topic)

class MessageQueue:
    def __init__(self):
        self._incoming = queue.Queue()
        self._outgoing = queue.Queue()
    
    ############################################################################

    def get(self, timeout: float | None = None) -> Message | None:
        try:
            return self._incoming.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_nowait(self) -> Message | None:
        try:
            return self._incoming.get_nowait()
        except queue.Empty:
            return None

    def put(self, message: Message) -> None:
        self._incoming.put_nowait(message)

    def incoming_message_handled(self):
        self._incoming.task_done()

    ############################################################################

    def send(self, message: Message) -> None:
        self._outgoing.put_nowait(message)

    def pull(self, timeout: float | None = None) -> Message | None:
        try:
            return self._outgoing.get(timeout=timeout)
        except queue.Empty:
            return None

    def pull_nowait(self) -> Message | None:
        try:
            return self._outgoing.get_nowait()
        except queue.Empty:
            return None
    
    def outgoing_message_handled(self):
        self._outgoing.task_done()
