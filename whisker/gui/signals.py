from typing import Callable, Optional, Any
from whisker.core.messaging.bus import BaseMessageBus
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

class MessageBus(QObject):
    """
    PyQt6 Thread-Safe Wrapper singleton. 
    Composes and interacts with the BaseMessageBus singleton instance.
    """
    _instance = None
    _message_emitted = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        # Link directly to the shared BaseMessageBus singleton instance
        self._core = BaseMessageBus.get()
        self._message_emitted.connect(self._handle_emitted_signal)

    @classmethod
    def get(cls) -> "MessageBus":
        if cls._instance is None:
            cls._instance = MessageBus()
        return cls._instance

    def subscribe(self, topic_pattern: str, callback: Callable):
        self._core.subscribe(topic_pattern, callback)

    def unsubscribe(self, topic_pattern: str, callback: Callable):
        self._core.unsubscribe(topic_pattern, callback)

    def enable_debug_output(self, enable: bool):
        self._core.enable_debug_output(enable)

    def publish(self, topic: str, payload: Optional[Any] = None):
        payload = payload if payload is not None else {}
        import threading
        if threading.current_thread() is threading.main_thread():
            self._core.publish_synchronous(topic, payload)
        else:
            self._message_emitted.emit(topic, payload)

    def send_message(self, message: Any):
        topic = getattr(message, "TOPIC_ID", None)
        if topic is None:
            topic = type(message).__name__
        self.publish(topic, message)

    @pyqtSlot(str, object)
    def _handle_emitted_signal(self, topic: str, payload: Any):
        self._core.publish_synchronous(topic, payload)