import logging
import traceback
import fnmatch
from typing import Callable, Dict, List

class BaseMessageBus:
    """
    The core synchronous Pub/Sub engine as a pure-Python singleton.
    Completely decoupled from any UI framework.
    """
    _instance = None

    def __init__(self):
        # Guard against accidental direct re-instantiation breaking the singleton
        if BaseMessageBus._instance is not None:
            raise RuntimeWarning("BaseMessageBus is a singleton. Use BaseMessageBus.get() instead.")
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._enable_debug_output = False

    @classmethod
    def get(cls) -> "BaseMessageBus":
        if cls._instance is None:
            cls._instance = BaseMessageBus()
        return cls._instance

    def subscribe(self, topic_pattern: str, callback: Callable):
        if topic_pattern not in self._subscriptions:
            self._subscriptions[topic_pattern] = []
        if callback not in self._subscriptions[topic_pattern]:
            self._subscriptions[topic_pattern].append(callback)

    def unsubscribe(self, topic_pattern: str, callback: Callable):
        if topic_pattern in self._subscriptions and callback in self._subscriptions[topic_pattern]:
            self._subscriptions[topic_pattern].remove(callback)

    def publish_synchronous(self, topic: str, payload: dict):
        if self._enable_debug_output:
            logging.info(f"[BaseMessageBus] Publishing to topic: {topic}")
            logging.debug(f"Payload: {payload}")

        for pattern, callbacks in list(self._subscriptions.items()):
            if fnmatch.fnmatch(topic, pattern):
                for callback in list(callbacks):
                    try:
                        callback(topic, payload)
                    except Exception as e:
                        logging.error(f"Error in callback for topic '{topic}': {e}\n{traceback.format_exc()}")

    def enable_debug_output(self, enable: bool):
        self._enable_debug_output = enable

