import dataclasses
from typing import Any, Callable, Dict, List, Optional

from ..node import Node
from ..messaging import Message

@dataclasses.dataclass
class CallbackRegistration:
    callback: Callable[[Message], Any]
    sender_id: str = ""

    def __getitem__(self, index):
        if index == 0:
            return self.callback
        elif index == 1:
            return self.sender_id
        raise IndexError("tuple index out of range")

    def __iter__(self):
        yield self.callback
        yield self.sender_id

    def __len__(self):
        return 2

class CallbackObserverNode(Node):
    """
    A Node that invokes registered callbacks when it receives messages on specific topics.

    The mapping of topics to callbacks can be modified dynamically at runtime.
    """
    def __init__(
        self,
        label: str,
        callbacks: Dict[str, Any] | None = None
    ):
        super().__init__(label=label)
        self._callbacks: Dict[str, List[CallbackRegistration]] = {}

        if callbacks:
            self.set_callbacks(callbacks)

    def register_callback(
        self,
        topic: str,
        callback: CallbackRegistration | Callable[[Message], Any] | tuple[Callable[[Message], Any], str],
        sender_id: Optional[str] = None,
    ) -> None:
        """Registers a callback for a specific message topic and optional sender ID."""
        if topic not in self._callbacks:
            self._callbacks[topic] = []
            self.add_subscription(topic)

        if isinstance(callback, CallbackRegistration):
            registration = callback
        elif isinstance(callback, tuple):
            registration = CallbackRegistration(callback=callback[0], sender_id=callback[1])
        else:
            registration = CallbackRegistration(callback=callback, sender_id=sender_id or "")

        self._callbacks[topic].append(registration)

    def unregister_callback(
        self,
        topic: str,
        callback: Callable[[Message], Any],
        sender_id: Optional[str] = None
    ) -> None:
        """Unregisters a specific callback (and optional sender ID) for a topic."""
        if topic in self._callbacks:
            try:
                to_remove = None
                for reg in self._callbacks[topic]:
                    if reg.callback == callback and (sender_id is None or reg.sender_id == (sender_id or "")):
                        to_remove = reg
                        break
                if to_remove:
                    self._callbacks[topic].remove(to_remove)
                else:
                    raise ValueError("Not found")
            except ValueError:
                self.logger.warning(
                    f"Callback registration {(callback, sender_id)} not found for topic {topic}"
                )

            if not self._callbacks[topic]:
                del self._callbacks[topic]
                self.remove_subscription(topic)

    def set_callbacks(
        self,
        callbacks: Dict[str, Any]
    ) -> None:
        """
        Replaces the current callback mappings.
        Handles single callbacks, (callback, sender_id) tuples, or lists of either, or CallbackRegistration.
        """
        self.clear_callbacks()
        for topic, value in callbacks.items():
            if isinstance(value, list):
                for item in value:
                    self.register_callback(topic, item)
            else:
                self.register_callback(topic, value)

    def clear_callbacks(self, topic: str | None = None) -> None:
        """
        Clears registered callbacks.
        If topic is specified, only callbacks for that topic are cleared.
        Otherwise, all callbacks are cleared.
        """
        if topic is not None:
            if topic in self._callbacks:
                del self._callbacks[topic]
                self.remove_subscription(topic)
        else:
            # Clear all subscriptions that were added for these callbacks
            topics_to_remove = list(self._callbacks.keys())
            self._callbacks.clear()
            self.remove_subscriptions(set(topics_to_remove))

    def handle_message(self, message: Message) -> bool:
        # Give parent Node a chance to process system commands (PING, SHUTDOWN, etc.)
        if super().handle_message(message):
            return True

        topic = message.header.topic
        cb_list = self._callbacks.get(topic)
        if cb_list:
            for registration in cb_list:
                if registration.sender_id and message.header.sender_id != registration.sender_id:
                    continue
                try:
                    registration.callback(message)
                except Exception as e:
                    self.logger.exception(
                        f"Error executing callback for topic '{topic}' in {self.uuid}: {e}"
                    )
            return True

        return False
