import logging
import threading

from .node import Node

class _Task:
    def __init__(self, uuid: str):
        self._uuid = uuid
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    ############################################################################

    def run(self):
        with self._lock:
            if self.is_running():
                return
            
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._entry_point, daemon=True)
            self._thread.start()

    def wait_for_shutdown(self, timeout: float = 2.0):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
    
        if self._thread is not None and self._thread.is_alive():
            raise TimeoutError(f"Task {self._uuid} did not shut down within the specified timeout.")

    def kill(self):
        with self._lock:
            if self._thread is not None:
                self._stop_event.set()
                self._thread = None
    
    def terminate(self, timeout: float = 2.0):
        try:
            self.wait_for_shutdown(timeout=timeout)
        except TimeoutError:
            self.kill()

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _entry_point(self):
        """Override this method in subclasses to implement the main loop of the task."""
        logging.warning(f"Task {self._uuid} has no entry point defined.")

class NodeTask(_Task):

    def __init__(self, node: Node, refresh_rate: float = 0.1):
        super().__init__(node.uuid)
        self._node = node
        self._refresh_rate = refresh_rate
    
    @property
    def node(self) -> Node:
        return self._node

    def _entry_point(self):
        try:
            self._node.setup()
            while not self._stop_event.wait(timeout=self._refresh_rate) and not self._node.received_shutdown:
                self._node.wakeup()
        except Exception as e:
            logging.exception(
                f"Unhandled exception in NodeTask for node {self._node.uuid}: {repr(e)}"
            )

        finally:
            try:
                self._node.shutdown()
            except Exception as e:
                logging.exception(
                    f"Unhandled exception during shutdown of NodeTask for node {self._node.uuid}: {repr(e)}"
                )
