import abc
from typing import Callable, Optional, Any

from .node import Node
from . import topics

class BaseJob(Node):
    """
    Abstract base class for long-running jobs in the Core layer.
    """
    def __init__(
        self, 
        label: str | None = None,
        **kwargs
    ):
        super().__init__(label or self.__class__.__name__, **kwargs)

    def cancel(self):
        """Flags the job to stop processing at the next opportunity."""
        self.request_shutdown()

    @property
    def is_cancelled(self) -> bool:
        return self.received_shutdown

    def report_progress(self, message: str, percent: int):        
        self.send_outgoing_message(
            payload=topics.job.ProgressUpdateTelemetry(message=message, percent=percent)
        )

    def report_data(self, data: Any):
        """Emits a rich data payload (e.g. metrics, images) to the listener."""        
        self.send_outgoing_message(
            payload=topics.job.DataReportTelemetry(data=data)
        )

    @abc.abstractmethod
    def run(self) -> str | dict | None:
        """
        Execute the job logic.
        Returns:
            Result data (type depends on implementation) or None.
        Raises:
            Exception: Any error encountered during execution.
        """
        pass