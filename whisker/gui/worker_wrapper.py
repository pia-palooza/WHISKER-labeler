import logging
import traceback
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from whisker.base.job import BaseJob
from whisker.base.core_nodes.callback_observer import CallbackObserverNode, CallbackRegistration
from whisker.base.task import NodeTask
from whisker.base import topics

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    started = pyqtSignal()
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int)
    data_update = pyqtSignal(object)

class Worker(QRunnable):
    """
    A generic worker thread that wraps a Core Job class.
    """
    def __init__(self, job: BaseJob):
        super().__init__()
        self.signals = WorkerSignals()
        self.job = job
        self.callback_node = CallbackObserverNode(
            label=f"{job.__class__.__name__}.CallbackObserverNode",
            callbacks={
                topics.job.Telemetry.PROGRESS_UPDATE.value: [
                    CallbackRegistration(
                        lambda message: self.signals.progress.emit(
                            message.payload.message, message.payload.percent
                        ),
                        sender_id=self.job.uuid
                    )
                ],
                topics.job.Telemetry.DATA_REPORT.value: [
                    CallbackRegistration(
                        lambda message: self.signals.data_update.emit(
                            message.payload.data
                        ),
                        sender_id=self.job.uuid
                    )
                ]
            }
        )
        self.callback_task = NodeTask(self.callback_node)
        self._id = None

    def set_id(self, worker_id: str):
        self._id = worker_id

    @pyqtSlot()
    def run(self):
        self.callback_task.run()
        self.signals.started.emit()
        try:
            result = self.job.run()
            self.signals.finished.emit(result)
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Worker job failed: {traceback.format_exc()}")
            self.signals.error.emit(error_msg)
        self.callback_task.terminate()

    def cancel(self):
        if hasattr(self.job, 'cancel'):
            self.job.cancel()