import uuid
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List
from collections import deque

from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool

from whisker.base.core_nodes.queue_router import QueueRouterNode
from whisker.base.task import NodeTask
from whisker.gui.worker_wrapper import Worker

class JobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    FINISHED = "Finished"
    ERROR = "Error"
    CANCELLED = "Cancelled"

@dataclass
class JobRecord:
    id: str
    name: str
    worker: Worker
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    message: str = ""

    @property
    def start_time_display(self) -> str:
        """Returns start time in AM/PM format."""
        return time.strftime("%I:%M:%S %p", time.localtime(self.start_time))

    @property
    def runtime_display(self) -> str:
        """Returns formatted runtime duration (HH:MM:SS)."""
        ref_time = self.end_time if self.end_time else time.time()
        duration = int(ref_time - self.start_time)
        
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

class JobManager(QObject):
    """
    Singleton registry for background jobs.
    Serializes execution for jobs with identical names using internal deques.
    """
    _instance = None

    # Signals for UI updates
    job_added = pyqtSignal(str)          # job_id
    job_updated = pyqtSignal(str)        # job_id
    job_removed = pyqtSignal(str)        # job_id

    def __init__(self):
        super().__init__()
        self._jobs: Dict[str, JobRecord] = {}
        self._queues: Dict[str, deque[str]] = {}  # {name: deque([job_id, ...])}
        self._thread_pool = QThreadPool()
        self._bus = QueueRouterNode("JobManagerQueueRouter")
        self._bus_task = NodeTask(self._bus)
        self._bus_task.run()
        
        # Reserve one thread for GUI/overhead if max > 1
        max_threads = self._thread_pool.maxThreadCount()
        self._thread_pool.setMaxThreadCount(max(1, max_threads - 1))

    @classmethod
    def get(cls) -> "JobManager":
        if cls._instance is None:
            cls._instance = JobManager()
        return cls._instance

    def submit_worker(self, name: str, worker: Worker) -> str:
        """Registers a worker and serializes it based on name."""
        self._bus.register_node(worker.job)
        self._bus.register_node(worker.callback_node)
        job_id = worker.job.uuid
        worker.set_id(job_id)
        
        record = JobRecord(id=job_id, name=name, worker=worker)
        self._jobs[job_id] = record
        
        # Wire up signals
        worker.signals.started.connect(lambda: self._on_job_started(job_id))
        worker.signals.progress.connect(lambda msg, p: self._on_job_progress(job_id, msg, p))
        worker.signals.finished.connect(lambda res: self._on_job_finished(job_id))
        worker.signals.error.connect(lambda err: self._on_job_error(job_id, err))
        
        # Queue management
        if name not in self._queues:
            self._queues[name] = deque()
        
        self._queues[name].append(job_id)
        self.job_added.emit(job_id)
        
        # Only launch if it's the head of its specific queue
        if len(self._queues[name]) == 1:
            self._thread_pool.start(worker)
            logging.info(f"Job started: {name} ({job_id})")
        else:
            logging.info(f"Job queued (serialized): {name} ({job_id})")
            
        return job_id

    def cancel_job(self, job_id: str):
        if record := self._jobs.get(job_id):
            is_pending = record.status == JobStatus.QUEUED
            
            if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                logging.info(f"Cancelling job: {record.name}")
                record.worker.cancel()
                record.status = JobStatus.CANCELLED
                record.end_time = time.time()
                record.message = "Cancelled by user"
                self.job_updated.emit(job_id)

            # If it was stuck in a deque and not yet running, remove it and advance
            if is_pending:
                self._advance_queue(record.name, job_id)

    def clear_finished_jobs(self):
        """Removes all non-active jobs from the registry."""
        to_remove = [
            jid for jid, rec in self._jobs.items() 
            if rec.status in (JobStatus.FINISHED, JobStatus.ERROR, JobStatus.CANCELLED)
        ]
        for jid in to_remove:
            del self._jobs[jid]
            self.job_removed.emit(jid)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[JobRecord]:
        return list(self._jobs.values())

    # --- Internal Queue Control ---

    def _advance_queue(self, name: str, finished_id: str):
        """Removes the finished/cancelled ID and starts the next available job for that name."""
        if name not in self._queues:
            return

        queue = self._queues[name]
        
        # Standard flow: the finished job is the one that was running (at the left)
        if queue and queue[0] == finished_id:
            queue.popleft()
        elif finished_id in queue:
            # Handle removal of a job that was cancelled while still waiting in line
            queue.remove(finished_id)

        # Start the next job in this specific queue
        if queue:
            next_id = queue[0]
            if next_record := self._jobs.get(next_id):
                self._thread_pool.start(next_record.worker)
        else:
            del self._queues[name]

    # --- Signal Handlers ---

    def _on_job_started(self, job_id: str):
        if record := self._jobs.get(job_id):
            record.status = JobStatus.RUNNING
            self.job_updated.emit(job_id)

    def _on_job_progress(self, job_id: str, message: str, percent: int):
        if record := self._jobs.get(job_id):
            record.progress = percent
            record.message = message
            self.job_updated.emit(job_id)

    def _on_job_finished(self, job_id: str):
        record = self._jobs.get(job_id)
        if record:
            self._bus.unregister_node(record.worker.job.uuid)
            self._bus.unregister_node(record.worker.callback_node.uuid)
            if record.status != JobStatus.CANCELLED:
                record.status = JobStatus.FINISHED
                record.end_time = time.time()
                record.progress = 100
                record.message = "Done"
            self.job_updated.emit(job_id)
            self._advance_queue(record.name, job_id)

    def _on_job_error(self, job_id: str, error: str):
        record = self._jobs.get(job_id)
        if record:
            self._bus.unregister_node(record.worker.job.uuid)
            self._bus.unregister_node(record.worker.callback_node.uuid)
            record.status = JobStatus.ERROR
            record.end_time = time.time()
            record.message = f"Error: {error}"
            self.job_updated.emit(job_id)
            self._advance_queue(record.name, job_id)