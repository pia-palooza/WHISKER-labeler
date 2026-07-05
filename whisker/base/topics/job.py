import dataclasses
from enum import Enum
from typing import Any

from ..messaging import register_topic

PREFIX = "job"
PREFIX_REQUEST = f"{PREFIX}/request"
PREFIX_REPLY = f"{PREFIX}/reply"
PREFIX_TELEMETRY = f"{PREFIX}/telemetry"

class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    FINISHED = "Finished"
    ERROR = "Error"
    CANCELLED = "Cancelled"
    REMOVED = "Removed"

class Request(str, Enum):
    RUN_JOB = f"{PREFIX_REQUEST}/run_job"
    CANCEL_JOB = f"{PREFIX_REQUEST}/cancel_job"

class Telemetry(str, Enum):
    JOB_STATUS = f"{PREFIX_TELEMETRY}/job_status"
    PROGRESS_UPDATE = f"{PREFIX_TELEMETRY}/progress_update"
    DATA_REPORT = f"{PREFIX_TELEMETRY}/data_report"

@register_topic(Request.RUN_JOB)
@dataclasses.dataclass(slots=True)
class RunJobRequest:
    job_name: str
    job_params: Any

@register_topic(Request.CANCEL_JOB)
class CancelJobRequest:
    job_id: str

@register_topic(Telemetry.JOB_STATUS)
@dataclasses.dataclass(slots=True)
class JobStatusTelemetry:
    job_id: str
    status: JobStatus
    message: str = ""

@register_topic(Telemetry.PROGRESS_UPDATE)
@dataclasses.dataclass(slots=True)
class ProgressUpdateTelemetry:
    message: str
    percent: int

@register_topic(Telemetry.DATA_REPORT)
@dataclasses.dataclass(slots=True)
class DataReportTelemetry:
    data: Any
