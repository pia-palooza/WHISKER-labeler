import enum
import dataclasses
from ..messaging import register_topic

class ErrorCodes(enum.IntEnum):
    INVALID_NODE_UUID = -1

class Request(str, enum.Enum):
    NODE_LIST = "queue_router/request/node_list"
    NODE_INFO = "queue_router/request/node_info"

class Reply(str, enum.Enum):
    NODE_LIST = "queue_router/reply/node_list"
    NODE_INFO = "queue_router/reply/node_info"
    ERROR = "queue_router/reply/error"

class Telemetry(str, enum.Enum):
    NODE_REGISTERED = "queue_router/telemetry/node_registered"

@register_topic(Reply.ERROR)
@dataclasses.dataclass(slots=True)
class ErrorResponse:
    failed_topic: str
    error_code: int
    message: str

@register_topic(Request.NODE_LIST)
@dataclasses.dataclass(slots=True)
class NodeListRequest:
    pass # DEV_NOTE: Empty dataclass acts as a strict typing token for the request

@register_topic(Reply.NODE_LIST)
@dataclasses.dataclass(slots=True)
class NodeListResponse:
    nodes: list[str]

@register_topic(Request.NODE_INFO)
@dataclasses.dataclass(slots=True)
class NodeInfoRequest:
    uuid: str

@register_topic(Reply.NODE_INFO)
@dataclasses.dataclass(slots=True)
class NodeInfoResponse:
    uuid: str
    label: str
    subscriptions: frozenset[str]

@register_topic(Telemetry.NODE_REGISTERED)
@dataclasses.dataclass(slots=True)
class NodeRegisteredTelemetry:
    uuid: str
    label: str
    subscriptions: frozenset[str]
