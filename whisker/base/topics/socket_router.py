import dataclasses
from enum import Enum
from ..messaging import register_topic

class Request(str, Enum):
    BIND = "socket_router/request/bind"
    CONNECT = "socket_router/request/connect"
    FORWARD_START = "socket_router/request/forward_start"

class Reply(str, Enum):
    BIND_SUCCESS = "socket_router/reply/bind_success"
    BIND_FAILURE = "socket_router/reply/bind_failure"
    CONNECT_SUCCESS = "socket_router/reply/connect_success"
    CONNECT_FAILURE = "socket_router/reply/connect_failure"
    FORWARDING_STARTED = "socket_router/reply/forwarding_started"

@register_topic(Request.BIND)
@dataclasses.dataclass(slots=True)
class BindRequest:
    address: str

@register_topic(Request.CONNECT)
@dataclasses.dataclass(slots=True)
class ConnectRequest:
    address: str

@register_topic(Request.FORWARD_START)
@dataclasses.dataclass(slots=True)
class StartForwardingRequest:
    patterns: list[str]

@register_topic(Reply.BIND_SUCCESS)
@dataclasses.dataclass(slots=True)
class BindSocketSuccessReply:
    address: str

@register_topic(Reply.BIND_FAILURE)
@dataclasses.dataclass(slots=True)
class BindSocketFailureReply:
    address: str
    error: str

@register_topic(Reply.CONNECT_SUCCESS)
@dataclasses.dataclass(slots=True)
class ConnectSocketSuccessReply:
    address: str

@register_topic(Reply.CONNECT_FAILURE)
@dataclasses.dataclass(slots=True)
class ConnectSocketFailureReply:
    address: str
    error: str

@register_topic(Reply.FORWARDING_STARTED)
@dataclasses.dataclass(slots=True)
class ForwardingStartedReply:
    patterns: list[str]
