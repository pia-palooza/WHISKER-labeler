import dataclasses
import logging
from enum import Enum

from ..messaging import register_topic

PREFIX = "node"
PREFIX_REQUEST = f"{PREFIX}/request"
PREFIX_REPLY = f"{PREFIX}/reply"
PREFIX_TELEMETRY = f"{PREFIX}/telemetry"

class NodeState(str, Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    SHUTDOWN = "SHUTDOWN"
    STOPPED = "STOPPED"

class Request(str, Enum):
    PING = f"{PREFIX_REQUEST}/ping"
    SHUTDOWN = f"{PREFIX_REQUEST}/shutdown"

class Reply(str, Enum):
    PONG = f"{PREFIX_REPLY}/pong"
    SHUTDOWN_ACK = f"{PREFIX_REPLY}/shutdown_ack"

class Telemetry(str, Enum):
    STATE = f"{PREFIX_TELEMETRY}/state"
    HEARTBEAT = f"{PREFIX_TELEMETRY}/heartbeat"

@register_topic(Request.PING)
@dataclasses.dataclass(slots=True)
class PingRequest:
    pass

@register_topic(Request.SHUTDOWN)
@dataclasses.dataclass(slots=True)
class ShutdownRequest:
    pass

@register_topic(Reply.PONG)
@dataclasses.dataclass(slots=True)
class PongReply:
    pass

@register_topic(Reply.SHUTDOWN_ACK)
@dataclasses.dataclass(slots=True)
class ShutdownAck:
    pass

@register_topic(Telemetry.STATE)
@dataclasses.dataclass(slots=True)
class StateTelemetry:
    state: NodeState
