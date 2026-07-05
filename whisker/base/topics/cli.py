import dataclasses
from enum import Enum
from ..messaging import register_topic

class Request(str, Enum):
    TOGGLE_ENABLE = "cli/requests/toggle_enable"

class Reply(str, Enum):
    ENABLE_STATE_TOGGLED = "cli/reply/enable_state_toggled"

class Telemetry(str, Enum):
    USER_INPUT = "cli/telemetry/user_input"

@register_topic(Request.TOGGLE_ENABLE)
@dataclasses.dataclass(slots=True)
class ToggleEnableRequest:
    enable: bool = True

@register_topic(Reply.ENABLE_STATE_TOGGLED)
@dataclasses.dataclass(slots=True)
class EnableStateToggledReply:
    enabled: bool

@register_topic(Telemetry.USER_INPUT)
@dataclasses.dataclass(slots=True)
class UserInputTelemetry:
    content: str
