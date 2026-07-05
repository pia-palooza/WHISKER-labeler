import dataclasses
from enum import Enum
from typing import Any

from whisker.base.messaging import register_topic

PREFIX = "gui"
PREFIX_REQUEST = f"{PREFIX}/request"
PREFIX_TELEMETRY = f"{PREFIX}/telemetry"

class Request(str, Enum):
    SWITCH_VIEW = f"{PREFIX_REQUEST}/switch_view"
    SELECT_TREE_ITEM = f"{PREFIX_REQUEST}/select_tree_item"
    CONTEXT_MENU_ACTION = f"{PREFIX_REQUEST}/context_menu_action"
    REGISTER_VIEW = f"{PREFIX_REQUEST}/register_view"
    NEW_WORKSPACE = f"{PREFIX_REQUEST}/new_workspace"
    OPEN_WORKSPACE = f"{PREFIX_REQUEST}/open_workspace"
    CLOSE_WORKSPACE = f"{PREFIX_REQUEST}/close_workspace"
    OPEN_THEMES_DIALOG = f"{PREFIX_REQUEST}/open_themes_dialog"
    CLOSE_MAIN_WINDOW = f"{PREFIX_REQUEST}/close_main_window"
    CREATE_PROJECT = f"{PREFIX_REQUEST}/create_project"
    SET_BLIND_MODE = f"{PREFIX_REQUEST}/set_blind_mode"
    SET_FILTER_OPTION = f"{PREFIX_REQUEST}/set_filter_option"



class Telemetry(str, Enum):
    WORKSPACE_CHANGED = f"{PREFIX_REQUEST}/workspace_changed"
    WORKFLOW_SELECTED = f"{PREFIX_REQUEST}/workflow_selected"
    STUDY_ITEM_SELECTED = f"{PREFIX_REQUEST}/study/item_selected"

@register_topic(Request.REGISTER_VIEW)
@dataclasses.dataclass(slots=True)
class RegisterViewRequest:
    view_name: str
    widget: object

@register_topic(Request.SWITCH_VIEW)
@dataclasses.dataclass(slots=True)
class SwitchViewRequest:
    view_name: str

@register_topic(Request.SELECT_TREE_ITEM)
@dataclasses.dataclass(slots=True)
class SelectTreeItemRequest:
    item_tree: str
    item_path: list[str]

@register_topic(Request.CONTEXT_MENU_ACTION)
@dataclasses.dataclass(slots=True)
class ContextMenuActionRequest:
    action_name: str
    item_path: str

@register_topic(Request.OPEN_THEMES_DIALOG)
@dataclasses.dataclass(slots=True)
class OpenThemesDialogRequest:
    pass

@register_topic(Request.CLOSE_MAIN_WINDOW)
@dataclasses.dataclass(slots=True)
class CloseMainWindowRequest:
    pass

@register_topic(Request.NEW_WORKSPACE)
@dataclasses.dataclass(slots=True)
class NewWorkspaceRequest:
    pass

@register_topic(Request.OPEN_WORKSPACE)
@dataclasses.dataclass(slots=True)
class OpenWorkspaceRequest:
    path: str | None = None

@register_topic(Request.CLOSE_WORKSPACE)
@dataclasses.dataclass(slots=True)
class CloseWorkspaceRequest:
    pass

@register_topic(Telemetry.WORKSPACE_CHANGED)
@dataclasses.dataclass(slots=True)
class WorkspaceChangedTelemetry:
    path: str | None
    workspace: Any

@register_topic(Telemetry.WORKFLOW_SELECTED)
@dataclasses.dataclass(slots=True)
class WorkflowSelectedTelemetry:
    name: str

@register_topic(Telemetry.STUDY_ITEM_SELECTED)
@dataclasses.dataclass(slots=True)
class StudyItemSelectedTelemetry:
    item_name: str
    item_value: str


@register_topic(Request.SET_BLIND_MODE)
@dataclasses.dataclass(slots=True)
class SetBlindModeRequest:
    enabled: bool


@register_topic(Request.SET_FILTER_OPTION)
@dataclasses.dataclass(slots=True)
class SetFilterOptionRequest:
    option: str


