# START_DIFF: whisker/gui/tabs/main_tabs.py [Bubble up signals from WelcomeTab]
import enum
import logging
from typing import cast, Dict, Optional, List

from PyQt6.QtCore import pyqtSignal, QThreadPool, QSettings
from PyQt6.QtWidgets import QTabWidget, QMainWindow, QLabel

from whisker.gui.job_manager import JobManager
from whisker.core.workspace import Project, Workspace
from whisker.gui.widgets import data_explorer
from whisker.gui.tabs import (
    BaseTab,
    WelcomeTab,
    ProjectsTab,
    InfoTab,
    LabelingTab,
    TrainingTab,
    PredictTab,
    AmendTab,
    EvaluationTab,
    FigureMakerTab,
    JobsTab,
    VerifyTab,
    HPOTab,
    PipelineDebugTab
)
from whisker.gui.workflows.behavior_classification.tabs import (
    DiscoveryBehaviorTab
)
from whisker.base.core_nodes.queue_router import QueueRouterNode
from whisker.base.task import NodeTask
from whisker.gui.workflows.workflow_factory import get_workflow_info_item_handlers
from whisker.gui.worker_wrapper import Worker
from whisker.gui.signals import MessageBus
from whisker.core.workflows.workflow_enum import Workflow


class TabEnum(str, enum.Enum):
    WELCOME = "Welcome"
    PROJECTS = "Projects"
    INFO = "Info"
    LABEL = "Label"
    JOBS = "Jobs"
    TRAIN = "Train"
    PREDICT = "Predict"
    AMEND = "Amend"
    EVAL = "Evaluate"
    VERIFY = "Verify"
    HPO = "HPO"
    DEBUG = "Debug"
    DISCOVER = "Discover"
    FIGURE_MAKER = "Figure Maker"


class MainTabs(BaseTab):
    tab_changed = pyqtSignal(str)
    # DEV_NOTE: This new signal fires whenever the overall dirty state of
    # any tab might have changed.
    any_tab_dirty_state_changed = pyqtSignal()

    def __init__(self, parent: QMainWindow):
        """
        Initializes and holds all the main application tabs.
        """
        super().__init__(parent)
        self.tabs_widget = QTabWidget()
        self.tabs_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs_widget.setMovable(True)

        # Instantiate all the tabs
        self._tabs: Dict[str, BaseTab] = {}
        self._tabs[TabEnum.WELCOME] = WelcomeTab(parent)
        self._tabs[TabEnum.PROJECTS] = ProjectsTab(parent)
        self._tabs[TabEnum.INFO] = InfoTab(
            parent,
            workflow_item_handlers=get_workflow_info_item_handlers()
        )
        self._tabs[TabEnum.JOBS] = JobsTab(parent)
        # Workflow-based tabs
        self._tabs[TabEnum.LABEL] = LabelingTab(parent)
        self._tabs[TabEnum.TRAIN] = TrainingTab(parent)
        self._tabs[TabEnum.PREDICT] = PredictTab(parent)
        self._tabs[TabEnum.AMEND] = AmendTab(parent)
        self._tabs[TabEnum.EVAL] = EvaluationTab(parent)
        self._tabs[TabEnum.VERIFY] = VerifyTab(parent)
        self._tabs[TabEnum.HPO] = HPOTab(parent)
        self._tabs[TabEnum.DISCOVER] = DiscoveryBehaviorTab(parent)
        self._tabs[TabEnum.FIGURE_MAKER] = FigureMakerTab(parent)
        self._tabs[TabEnum.DEBUG] = PipelineDebugTab(parent)
        for tab in self._tabs.values():
            tab.request_launch_worker.connect(self._on_request_launch_worker)

        # Add tabs to the widget and connect their dirty signals
        self._tab_names_by_index = {}
        self._tab_names_by_index_by_name = {}
        for tab_name, tab in self.tabs.items():
            tab_index = self.tabs_widget.addTab(tab, tab_name)
            self._tab_names_by_index[tab_index] = tab_name
            self._tab_names_by_index_by_name[tab_name] = tab_index
            tab.dirty_state_changed.connect(self.any_tab_dirty_state_changed)

        self.tabs_widget.currentChanged.connect(self._on_main_tab_changed)

        # --- Bubble up signals from WelcomeTab ---
        welcome_tab = cast(WelcomeTab, self._tabs[TabEnum.WELCOME])
        self.request_set_workspace = welcome_tab.request_set_workspace
        self.request_create_project = welcome_tab.request_create_project
        self.request_create_dataset = welcome_tab.request_create_dataset
        self.request_import_labels = welcome_tab.request_import_labels
        # --- End signal bubbling ---

    @property
    def tabs(self) -> Dict[str, BaseTab]:
        return self._tabs

    @property
    def current_tab_enum(self) -> TabEnum:
        return TabEnum(list(self.tabs.keys())[self.tabs_widget.currentIndex()])

    @property
    def current_tab(self) -> BaseTab:
        return self.tabs[self.current_tab_enum]

    def get_dirty_tabs(self) -> List[BaseTab]:
        """Returns a list of all tabs that currently have unsaved changes."""
        return [tab for tab in self.tabs.values() if tab.is_dirty()]

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        self.current_tab.on_data_explorer_item_selected(selection)

    def _on_main_tab_changed(self, current_tab_index: int):
        if self._tab_names_by_index.get(current_tab_index) not in [
            TabEnum.PREDICT,
            TabEnum.AMEND,
            TabEnum.EVAL,
            TabEnum.DEBUG
        ]:
            MessageBus.get().publish("selection/model_run/changed", {"name": ""})

        if self.tabs_widget.widget(current_tab_index):
            self.tab_changed.emit(self.current_tab_enum)

    def _on_request_launch_worker(self, worker_name: str, worker: Worker):
        """Delegates worker launching to the JobManager."""
        JobManager.get().submit_worker(worker_name, worker)
        logging.info(f"Worker {worker_name} submitted to JobManager.")

    def set_workspace(self, workspace: Workspace | None):
        super().set_workspace(workspace)
        for tab in self.tabs.values():
            tab.set_workspace(workspace)
            tab.set_dirty(False)

    def on_project_changed(self, project: Optional[Project]):
        """
        Slot to receive the new active project and pass it to all child tabs.
        """
        self.set_project(project)
        for tab in self.tabs.values():
            tab.set_project(project)
            tab.set_dirty(False)

    def set_active_workflow(self, workflow: Workflow):
        # Optimization: Disable updates during the potentially heavy tab switching
        # and visibility toggling.
        self.tabs_widget.setUpdatesEnabled(False)
        try:
            for tab in self.tabs.values():
                tab.set_active_workflow(workflow)

            match workflow:
                case Workflow.POSE_ESTIMATION:
                    self._set_tab_visible(TabEnum.AMEND, True)
                    self._set_tab_visible(TabEnum.VERIFY, True)
                    self._set_tab_visible(TabEnum.DISCOVER, False)
                    self._set_tab_visible(TabEnum.DEBUG, True)
                case Workflow.BEHAVIOR_CLASSIFICATION:
                    self._set_tab_visible(TabEnum.AMEND, True)
                    self._set_tab_visible(TabEnum.VERIFY, True)
                    self._set_tab_visible(TabEnum.DISCOVER, True)
                    self._set_tab_visible(TabEnum.DEBUG, True)
        finally:
            self.tabs_widget.setUpdatesEnabled(True)
            self.tabs_widget.update()
    
    def _set_tab_visible(self, tab_name: TabEnum, visible: bool):
        if tab_name not in self._tab_names_by_index_by_name:
            logging.warning(f"Attempted to set visibility of unknown tab: {tab_name}")
            return
        self.tabs_widget.setTabVisible(self._tab_names_by_index_by_name[tab_name], visible)
    
    def refresh_ui(self, settings: QSettings):
        enable_money_mode = settings.value("enable_money_mode", False, type=bool)
        for tab in self.tabs.values():
            if hasattr(tab, "apply_money_mode"):
                tab.apply_money_mode(enable_money_mode)