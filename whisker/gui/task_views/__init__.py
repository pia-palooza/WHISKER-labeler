from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout

from whisker.gui.tabs import (
    WelcomeTab,
    ProjectsTab,
    InfoTab,
    JobsTab,
)
from whisker.gui.tabs.base_tab import BaseTab

from whisker.gui.workflows.pose_estimation.tabs import (
    LabelingPosesTab,
    EvaluationPoseTab,
)
from whisker.gui.workflows.behavior_classification.tabs import (
    LabelingBehaviorsTab,
)
from whisker.gui.workflows.roi_tracking.tabs.labeling import RoiTrackingLabelingTab


class _NotAvailableTab(BaseTab):
    """A plain "not available yet" placeholder for a workflow-scoped task that only some
    workflows implement (e.g. Evaluate, currently Pose Estimation only)."""
    def __init__(self, message: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)


class WelcomeView(WelcomeTab):
    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge

class ProjectsView(ProjectsTab):
    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge

class InfoView(InfoTab):
    def __init__(self, bridge, parent=None):
        from whisker.gui.workflows.workflow_factory import get_workflow_info_item_handlers
        super().__init__(parent, workflow_item_handlers=get_workflow_info_item_handlers())
        self.bridge = bridge

class JobsView(JobsTab):
    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge

class LabelView:
    def __new__(cls, bridge, workflow_name, parent=None):
        if workflow_name == "Pose Estimation":
            obj = LabelingPosesTab(parent)
        elif workflow_name == "Behavior Classification":
            obj = LabelingBehaviorsTab(parent)
        elif workflow_name == "ROI Tracking":
            obj = RoiTrackingLabelingTab(parent)
        else:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        obj.bridge = bridge
        return obj

class EvaluateView:
    def __new__(cls, workflow_name, parent=None):
        if workflow_name == "Pose Estimation":
            return EvaluationPoseTab(parent)
        return _NotAvailableTab("Evaluate isn't available for this workflow yet.", parent)

# ML views removed