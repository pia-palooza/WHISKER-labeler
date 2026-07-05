from whisker.gui.tabs import (
    WelcomeTab,
    ProjectsTab,
    InfoTab,
    JobsTab,
)

from whisker.gui.workflows.pose_estimation.tabs import (
    LabelingPosesTab,
)
from whisker.gui.workflows.behavior_classification.tabs import (
    LabelingBehaviorsTab,
)


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
        else:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        obj.bridge = bridge
        return obj

# ML views removed