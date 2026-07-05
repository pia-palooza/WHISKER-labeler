import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QButtonGroup
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize

from ...core.workflows.workflow_enum import Workflow
from ..base.collapsible_panel import VerticalCollapsiblePanel
from ..base.panel_section import PanelSection
from ..topics import gui as gui
from ..constants import ASSETS_DIR
from ..task_views import (
    WelcomeView, ProjectsView, InfoView, JobsView,
    LabelView
)

TASKS_LAYOUT_SCHEMA = (
    ("global", "❖ Welcome", lambda bridge, _: WelcomeView(bridge)),
    ("global", "❖ Projects", lambda bridge, _: ProjectsView(bridge)),
    ("global", "❖ Info", lambda bridge, _: InfoView(bridge)),
    ("global", "❖ Jobs", lambda bridge, _: JobsView(bridge)),
    ("workflow", "❖ Label", lambda bridge, wf: LabelView(bridge, wf)),
)

WORKFLOW_COLORS = [ "orange", "blue", "magenta", "green" ]

class NavigationPanel(VerticalCollapsiblePanel):
    """Encapsulates the left sidebar navigation items and workflow controls."""
    def __init__(self, bridge, settings, parent=None):
        self.title_label = QLabel("NAVIGATION")
        self.title_label.setObjectName("HeaderLabel")
        
        super().__init__(self.title_label, parent=parent)
        self.bridge = bridge
        self.settings = settings
        self._current_workflow = None
        
        # Minimize margins
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        
        self.workflow_group = QButtonGroup(self)
        self.task_group = QButtonGroup(self)
        
        self._setup_contents()
        
        # Set default expanded size to minimum size hint needed for text
        self.expanded_size = self.sizeHint().width()
        self.setFixedWidth(self.expanded_size)

    def _create_local_nav_button(self, name):
        btn = QPushButton(name)
        btn.setObjectName("MenuButton")
        btn.clicked.connect(lambda: self._on_button_clicked(btn))
        return btn

    def _on_button_clicked(self, button_or_name):
        if isinstance(button_or_name, str):
            view_name = button_or_name
        else:
            view_name = button_or_name.property("view_name") or button_or_name.text()
        self.bridge.send_message(gui.SwitchViewRequest(view_name=view_name))

    def _setup_contents(self):
        # Instantiate and register global views
        global_views = [(v[1], v[2]) for v in TASKS_LAYOUT_SCHEMA if v[0] == "global"]
        self.global_views = {
            k : v(self.bridge, None) for k, v in global_views
        }
        for name, view_obj in self.global_views.items():
            self.bridge.send_message(gui.RegisterViewRequest(view_name=name, widget=view_obj))

        # Instantiate and register workflow-specific views
        workflow_names = [wf.to_display_name() for wf in Workflow.__members__.values()]
        workflow_views = [(v[1], v[2]) for v in TASKS_LAYOUT_SCHEMA if v[0] == "workflow"]
        self.workflow_views = {}
        for wf in workflow_names:
            self.workflow_views[wf] = {
                name: builder(self.bridge, wf) for (name, builder) in workflow_views
            }
            for task_name, view_obj in self.workflow_views[wf].items():
                view_name = f"{task_name} ({wf})"
                self.bridge.send_message(gui.RegisterViewRequest(view_name=view_name, widget=view_obj))

        workflows = [
            (name.to_display_name(), WORKFLOW_COLORS[i], f"{name.to_var_name()}_icon.png")
            for i, name in enumerate(Workflow.__members__.values())
        ]
        section_workflows = PanelSection("Workflows")
        
        for idx, (name, theme, icon_file) in enumerate(workflows):
            btn = self._create_local_nav_button(name)
            btn.setCheckable(True)
            btn.setProperty("hoverTheme", theme)
            
            icon_path = os.path.join(ASSETS_DIR, icon_file)
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
                btn.setIconSize(QSize(16, 16))
            
            self.workflow_group.addButton(btn)
            section_workflows.add_widget(btn)
            if idx == 0:
                btn.setChecked(True)
                
        self.content_layout.addWidget(section_workflows)

        section_tasks = PanelSection("Tasks")
        first_btn = None
        
        for task_type, name, _ in TASKS_LAYOUT_SCHEMA:
            if task_type == "global":
                btn = self._create_local_nav_button(name)
                btn.setCheckable(True)
                btn.setProperty("view_name", name)
                self.task_group.addButton(btn)
                section_tasks.add_widget(btn)
                if first_btn is None:
                    first_btn = btn
            else:
                for wf in workflow_names:
                    btn = self._create_local_nav_button(name)
                    btn.setCheckable(True)
                    btn.setProperty("view_name", f"{name} ({wf})")
                    btn.setProperty("workflow", wf)
                    self.task_group.addButton(btn)
                    section_tasks.add_widget(btn)
                    
        if first_btn is not None:
            first_btn.setChecked(True)
            
        self.content_layout.addWidget(section_tasks)
        
        self.workflow_group.buttonClicked.connect(self._on_workflow_selected)
        pass

    def _on_workflow_selected(self, button):
        workflow_name = button.text()
        if workflow_name == self._current_workflow:
            return
        self._current_workflow = workflow_name
        
        checked_task_btn = self.task_group.checkedButton()
        current_task_name = None
        if checked_task_btn is not None:
            if checked_task_btn.property("workflow") is not None:
                current_task_name = checked_task_btn.text()
                
        self._update_task_buttons_visibility(workflow_name)
        self.settings.setValue("workflow_name", workflow_name)
        self.bridge.send_message(gui.WorkflowSelectedTelemetry(name=workflow_name))
        
        if current_task_name is not None:
            for btn in self.task_group.buttons():
                if btn.property("workflow") == workflow_name and btn.text() == current_task_name:
                    btn.setChecked(True)
                    view_name = btn.property("view_name")
                    self.bridge.send_message(gui.SwitchViewRequest(view_name=view_name))
                    break

    def _update_task_buttons_visibility(self, workflow_name):
        for btn in self.task_group.buttons():
            wf = btn.property("workflow")
            if wf is not None:
                if wf == workflow_name:
                    btn.show()
                else:
                    btn.hide()
    
    def restore_from_settings(self):
        workflow_name = self.settings.value("workflow_name", "Animal Detection", type=str)
        workflow_name = self.settings.value("workflow_name", "Pose Estimation", type=str)
        
        # Find the workflow button matching the saved name
        for btn in self.workflow_group.buttons():
            if btn.text() == workflow_name:
                btn.setChecked(True)
                self._on_workflow_selected(btn)
                break

    def update_animal_detection_visibility(self):
        pass

    def select_view(self, view_name: str):
        # Temporarily block signals to avoid triggering SwitchViewRequest again
        self.task_group.blockSignals(True)
        for btn in self.task_group.buttons():
            if btn.property("view_name") == view_name or btn.text() == view_name:
                btn.setChecked(True)
                
                wf = btn.property("workflow")
                if wf:
                    self.workflow_group.blockSignals(True)
                    for wf_btn in self.workflow_group.buttons():
                        if wf_btn.text() == wf:
                            wf_btn.setChecked(True)
                            self._update_task_buttons_visibility(wf)
                            break
                    self.workflow_group.blockSignals(False)
                break
        self.task_group.blockSignals(False)
        