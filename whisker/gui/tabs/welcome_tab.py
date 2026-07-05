from pathlib import Path
import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QGroupBox,
)

from whisker.core.workspace import Workspace
from whisker.gui.widgets import ScalableImageLabel
from whisker.gui.constants import ASSETS_DIR

from .base_tab import BaseTab, Workflow


class WelcomeTab(BaseTab):
    # --- New Signals ---
    # These signals announce an *intent* to the main window.
    request_set_workspace = pyqtSignal()
    request_create_project = pyqtSignal()
    request_create_dataset = pyqtSignal()
    request_import_labels = pyqtSignal()
    # --- End New Signals ---

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)
        self._current_workflow = Workflow.POSE_ESTIMATION
        self._money_mode_enabled = False

        # --- Left Panel (Image) ---
        image_layout = QVBoxLayout()
        self._create_welcome_image(image_layout)
        main_layout.addLayout(image_layout, stretch=2)

        # --- Right Panel (Controls) ---
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("<h2>🐾 Welcome to <span style='color:#4F8A8B;'>Whisker</span>!</h2>")
        title.setTextFormat(Qt.TextFormat.RichText)
        controls_layout.addWidget(title)
        
        # --- Workspace Info ---
        workspace_group = QGroupBox("Current Workspace")
        workspace_layout = QVBoxLayout(workspace_group)
        
        self.workspace_label = QLabel("No workspace loaded.")
        self.workspace_label.setWordWrap(True)
        self.workspace_label.setStyleSheet("font-size: 11pt; font-style: italic;")
        workspace_layout.addWidget(self.workspace_label)

        self.set_workspace_btn = QPushButton("Set/Change Workspace...")
        workspace_layout.addWidget(self.set_workspace_btn, 0, Qt.AlignmentFlag.AlignLeft)
        controls_layout.addWidget(workspace_group)

        # --- Quick Actions ---
        self.quick_actions_group = QGroupBox("Quick Start")
        quick_actions_layout = QVBoxLayout(self.quick_actions_group)
        
        self.create_project_btn = QPushButton("Create New Project...")
        self.create_dataset_btn = QPushButton("Create New Dataset...")
        self.import_labels_btn = QPushButton("Import Pose Labels...")
        
        quick_actions_layout.addWidget(self.create_project_btn)
        quick_actions_layout.addWidget(self.create_dataset_btn)
        quick_actions_layout.addWidget(self.import_labels_btn)
        
        controls_layout.addWidget(self.quick_actions_group)
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout, stretch=1)

        # --- Connect Signals ---
        self.set_workspace_btn.clicked.connect(self.request_set_workspace.emit)
        self.create_project_btn.clicked.connect(self.request_create_project.emit)
        self.create_dataset_btn.clicked.connect(self.request_create_dataset.emit)
        self.import_labels_btn.clicked.connect(self.request_import_labels.emit)

        # Initial state
        self.quick_actions_group.setEnabled(False)

    def set_workspace(self, workspace: Optional[Workspace]):
        """Updates the UI based on whether a workspace is loaded."""
        super().set_workspace(workspace)
        if workspace:
            self.workspace_label.setText(str(workspace.base_dir))
            self.quick_actions_group.setEnabled(True)
        else:
            self.workspace_label.setText("No workspace loaded. Please set one to begin.")
            self.quick_actions_group.setEnabled(False)

    def set_active_workflow(self, workflow: Workflow):
        """Updates the welcome image based on the active workflow."""
        self._current_workflow = workflow
        
        if self._money_mode_enabled:
            image_name = "welcome_image_hodl.png"
        elif workflow == Workflow.BEHAVIOR_CLASSIFICATION:
            image_name = "welcome_image_green.png"
        else:
            image_name = "welcome_image_lilac.png"

        image_path = ASSETS_DIR / image_name
        if image_path.exists():
            self.image_label.setPixmap(QPixmap(str(image_path)))
        else:
            self.image_label.setText(f"Image not found: {image_name}")
            logging.warning(f"Welcome image not found at {image_path}")

    # --- UI Helper Methods ---
    def _create_welcome_image(self, layout: QVBoxLayout):
        # Default to lilac/pose estimation
        image_path = ASSETS_DIR / "welcome_image_lilac.png"

        self.image_label = ScalableImageLabel()
        if image_path.exists():
            self.image_label.setPixmap(QPixmap(str(image_path)))
        else:
            self.image_label.setText(f"Image not found: {image_path}")
            logging.warning(f"Welcome image not found at {image_path}")

        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

    def apply_money_mode(self, enable: bool):
        self._money_mode_enabled = enable
        self.set_active_workflow(self._current_workflow)
