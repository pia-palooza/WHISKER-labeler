import enum
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QDialog,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDialogButtonBox,
    QGroupBox,
    QRadioButton,
    QComboBox,
    QSizePolicy
)

from whisker.core.constants import BackendEnum
from whisker.core.workspace import Workspace


class ImportDataType(str, enum.Enum):
    POSES = "poses"
    BEHAVIORS = "behaviors"


class ImportLabelsDialog(QDialog):
    """A dialog for importing labels from various formats."""

    def __init__(self, workspace: Optional[Workspace], parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace = workspace
        self.setWindowTitle("Import Labels")
        
        # Scaling and Sizing
        screen = QApplication.primaryScreen()
        dpi_scale = screen.logicalDotsPerInch() / 96.0
        self.setMinimumWidth(int(600 * dpi_scale))

        main_layout = QVBoxLayout(self)
        
        # --- Grid Layout for Top Selectors ---
        # Using a grid ensures labels and inputs align perfectly in a 3x2 (plus buttons) structure
        grid_layout = QGridLayout()
        grid_layout.setSpacing(int(10 * dpi_scale))
        
        # Row 0: Import Path
        grid_layout.addWidget(self._create_right_label("Import Path:"), 0, 0)
        path_container = QWidget()
        path_hbox = QHBoxLayout(path_container)
        path_hbox.setContentsMargins(0, 0, 0, 0)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select file or folder to import...")
        self.path_edit.textChanged.connect(self._validate_state)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_clicked)
        
        path_hbox.addWidget(self.path_edit)
        path_hbox.addWidget(browse_btn)
        grid_layout.addWidget(path_container, 0, 1)

        # Row 1: Dataset Name
        grid_layout.addWidget(self._create_right_label("Dataset Name:"), 1, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter a name for the new dataset...")
        self.name_edit.textChanged.connect(self._validate_state)
        grid_layout.addWidget(self.name_edit, 1, 1)

        # Row 2: Project Selection
        grid_layout.addWidget(self._create_right_label("Project:"), 2, 0)
        self.project_combo = QComboBox()
        self.project_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if self._workspace:
            project_names = sorted(self._workspace.projects.keys())
            if project_names:
                self.project_combo.addItems(project_names)
            else:
                self.project_combo.addItem("No projects found in workspace")
                self.project_combo.setEnabled(False)
        self.project_combo.currentIndexChanged.connect(self._validate_state)
        grid_layout.addWidget(self.project_combo, 2, 1)

        # Ensure the input column expands more than the label column
        grid_layout.setColumnStretch(1, 1)
        
        main_layout.addLayout(grid_layout)
        main_layout.addSpacing(int(10 * dpi_scale))
        
        # --- Groups ---
        main_layout.addWidget(self._create_data_type_group())
        main_layout.addWidget(self._create_data_format_group())

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self._validate_state()

    def _create_right_label(self, text: str) -> QLabel:
        """Creates a right-aligned label for the grid."""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _validate_state(self):
        """Enables or disables the OK button based on the dialog's state."""
        is_name_valid = bool(self.name_edit.text().strip())
        is_path_valid = bool(self.path_edit.text().strip())
        is_project_valid = (
            self.project_combo.isEnabled() and self.project_combo.currentIndex() >= 0
        )

        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            is_name_valid and is_path_valid and is_project_valid
        )

    @property
    def selected_project_name(self) -> str:
        return self.project_combo.currentText()

    @property
    def selected_data_format(self) -> BackendEnum:
        for fmt, radio in self.format_radios.items():
            if radio.isChecked():
                return fmt
        return None

    @property
    def selected_data_type(self) -> ImportDataType:
        return (
            ImportDataType.POSES
            if self.poses_radio.isChecked()
            else ImportDataType.BEHAVIORS
        )

    @property
    def selected_path(self) -> str:
        return self.path_edit.text()

    @property
    def selected_dataset_name(self) -> str:
        return self.name_edit.text().strip()

    def _on_browse_clicked(self):
        select_dir = True
        settings_pair = (self.selected_data_format, self.selected_data_type)
        if settings_pair == (BackendEnum.MARS, ImportDataType.POSES):
            select_dir = False

        if select_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Folder to Import")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File to Import")

        if path:
            self.path_edit.setText(path)
            self.name_edit.setText(Path(path).name)

    def _create_data_type_group(self) -> QGroupBox:
        group_box = QGroupBox("Data Type")
        layout = QHBoxLayout(group_box)
        self.poses_radio = QRadioButton("Poses")
        self.behaviors_radio = QRadioButton("Behaviors")
        self.poses_radio.setChecked(True)
        layout.addWidget(self.poses_radio)
        layout.addWidget(self.behaviors_radio)
        layout.addStretch()
        return group_box

    def _create_data_format_group(self) -> QGroupBox:
        group_box = QGroupBox("Data Format")
        layout = QVBoxLayout(group_box)
        self.format_radios = {}
        is_first = True
        for fmt in BackendEnum:
            radio = QRadioButton(fmt.value)
            if is_first:
                radio.setChecked(True)
                is_first = False
            self.format_radios[fmt] = radio
            layout.addWidget(radio)
        return group_box
