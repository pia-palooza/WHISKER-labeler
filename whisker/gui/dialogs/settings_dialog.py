from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QCheckBox,
    QDialogButtonBox,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import QSettings
from whisker.gui.signals import MessageBus

class SettingsDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings

        self.layout = QVBoxLayout(self)

        # Checkbox for Animal Detection Button
        self.show_animal_detection_cb = QCheckBox("Show the Animal Detection Button")
        # Default to True
        is_visible = self.settings.value("show_animal_detection_button", True, type=bool)
        self.show_animal_detection_cb.setChecked(is_visible)

        self.layout.addWidget(self.show_animal_detection_cb)

        # Checkbox for $$$$ Money Mode $$$ Button
        self.enable_money_mode_cb = QCheckBox("Enable $$$$ Money Mode $$$")
        # Default to False
        is_visible = self.settings.value("enable_money_mode", False, type=bool)
        self.enable_money_mode_cb.setChecked(is_visible)
        self.layout.addWidget(self.enable_money_mode_cb)

        # Checkbox for Show Console
        self.show_console_cb = QCheckBox("Show Console")
        # Default to True
        is_visible = self.settings.value("show_console", True, type=bool)
        self.show_console_cb.setChecked(is_visible)
        self.layout.addWidget(self.show_console_cb)

        # Workspace Section
        self.layout.addSpacing(10)
        self.set_workspace_button = QPushButton("Set Workspace...")
        self.set_workspace_button.clicked.connect(self._show_set_workspace_dialog)
        self.layout.addWidget(self.set_workspace_button)

        # Refresh Workspace Button
        self.refresh_button = QPushButton("Refresh Workspace")
        self.refresh_button.clicked.connect(self._refresh_workspace)
        self.layout.addWidget(self.refresh_button)

        # OK / Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.layout.addWidget(self.button_box)

    def _show_set_workspace_dialog(self):
        current_workspace = self.settings.value("workspace_dir", str(Path.cwd()))
        dialog = QFileDialog(
            self, "Select Workspace Directory", directory=str(current_workspace)
        )
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                workspace_path = Path(selected_dirs[0])
                # We update the setting immediately and notify the main window via a signal or direct call if possible.
                # Since MainWindow calls set_workspace on success, we might want to let MainWindow handle it.
                # However, the user wants it in the Settings dialog.
                
                # We'll emit a signal via MessageBus so MainWindow can react.
                MessageBus.get().publish("request/workspace/set", {"path": workspace_path})

    def _refresh_workspace(self):
        bus = MessageBus.get()
        bus.publish("request/workspace/projects/refresh")
        bus.publish("request/workspace/datasets/refresh")
        bus.publish("request/workspace/labels/refresh")
        bus.publish("request/workspace/models/refresh")
        bus.publish("request/workspace/predictions/refresh")

    def accept(self):
        # Save settings
        self.settings.setValue(
            "show_animal_detection_button",
            self.show_animal_detection_cb.isChecked()
        )
        self.settings.setValue(
            "enable_money_mode",
            self.enable_money_mode_cb.isChecked()
        )
        self.settings.setValue(
            "show_console",
            self.show_console_cb.isChecked()
        )
        super().accept()
