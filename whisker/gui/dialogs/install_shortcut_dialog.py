import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from whisker.core.utils import desktop_shortcut as ds


class InstallShortcutDialog(QDialog):
    """Lets the user put a WHISKER icon on their Desktop / Start Menu (Windows) or
    Desktop / Applications folder (macOS) so the GUI opens with a double-click."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install Desktop Shortcut")
        self.setModal(True)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Add a WHISKER icon so you can open the labeler with a double-click, "
            "without using a terminal."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self._platform = QComboBox()
        for key in ds.PLATFORMS:
            self._platform.addItem(ds.PLATFORM_LABELS[key], key)
        detected = ds.detect_platform()
        if detected:
            self._platform.setCurrentIndex(ds.PLATFORMS.index(detected))
        self._platform.currentIndexChanged.connect(self._on_platform_changed)
        form.addRow("Computer type:", self._platform)

        self._desktop = QCheckBox()
        self._desktop.setChecked(True)
        self._menu = QCheckBox()
        self._menu.setChecked(True)
        form.addRow("Create icon in:", self._desktop)
        form.addRow("", self._menu)
        layout.addLayout(form)

        self._notice = QLabel()
        self._notice.setWordWrap(True)
        layout.addWidget(self._notice)

        self._buttons = QDialogButtonBox()
        self._install_btn = self._buttons.addButton("Install", QDialogButtonBox.ButtonRole.AcceptRole)
        self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._install)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._on_platform_changed()

    def _selected_platform(self) -> str:
        return self._platform.currentData()

    def _on_platform_changed(self):
        platform = self._selected_platform()
        desktop_label, menu_label = ds.LOCATION_LABELS[platform]
        self._desktop.setText(desktop_label)
        self._menu.setText(menu_label)

        matches = platform == ds.detect_platform()
        self._install_btn.setEnabled(matches)
        if matches:
            self._notice.setText("")
        else:
            here = ds.PLATFORM_LABELS.get(ds.detect_platform(), "another operating system")
            self._notice.setText(
                f"<i>This computer is running {here}. A {ds.PLATFORM_LABELS[platform]} icon "
                f"can only be created on a {ds.PLATFORM_LABELS[platform]} computer.</i>"
            )

    def _install(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = ds.install_shortcut(
                platform=self._selected_platform(),
                desktop=self._desktop.isChecked(),
                start_menu=self._menu.isChecked(),
            )
        except ds.ShortcutError as e:
            QApplication.restoreOverrideCursor()
            logging.error(f"Shortcut install failed: {e}")
            QMessageBox.warning(self, "Could Not Create Shortcut", str(e))
            return
        except Exception as e:
            QApplication.restoreOverrideCursor()
            logging.exception("Unexpected error while creating the shortcut")
            QMessageBox.critical(self, "Could Not Create Shortcut", f"Unexpected error: {e}")
            return
        QApplication.restoreOverrideCursor()

        lines = ["Created:"] + [f"  • {p}" for p in result.created]
        if result.platform == "windows" and self._menu.isChecked():
            lines += ["", "To pin it to the taskbar: open the Start Menu, right-click "
                          f"“{ds.APP_NAME}”, and choose Pin to taskbar."]
        elif result.platform == "macos":
            lines += ["", "To keep it in the Dock: open it once, then right-click its Dock "
                          "icon and choose Options → Keep in Dock."]
        lines += ["", f"It starts in {ds.default_working_dir()} the first time. Use File → "
                      "Open Workspace… to point it at your data; WHISKER remembers your "
                      "choice."]
        lines += [""] + result.warnings if result.warnings else []
        QMessageBox.information(self, "Shortcut Created", "\n".join(lines))
        self.accept()
