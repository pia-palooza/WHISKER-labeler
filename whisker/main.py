# UPDATE_FILE: whisker/gui/workflows/pose_estimation/widgets/pose_labeling/main.py
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, 
    QLabel, QFileDialog, QWidget, QListWidget, QGridLayout, QMessageBox,
    QListWidgetItem
)
from PyQt6.QtCore import QSettings, Qt

from whisker.gui.constants import ASSETS_DIR
from whisker.core.dataset import Dataset, DatasetType, IMAGE_FILE_EXTENSIONS
from whisker.core.project import Project
from whisker.core.workflows.pose_estimation.data_structures import PoseDataset
from whisker.gui.workflows.pose_estimation.widgets.pose_labeling.widget import PoseLabelingWidget
from PyQt6.QtGui import QIcon


# --- Lightweight Mocks for Standalone Operation ---

class MockDatasetOperations:
    def __init__(self, base_path: Path, files: list[str]):
        self.ds = Dataset(
            name="Standalone", 
            type=DatasetType.IMAGE_COLLECTION, 
            base_data_path=str(base_path), 
            files=files
        )

    def get_dataset(self, name: str) -> Dataset:
        return self.ds

class MockPoseLabelOperations:
    def __init__(self, labels_path: Path):
        self.labels_path = labels_path
        self.ds = None

    def get_pose_dataset(self, name: str, raise_if_missing: bool = False):
        if self.ds is None and self.labels_path.exists():
            try:
                self.ds = PoseDataset.from_file(self.labels_path)
            except Exception as e:
                if raise_if_missing: raise e
        return self.ds

    def set_pose_labels(self, name: str, ds: PoseDataset):
        self.ds = ds

    def save_pose_labels(self, name: str):
        if self.ds:
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            self.ds.to_file(self.labels_path)


# --- UI Implementation ---

class PathConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Standalone Labeling Configuration")
        self.settings = QSettings("WHISKER", "StandalonePoseLabeling")
        self.inputs = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        
        configs = [
            ("Image Dataset Folder", "path/dataset", True),
            ("Project File (.json)", "path/project", False),
        ]

        for i, (label_text, key, is_dir) in enumerate(configs):
            lbl = QLabel(label_text)
            edit = QLineEdit(self.settings.value(key, ""))
            edit.setMinimumWidth(int(QApplication.primaryScreen().size().width() * 0.4))
            btn = QPushButton("Browse")
            
            edit.setMinimumHeight(35)
            btn.setMinimumHeight(35)
            btn.setFixedWidth(100)
            
            btn.clicked.connect(lambda _, e=edit, k=key, d=is_dir: self._browse(e, k, d))
            
            grid.addWidget(lbl, i, 0)
            grid.addWidget(edit, i, 1)
            grid.addWidget(btn, i, 2)
            self.inputs[key] = edit

        layout.addLayout(grid)
        
        ok_btn = QPushButton("Launch Labeling Tool")
        ok_btn.setMinimumHeight(40)
        ok_btn.clicked.connect(self._handle_accept)
        layout.addWidget(ok_btn)
        
        self.setMinimumWidth(500)

    def _browse(self, edit: QLineEdit, key: str, is_dir: bool):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, f"Select {key}")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Project File", "", "JSON files (*.json)"
            )
            
        if path:
            edit.setText(path)

    def _handle_accept(self):
        for key, edit in self.inputs.items():
            self.settings.setValue(key, edit.text())
        self.accept()

    def get_paths(self) -> dict[str, str]:
        return {key: edit.text() for key, edit in self.inputs.items()}


class MainContainer(QWidget):
    def __init__(self, paths: dict[str, str]):
        super().__init__()
        self.dataset_path = Path(paths["path/dataset"])
        self.project_path = Path(paths["path/project"])

        # Main layout is now vertical to accommodate the footer toolbar
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Middle section: Sidebar + Toggle + Editor
        middle_container = QWidget()
        middle_layout = QHBoxLayout(middle_container)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)

        self._setup_sidebar(middle_layout)
        
        # Sleek vertical toggle strip
        self.toggle_btn = QPushButton("‹")
        self.toggle_btn.setFixedWidth(12)
        self.toggle_btn.setFixedHeight(80)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d; color: #888; border: none;
                border-top-right-radius: 4px; border-bottom-right-radius: 4px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3d3d3d; color: #fff; }
        """)
        self.toggle_btn.clicked.connect(self._toggle_sidebar)
        
        toggle_layout = QVBoxLayout()
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.toggle_btn)
        toggle_layout.addStretch()
        middle_layout.addLayout(toggle_layout)

        self.pose_widget = PoseLabelingWidget()
        middle_layout.addWidget(self.pose_widget, stretch=1)
        outer_layout.addWidget(middle_container, stretch=1)

        # Bottom Toolbar
        self.status_bar = QWidget()
        self.status_bar.setFixedHeight(30)
        self.status_bar.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #333; color: #aaa;")
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)
        
        self.status_label = QLabel("Initializing...")
        status_layout.addWidget(self.status_label)
        outer_layout.addWidget(self.status_bar)

        self._init_data()
        self._connect_signals()

    def _setup_sidebar(self, layout):
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setStyleSheet("#sidebar { border-right: 1px solid #333; }")
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        
        # Output field
        sidebar_layout.addWidget(QLabel("Output Labels (.h5)"))
        out_layout = QHBoxLayout()
        self.out_file_edit = QLineEdit(str(self.dataset_path / "standalone_labels.h5"))
        out_btn = QPushButton("...")
        out_btn.setFixedWidth(30)
        out_btn.clicked.connect(self._browse_output_file)
        out_layout.addWidget(self.out_file_edit)
        out_layout.addWidget(out_btn)
        sidebar_layout.addLayout(out_layout)
        self.out_file_edit.textChanged.connect(self._on_output_file_changed)

        # Image list
        sidebar_layout.addWidget(QLabel("Image Selection"))
        self.list_widget = QListWidget()
        sidebar_layout.addWidget(self.list_widget)
        layout.addWidget(self.sidebar)

    def _toggle_sidebar(self):
        visible = self.sidebar.isVisible()
        self.sidebar.setVisible(not visible)
        self.toggle_btn.setText("›" if visible else "‹")
        self.toggle_btn.setFixedWidth(16 if visible else 12)

    def _update_status_bar(self):
        """Updates the footer with path and labeling progress."""
        item = self.list_widget.currentItem()
        path_str = item.data(Qt.ItemDataRole.UserRole) if item else "None"
        
        total = self.list_widget.count()
        labeled_count = 0
        for i in range(total):
            if "✓" in self.list_widget.item(i).text():
                labeled_count += 1
        
        self.status_label.setText(f"Current image: {path_str}  |  {labeled_count} / {total} Labeled")

    def _init_data(self):
        try:
            self.project = Project.from_json(self.project_path.read_text(encoding='utf-8'))
        except Exception as e:
            QMessageBox.critical(self, "Project Error", f"Failed to load project:\n{e}")
            sys.exit(1)

        # Eager Globbing for supported image extensions
        all_files = []
        for ext in IMAGE_FILE_EXTENSIONS:
            all_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        rel_files = [str(f.relative_to(self.dataset_path)).replace("\\", "/") for f in all_files]
        
        for f in sorted(rel_files):
            item = QListWidgetItem(f)
            item.setData(Qt.ItemDataRole.UserRole, f)  # Store pristine path in user role
            self.list_widget.addItem(item)

        # Initialize Mocks
        self.ds_ops = MockDatasetOperations(self.dataset_path, rel_files)
        self.lbl_ops = MockPoseLabelOperations(Path(self.out_file_edit.text()))

        # Pass None to bypass predictions
        self.pose_widget.set_context(self.lbl_ops, None)
        self.pose_widget.set_project(self.project)

        self._refresh_list_indicators()

    def _connect_signals(self):
        self.list_widget.itemSelectionChanged.connect(self._on_image_selected)
        self.pose_widget.request_select_prev_image.connect(self._select_prev)
        self.pose_widget.request_select_next_image.connect(self._select_next)
        self.pose_widget.labels_saved.connect(self._on_labels_saved)

    def _browse_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", self.out_file_edit.text(), "HDF5 files (*.h5)"
        )
        if path:
            self.out_file_edit.setText(path)

    def _on_output_file_changed(self, new_path: str):
        if new_path.strip():
            self.lbl_ops.labels_path = Path(new_path.strip())
            self.lbl_ops.ds = None  # Clear cache to force reload
            self._refresh_list_indicators()
            self._on_image_selected()  # Refresh the canvas context

    def _refresh_list_indicators(self):
        """Reads the current .h5 file and adds a checkmark to labeled frames."""
        labeled_keys = set()
        if self.lbl_ops.labels_path.exists():
            try:
                ds = PoseDataset.from_file(self.lbl_ops.labels_path)
                labeled_keys = set(ds.frame_indices)
            except Exception:
                pass
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            rel_path = item.data(Qt.ItemDataRole.UserRole)
            if rel_path in labeled_keys:
                item.setText(f"✓ {rel_path}")
            else:
                item.setText(rel_path)

    def _on_image_selected(self):
        item = self.list_widget.currentItem()
        if not item: return
        
        rel_path = item.data(Qt.ItemDataRole.UserRole)
        self.pose_widget.set_media(self.ds_ops, "Standalone", self.dataset_path / rel_path)
        self._update_status_bar()


    def _select_prev(self):
        curr_row = self.list_widget.currentRow()
        if curr_row > 0:
            self.list_widget.setCurrentRow(curr_row - 1)

    def _select_next(self):
        curr_row = self.list_widget.currentRow()
        if curr_row < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(curr_row + 1)

    def _on_labels_saved(self, dataset_name: str, media_path: str):
        rel_path = str(Path(media_path).relative_to(self.dataset_path)).replace("\\", "/")
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == rel_path:
                if "✓" not in item.text():
                    item.setText(f"✓ {rel_path}")
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("WHISKER Standalone Pose Labeling")
    
    icon_path = ASSETS_DIR / "favicon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    config_dialog = PathConfigDialog()
    if config_dialog.exec() == QDialog.DialogCode.Accepted:
        selected_paths = config_dialog.get_paths()
        
        container = MainContainer(paths=selected_paths)
        container.setWindowTitle("WHISKER Standalone Pose Labeling")
        container.showMaximized()
        
        if container.list_widget.count() > 0:
            container.list_widget.setCurrentRow(0)
            
        sys.exit(app.exec())
    else:
        sys.exit(0)