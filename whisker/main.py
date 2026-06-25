# UPDATE_FILE: whisker/main.py
#
# Standalone launcher for the WHISKER labeler.
#
# The app opens an existing WHISKER *workspace* folder (the same layout the full
# WHISKER app uses: datasets/, projects/, workflows/<wf>/labels/) and presents:
#   * a Data Explorer (left) listing every dataset, expandable to its files;
#   * a Project selector (toolbar) sourcing body parts / identities / behaviors;
#   * two workflow tabs -- "Pose Estimation" (images) and
#     "Behavior Classification" (videos) -- switchable at will.
#
# Selecting a dataset/file in the explorer routes it to the matching tab.
#
# The workspace opens automatically with NO prompt, resolved in this order
# (mirroring full WHISKER, which defaults to the current working directory):
#   1. --workspace <path> command-line argument
#   2. WHISKER_WORKSPACE environment variable
#   3. the last workspace opened (remembered in QSettings)
#   4. the current working directory
# Use the toolbar "Open Workspace…" button to switch at any time.
#
# This tool only supports HAND ANNOTATION -- there is no model training.
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QMessageBox, QMainWindow, QFileDialog,
    QTabWidget, QSplitter, QComboBox, QToolBar, QVBoxLayout, QDialog,
)

from whisker.gui.constants import ASSETS_DIR
from whisker.core.dataset import Dataset, DatasetType
from whisker.core.workspace import Workspace
from whisker.gui.widgets.data_explorer import DataExplorerWidget
from whisker.gui.dialogs.new_project_dataset_dialog import NewProjectDatasetDialog
from whisker.gui.dialogs.import_dialog import ImportDialog
from whisker.gui.workflows.pose_estimation.widgets.pose_labeling.widget import PoseLabelingWidget
from whisker.gui.workflows.behavior_classification.widgets.behavior_labeling.widget import (
    BehaviorLabelingWidget,
)

SETTINGS_ORG = "WHISKER"
SETTINGS_APP = "Labeler"
WORKSPACE_KEY = "workspace/path"
PROJECT_KEY = "workspace/last_project"


def _default_workspace_dir() -> Path:
    """The labeler's own workspace folder, kept inside the install directory so
    projects, datasets, and annotations persist between launches."""
    # main.py lives at <repo>/whisker/main.py -> parents[1] is the repo root.
    return Path(__file__).resolve().parents[1] / "workspace"


class WorkspaceWindow(QMainWindow):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.workspace: Optional[Workspace] = None
        self._current_dataset: Optional[str] = None

        self._build_toolbar()
        self._build_central()
        self._connect_signals()

        self.load_workspace(workspace)

    # --- Construction ---

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_action = QAction("➕ New Project / Dataset…", self)
        new_action.triggered.connect(self._on_new_project_dataset)
        toolbar.addAction(new_action)

        import_action = QAction("⬇ Import…", self)
        import_action.setToolTip(
            "Copy existing projects, datasets, and labels from a WHISKER "
            "workspace or export folder into this labeler."
        )
        import_action.triggered.connect(self._on_import)
        toolbar.addAction(import_action)

        export_action = QAction("⬆ Export Labels…", self)
        export_action.setToolTip(
            "Copy this dataset's labels into a folder, ready to drop into a "
            "full WHISKER workspace."
        )
        export_action.triggered.connect(self._on_export_labels)
        toolbar.addAction(export_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Project: "))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        toolbar.addWidget(self.project_combo)

    def _build_central(self):
        self.explorer = DataExplorerWidget()

        self.pose_widget = PoseLabelingWidget()
        self.behavior_widget = BehaviorLabelingWidget()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._wrap(self.pose_widget), "Pose Estimation")
        self.tabs.addTab(self._wrap(self.behavior_widget), "Behavior Classification")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.explorer)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1100])
        self.setCentralWidget(splitter)

        self.status_label = QLabel("Open a workspace to begin.")
        self.statusBar().addWidget(self.status_label)

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return container

    def _connect_signals(self):
        self.explorer.dataset_activated.connect(self._on_dataset_activated)
        self.explorer.file_activated.connect(self._on_file_activated)

        self.pose_widget.request_select_prev_image.connect(
            lambda: self.explorer.step_selection(-1)
        )
        self.pose_widget.request_select_next_image.connect(
            lambda: self.explorer.step_selection(1)
        )
        self.pose_widget.labels_saved.connect(self._on_labels_saved)

        self.behavior_widget.request_select_prev_video.connect(
            lambda: self.explorer.step_selection(-1)
        )
        self.behavior_widget.request_select_next_video.connect(
            lambda: self.explorer.step_selection(1)
        )
        self.behavior_widget.labels_saved.connect(self._on_labels_saved)

    # --- Workspace loading ---

    def load_workspace(self, workspace: Workspace):
        self.workspace = workspace
        self._current_dataset = None
        self.setWindowTitle(f"WHISKER Labeler - {workspace.base_dir}")
        self.settings.setValue(WORKSPACE_KEY, str(workspace.base_dir))

        # Rebind label operations to the new workspace.
        self.pose_widget.set_context(workspace.pose_labels, None)
        self.behavior_widget.set_context(workspace.behavior_labels)

        # Populate the project selector.
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        project_names = sorted(workspace.projects.keys())
        self.project_combo.addItems(project_names)
        self.project_combo.blockSignals(False)

        if project_names:
            last = self.settings.value(PROJECT_KEY, "")
            self.project_combo.setCurrentText(
                last if last in project_names else project_names[0]
            )
            self._on_project_changed(self.project_combo.currentText())
        else:
            self._apply_project(None)

        self.explorer.set_workspace(workspace)
        n_ds = len(workspace.datasets.keys())
        if n_ds == 0 and not project_names:
            self.status_label.setText(
                "Empty workspace — click '➕ New Project / Dataset…' to get started."
            )
        else:
            self.status_label.setText(
                f"Workspace: {workspace.base_dir}  |  {n_ds} dataset(s)  |  "
                f"{len(project_names)} project(s)"
            )

    # --- Create new project + dataset ---

    def _on_new_project_dataset(self):
        if not self.workspace:
            return
        projects = {
            name: self.workspace.projects.get(name)
            for name in self.workspace.projects.keys()
        }
        dialog = NewProjectDatasetDialog(
            self,
            projects=projects,
            existing_datasets=set(self.workspace.datasets.keys()),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        spec = dialog.get_spec()
        if not spec:
            return
        try:
            self._create_project_dataset(spec)
        except Exception as e:
            QMessageBox.critical(
                self, "Creation Failed", f"Could not create project/dataset:\n{e}"
            )

    def _create_project_dataset(self, spec):
        # 1. Project -> workspace/projects/<name>.json (skip when reusing one).
        if not spec.use_existing_project:
            self.workspace.projects.create_project(
                spec.project_name,
                body_parts=spec.body_parts,
                identities=spec.identities,
                skeleton=spec.skeleton,
                behaviors=spec.behaviors,
                warn_if_exists=lambda _msg: True,  # already confirmed in the dialog
            )

        # 2. Dataset manifest -> workspace/datasets/<name>/manifest.json.
        #    The media stays in place; only the manifest (pointing at the source
        #    folder) and future labels live in the labeler workspace.
        dataset = Dataset(
            name=spec.dataset_name,
            type=spec.dataset_type,
            base_data_path=str(spec.data_folder.resolve()),
            files=spec.files,
        )
        self.workspace.datasets.add_dataset(
            spec.dataset_name, dataset, warn_if_exists=lambda _msg: True
        )

        # 3. Refresh UI and jump to the new project/dataset.
        self.load_workspace(self.workspace)
        self.project_combo.setCurrentText(spec.project_name)
        self.explorer.select_first_file_of_dataset(spec.dataset_name)
        self.status_label.setText(
            f"Created project '{spec.project_name}' and dataset "
            f"'{spec.dataset_name}' ({len(spec.files)} files)."
        )

    # --- Project selection ---

    def _on_project_changed(self, name: str):
        if not self.workspace or not name:
            return
        self.settings.setValue(PROJECT_KEY, name)
        self._apply_project(self.workspace.projects.get(name))

    def _apply_project(self, project):
        self.pose_widget.set_project(project)
        self.behavior_widget.set_project(project)

    # --- Explorer routing ---

    def _on_dataset_activated(self, dataset_name: str):
        ds = self.workspace.datasets.get(dataset_name)
        if not ds:
            return
        self._current_dataset = dataset_name
        # Switch to the tab that matches the dataset type, then jump to its
        # first file (which triggers file_activated).
        self.tabs.setCurrentIndex(1 if ds.type == DatasetType.VIDEO_COLLECTION else 0)
        self.explorer.select_first_file_of_dataset(dataset_name)

    def _on_file_activated(self, dataset_name: str, rel_path: str):
        ds = self.workspace.datasets.get(dataset_name)
        if not ds:
            return
        self._current_dataset = dataset_name
        abs_path = Path(ds.base_data_path) / rel_path
        if ds.type == DatasetType.VIDEO_COLLECTION:
            self.tabs.setCurrentIndex(1)
            self.behavior_widget.set_media(self.workspace.datasets, dataset_name, abs_path)
        else:
            self.tabs.setCurrentIndex(0)
            self.pose_widget.set_media(self.workspace.datasets, dataset_name, abs_path)
        self.status_label.setText(f"{dataset_name}  /  {rel_path}")

    def _on_labels_saved(self, dataset_name: str, media_path: str):
        self.explorer.refresh_labels_for(dataset_name)

    # --- Import existing projects / datasets / labels ---

    def _on_import(self):
        if not self.workspace:
            return
        dialog = ImportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sel = dialog.get_selection()
        if not sel:
            return
        try:
            self._do_import(sel)
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Could not import:\n{e}")

    def _do_import(self, sel):
        src = sel.source

        # Warn about anything that would be overwritten in the labeler workspace.
        conflicts = [f"project '{p}'" for p in sel.projects
                     if p in self.workspace.projects.keys()]
        conflicts += [f"dataset '{d}' (and its labels)" for d in sel.datasets
                      if d in self.workspace.datasets.keys()]
        if conflicts and QMessageBox.question(
            self, "Overwrite?",
            "These already exist in the labeler and will be overwritten:\n\n"
            + "\n".join(conflicts) + "\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return

        def _copy(src_path: Path, dst_path: Path):
            if src_path.is_file():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                return True
            return False

        n_proj = 0
        for p in sel.projects:
            if _copy(src / "projects" / f"{p}.json",
                     self.workspace.projects.base_dir / f"{p}.json"):
                n_proj += 1

        n_ds = n_pose = n_beh = 0
        for d in sel.datasets:
            if _copy(src / "datasets" / d / "manifest.json",
                     self.workspace.datasets.base_dir / d / "manifest.json"):
                n_ds += 1
            if _copy(src / "workflows" / "pose_estimation" / "labels" / d / "labels.h5",
                     self.workspace.pose_labels.base_dir / d / "labels.h5"):
                n_pose += 1
            if _copy(src / "workflows" / "behavior_classification" / "labels" / d / "labels.h5",
                     self.workspace.behavior_labels.base_dir / d / "labels.h5"):
                n_beh += 1

        # Re-scan the (unchanged-object) workspace and refresh the UI.
        self.workspace.scan()
        self.load_workspace(self.workspace)

        QMessageBox.information(
            self, "Import Complete",
            f"Imported {n_proj} project(s) and {n_ds} dataset(s) "
            f"({n_pose} pose + {n_beh} behavior label file(s)).",
        )

    # --- Export labels for import into full WHISKER ---

    def _on_export_labels(self):
        if not self.workspace:
            return
        name = self._current_dataset
        if not name:
            QMessageBox.information(
                self, "Export Labels",
                "Select a dataset in the Data Explorer first.",
            )
            return
        ds = self.workspace.datasets.get(name)
        if not ds:
            QMessageBox.warning(self, "Export Labels", f"Dataset '{name}' not found.")
            return

        if ds.type == DatasetType.VIDEO_COLLECTION:
            wf_dir, kind = "behavior_classification", "behavior"
            src = self.workspace.behavior_labels.get_behavior_labels_path(name)
        else:
            wf_dir, kind = "pose_estimation", "pose"
            src = self.workspace.pose_labels.base_dir / name / "labels.h5"

        if not src.exists():
            QMessageBox.information(
                self, "No Labels Yet",
                f"No saved {kind} labels for '{name}' yet.\n\n"
                "Label some media and press Save first.",
            )
            return

        dest_root = QFileDialog.getExistingDirectory(
            self, "Choose export destination folder"
        )
        if not dest_root:
            return
        dest_root = Path(dest_root)

        # Recreate the full-WHISKER workspace layout so it drops straight in.
        labels_dest = dest_root / "workflows" / wf_dir / "labels" / name / "labels.h5"
        labels_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, labels_dest)
        written = [str(labels_dest)]

        # Include the dataset manifest so the export is self-contained and can
        # be re-imported (full WHISKER simply ignores it if it already has one).
        manifest_src = self.workspace.datasets.base_dir / name / "manifest.json"
        if manifest_src.exists():
            manifest_dest = dest_root / "datasets" / name / "manifest.json"
            manifest_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_src, manifest_dest)
            written.append(str(manifest_dest))

        # Also export the matching project so behaviors/body parts line up.
        project_name = self.project_combo.currentText()
        if project_name:
            proj_src = self.workspace.projects.base_dir / f"{project_name}.json"
            if proj_src.exists():
                proj_dest = dest_root / "projects" / f"{project_name}.json"
                proj_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(proj_src, proj_dest)
                written.append(str(proj_dest))

        QMessageBox.information(
            self, "Export Complete",
            "Exported into a WHISKER-compatible layout:\n\n"
            + "\n".join(written)
            + "\n\nMerge the 'workflows' (and 'projects') folder into your full "
            f"WHISKER workspace to import the {kind} labels for '{name}'.",
        )


# --- Automatic workspace resolution (no prompt) ---

def _open_workspace(path: Path) -> Optional[Workspace]:
    try:
        return Workspace(path)
    except Exception as e:
        print(f"[WHISKER Labeler] Could not open workspace '{path}': {e}", file=sys.stderr)
        return None


def _resolve_initial_workspace(cli_workspace: Optional[str]) -> Workspace:
    """
    Resolve the startup workspace without ever prompting the user.
    Order: --workspace arg, WHISKER_WORKSPACE env, then the labeler's own
    internal workspace folder (the default — keeps everything self-contained).
    """
    for cand in [cli_workspace, os.environ.get("WHISKER_WORKSPACE")]:
        if cand and Path(cand).is_dir():
            ws = _open_workspace(Path(cand))
            if ws:
                return ws

    # Default: the labeler's own workspace inside the install directory. Create
    # it on first launch so projects/datasets/labels persist here.
    default = _default_workspace_dir()
    default.mkdir(parents=True, exist_ok=True)
    return Workspace(default)


def main():
    parser = argparse.ArgumentParser(description="WHISKER standalone labeler")
    parser.add_argument(
        "--workspace", "-w", default=None,
        help="Path to the WHISKER workspace folder to open (overrides the "
             "remembered workspace and the current directory).",
    )
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setApplicationName("WHISKER Labeler")

    icon_path = ASSETS_DIR / "favicon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    workspace = _resolve_initial_workspace(args.workspace)

    window = WorkspaceWindow(workspace)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
