"""Data Explorer "Copy Source Path" context-menu action: already existed for a whole dataset
(copies dataset.base_data_path); this covers the per-video version added so a user can right-click
a specific video inside a VIDEO_COLLECTION dataset and copy the folder it actually lives in - the
right place to manually drop a new video alongside it, even when files sit in subfolders.
"""
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt, QThreadPool
from PyQt6.QtWidgets import QApplication, QMenu, QTreeWidget, QTreeWidgetItem, QWidget

from fixtures import WorkspaceCase
from whisker.core.study.dataset import DatasetType
from whisker.gui.widgets.data_explorer import action_handler as ah

_app = QApplication.instance() or QApplication(["whisker_test_copy_path"])


def build_video_item(tree: QTreeWidget, dataset_name: str, video_relative_path: str) -> QTreeWidgetItem:
    """A minimal (root -> dataset -> video) tree matching what the real Data Explorer builds for
    a VIDEO_COLLECTION dataset - enough for infer_selection_item_type/get_item_hierarchy to resolve
    this leaf as a DATASET_VIDEO with the given dataset name and full relative path."""
    root = QTreeWidgetItem(tree, ["Video Collection"])
    ds_item = QTreeWidgetItem(root, [dataset_name])
    ds_item.setData(0, Qt.ItemDataRole.UserRole, dataset_name)
    video_item = QTreeWidgetItem(ds_item, [Path(video_relative_path).name])
    video_item.setData(0, Qt.ItemDataRole.UserRole, video_relative_path)
    return video_item


class CopySourcePath(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.ws = self.new_workspace()
        self.parent = QWidget()
        self.addCleanup(self.parent.deleteLater)
        self.handler = ah.ActionHandler(self.parent, QThreadPool())
        self.handler.update_workspace(self.ws)
        self.tree = QTreeWidget()
        self.addCleanup(self.tree.deleteLater)

    def build_menu_for(self, item: QTreeWidgetItem) -> QMenu:
        """Runs the real context-menu-building code without letting the real (blocking) QMenu.exec
        run, and returns the menu it built."""
        captured = []

        def fake_exec(menu_self, *args, **kwargs):
            captured.append(menu_self)

        with mock.patch.object(QMenu, "exec", new=fake_exec):
            self.handler._handle_dataset_context_menu(item, QPoint(0, 0), self.tree)
        self.assertTrue(captured, "menu.exec was never reached - no actions were added")
        return captured[0]

    def find_action(self, menu: QMenu, text: str):
        return next((a for a in menu.actions() if a.text() == text), None)

    def test_copy_source_path_copies_the_videos_own_folder(self):
        media_root = self.tmp / "media"
        ds = self.add_dataset(self.ws, "vid_ds", ["myvideo.mp4"], dtype=DatasetType.VIDEO_COLLECTION,
                               media_root=media_root)
        item = build_video_item(self.tree, "vid_ds", "myvideo.mp4")

        menu = self.build_menu_for(item)
        action = self.find_action(menu, "Copy Source Path")
        self.assertIsNotNone(action)
        action.trigger()

        self.assertEqual(QApplication.clipboard().text(), str(media_root))

    def test_a_video_in_a_subfolder_copies_that_subfolder_not_the_dataset_root(self):
        media_root = self.tmp / "media"
        self.add_dataset(self.ws, "vid_ds", ["sub/myvideo.mp4"], dtype=DatasetType.VIDEO_COLLECTION,
                          media_root=media_root)
        item = build_video_item(self.tree, "vid_ds", "sub/myvideo.mp4")

        menu = self.build_menu_for(item)
        action = self.find_action(menu, "Copy Source Path")
        action.trigger()

        self.assertEqual(QApplication.clipboard().text(), str(media_root / "sub"))

    def test_the_dataset_level_copy_source_path_is_unaffected(self):
        media_root = self.tmp / "media"
        self.add_dataset(self.ws, "vid_ds", ["myvideo.mp4"], dtype=DatasetType.VIDEO_COLLECTION,
                          media_root=media_root)
        root = QTreeWidgetItem(self.tree, ["Video Collection"])
        ds_item = QTreeWidgetItem(root, ["vid_ds"])
        ds_item.setData(0, Qt.ItemDataRole.UserRole, "vid_ds")

        menu = self.build_menu_for(ds_item)
        action = self.find_action(menu, "Copy Source Path")
        action.trigger()

        self.assertEqual(QApplication.clipboard().text(), str(media_root))
