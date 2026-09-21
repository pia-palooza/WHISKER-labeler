"""Label edits still open in a labeling tab must not be silently left out of an export.

An export copies the label files as saved on disk. Covers: the decision to save / skip / cancel,
that the real save paths (pose and behavior) put an open edit on disk so the export contains it,
and that the export entry points ask first.
"""
import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PyQt6.QtCore import QPointF, QThreadPool
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from fixtures import WorkspaceCase, frame_names
from whisker.core import bundle as fmt
from whisker.gui import unsaved_labels as ul
from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.widgets.data_explorer import action_handler as ah
from whisker.gui.workflows.behavior_classification.widgets.behavior_labeling import BehaviorsLabelingWidget
from whisker.gui.workflows.pose_estimation.widgets.pose_labeling.model import PoseLabelingModel
from whisker.gui.workflows.pose_estimation.widgets.pose_labeling.widget import PoseLabelingWidget
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset

_app = QApplication.instance() or QApplication([])


class FakeView:
    def __init__(self, unsaved=True, saves=True):
        self.unsaved, self.saves, self.saved = unsaved, saves, 0

    def has_unsaved_labels(self):
        return self.unsaved

    def save_labels(self):
        self.saved += 1
        return self.saves


class SettleTests(WorkspaceCase):
    def test_nothing_unsaved_goes_ahead_without_asking(self):
        ask = mock.Mock()
        self.assertTrue(ul.settle_unsaved_labels({"a": FakeView(unsaved=False), "b": object()}, ask))
        ask.assert_not_called()

    def test_save_saves_every_tab_holding_edits_and_only_those(self):
        dirty, clean = FakeView(), FakeView(unsaved=False)
        self.assertTrue(ul.settle_unsaved_labels({"poses": dirty, "clean": clean}, lambda names: ul.SAVE))
        self.assertEqual((dirty.saved, clean.saved), (1, 0))

    def test_the_question_names_the_tabs_holding_edits(self):
        ask = mock.Mock(return_value=ul.CANCEL)
        ul.settle_unsaved_labels({"poses": FakeView(), "clean": FakeView(unsaved=False), "behaviors": FakeView()}, ask)
        ask.assert_called_once_with(["poses", "behaviors"])

    def test_skip_exports_what_is_saved_and_saves_nothing(self):
        view = FakeView()
        self.assertTrue(ul.settle_unsaved_labels({"poses": view}, lambda names: ul.SKIP))
        self.assertEqual(view.saved, 0)

    def test_cancel_stops_the_export_and_saves_nothing(self):
        view = FakeView()
        self.assertFalse(ul.settle_unsaved_labels({"poses": view}, lambda names: ul.CANCEL))
        self.assertEqual(view.saved, 0)

    def test_a_failed_save_stops_the_export_rather_than_exporting_a_stale_copy(self):
        self.assertFalse(ul.settle_unsaved_labels({"poses": FakeView(saves=False)}, lambda names: ul.SAVE))

    def test_plain_tabs_have_nothing_to_settle(self):
        tab = BaseTab()
        self.addCleanup(tab.deleteLater)
        self.assertFalse(tab.has_unsaved_labels())
        self.assertTrue(tab.save_labels())


class SavedEditsReachTheExport(WorkspaceCase):
    """The real save paths, with the workspace state the labeling tabs really leave behind."""

    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.ws = self.new_workspace("src")
        self.project = self.add_project(self.ws)
        self.files = frame_names(3)
        self.add_dataset(self.ws, "d1", self.files)

    def exported(self, name="out"):
        plan = fmt.build_export_plan(self.ws, "d1", "proj")
        out = self.tmp / name
        fmt.export_annotation_bundle(plan, out)
        return out

    def test_an_open_pose_edit_is_missing_from_an_export_until_it_is_saved(self):
        self.add_pose_labels(self.ws, "d1", self.files[:2])
        key = self.files[0]
        model = PoseLabelingModel()
        model.set_data(self.ws.pose_labels.get_pose_dataset("d1"), key, self.project, "d1")
        model.update_keypoint_position("mouse1", "nose", QPointF(123.0, 45.0))
        self.assertTrue(model.is_dirty())

        def exported_nose():
            df = PoseDataset.from_file(self.exported(f"out_{model.is_dirty()}") / "pose_labels" / "labels.h5").keypoint_data
            return float(df.loc[(key, "mouse1", "nose"), "x"])

        self.assertNotEqual(exported_nose(), 123.0)          # the gap this step closes

        widget = SimpleNamespace(
            model=model, _pose_label_operations=self.ws.pose_labels, _selected_dataset=self.ws.datasets.get("d1"),
            _project=self.project, _image_path="x.png", labels_saved=mock.Mock(),
        )
        self.assertTrue(PoseLabelingWidget.save(widget))
        self.assertFalse(model.is_dirty())
        widget.labels_saved.emit.assert_called_once_with("d1", "x.png")
        self.assertEqual(exported_nose(), 123.0)

    def test_an_open_behavior_edit_is_missing_from_an_export_until_it_is_saved(self):
        self.add_behavior_labels(self.ws, "d1", ["a", "b"])
        labels = self.ws.behavior_labels.get_behavior_labels("d1")
        labels.bouts = pd.concat([labels.bouts, pd.DataFrame(
            [{"video_key": "c", "behavior": "rear", "start_frame": 5, "end_frame": 8, "p": float("nan")}]
        ).astype(labels.bouts.dtypes.to_dict())], ignore_index=True)

        def exported_keys(name):
            return set(BehaviorDataset.from_file(self.exported(name) / "behavior_labels" / "labels.h5").bouts["video_key"])

        self.assertEqual(exported_keys("before"), {"a", "b"})          # the gap this step closes

        widget = SimpleNamespace(_workspace=self.ws, _dataset_name="d1", _video_path="c.mp4", labels_saved=mock.Mock())
        self.assertTrue(BehaviorsLabelingWidget.save(widget))
        widget.labels_saved.emit.assert_called_once_with("d1", "c.mp4")
        self.assertEqual(exported_keys("after"), {"a", "b", "c"})

    def test_a_save_that_cannot_happen_says_so(self):
        widget = SimpleNamespace(_workspace=None, _dataset_name=None, _video_path=None, labels_saved=mock.Mock())
        with mock.patch("whisker.gui.workflows.behavior_classification.widgets.behavior_labeling.QMessageBox") as box:
            self.assertFalse(BehaviorsLabelingWidget.save(widget))
        box.critical.assert_called_once()
        widget.labels_saved.emit.assert_not_called()


class MainWindowQuestion(WorkspaceCase):
    """The main window's dialog, driven without opening a window: each button must mean what it says."""

    def answer(self, pick: int, unsaved=True):
        """Run the window's check with a stand-in message box on which button number ``pick``
        (0 Save and Export, 1 Export Saved Version, 2 Cancel) is clicked."""
        from whisker.gui import main_window
        view = FakeView(unsaved=unsaved)
        with mock.patch.object(main_window, "QMessageBox") as box_class:
            box = box_class.return_value
            buttons = [object(), object(), object()]
            box.addButton.side_effect = buttons
            box.clickedButton.return_value = buttons[pick]
            proceed = main_window.MainWindow._settle_unsaved_labels_before_export(SimpleNamespace(views={"Labeling Poses": view}))
        return proceed, view, box

    def test_save_and_export_saves_then_goes_ahead(self):
        proceed, view, box = self.answer(0)
        self.assertEqual((proceed, view.saved), (True, 1))
        self.assertIn("Labeling Poses", box.setInformativeText.call_args.args[0])

    def test_export_saved_version_goes_ahead_without_saving(self):
        proceed, view, _ = self.answer(1)
        self.assertEqual((proceed, view.saved), (True, 0))

    def test_cancel_stops_the_export(self):
        proceed, view, _ = self.answer(2)
        self.assertEqual((proceed, view.saved), (False, 0))

    def test_no_question_when_nothing_is_unsaved(self):
        proceed, _, box = self.answer(2, unsaved=False)
        self.assertTrue(proceed)
        box.exec.assert_not_called()


class ExportEntryPointsAsk(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        ws = self.new_workspace("src")
        self.add_project(ws)
        self.add_dataset(ws, "d1", frame_names(2))
        self.parent = QWidget()
        self.addCleanup(self.parent.deleteLater)
        self.handler = ah.ActionHandler(self.parent, QThreadPool())
        self.handler.update_workspace(ws)
        # Reaching the dialog is as far as these tests go; declining it ends the export.
        patcher = mock.patch.multiple(ah, ExportAnnotationsDialog=mock.DEFAULT, ExportCompilationDialog=mock.DEFAULT)
        self.dialogs = patcher.start()
        self.addCleanup(patcher.stop)
        for dialog in self.dialogs.values():
            dialog.return_value.exec.return_value = QDialog.DialogCode.Rejected

    def test_single_export_opens_its_dialog_when_the_hook_agrees(self):
        hook = mock.Mock(return_value=True)
        self.handler.set_before_export_hook(hook)
        self.handler._export_annotations("d1")
        hook.assert_called_once_with()
        self.dialogs["ExportAnnotationsDialog"].assert_called_once()

    def test_single_export_is_cancelled_when_the_hook_says_no(self):
        self.handler.set_before_export_hook(lambda: False)
        self.handler._export_annotations("d1")
        self.dialogs["ExportAnnotationsDialog"].assert_not_called()

    def test_compilation_export_is_cancelled_when_the_hook_says_no(self):
        self.handler.set_before_export_hook(lambda: False)
        self.handler.show_export_compilation_dialog()
        self.dialogs["ExportCompilationDialog"].assert_not_called()

    def test_compilation_export_opens_its_dialog_when_the_hook_agrees(self):
        hook = mock.Mock(return_value=True)
        self.handler.set_before_export_hook(hook)
        self.handler.show_export_compilation_dialog()
        hook.assert_called_once_with()
        self.dialogs["ExportCompilationDialog"].assert_called_once()

    def test_without_a_hook_exports_open_as_before(self):
        self.handler._export_annotations("d1")
        self.dialogs["ExportAnnotationsDialog"].assert_called_once()
