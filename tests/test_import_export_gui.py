"""Click-through tests for the import / export dialogs (headless Qt, real bundles)."""
import os
import shutil
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog, QRadioButton

from fixtures import WorkspaceCase, frame_names
from whisker.core import bundle_import as bi
from whisker.core.bundle_import import LabelPolicy
from whisker.core.study.dataset import DatasetType
from whisker.gui.dialogs import AttachLabelsDialog, ExportAnnotationsDialog, ImportBundleDialog
from whisker.gui.dialogs import attach_labels_dialog as attach_mod
from whisker.gui.dialogs import import_bundle_dialog as import_mod
from whisker.gui.widgets.data_explorer.action_handler import ActionHandler

_app = QApplication.instance() or QApplication([])


class GuiCase(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.src = self.new_workspace("src")
        self.add_project(self.src)
        self.files = frame_names(5)
        self.add_dataset(self.src, "d1", self.files)
        self.add_pose_labels(self.src, "d1", self.files[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.bundle = self.export_bundle(self.src, "d1")
        self.dst = self.new_workspace("dst")

    def dialog(self, path=None):
        d = ImportBundleDialog(self.dst)
        self.addCleanup(d.deleteLater)
        if path is not None:
            d.set_path(str(path))
        return d

    def checks(self, d):
        return {k: (c.isChecked(), c.isEnabled()) for k, (c, _n) in d._rows.items()}


class ImportDialogTests(GuiCase):
    def test_picking_the_export_ticks_everything_and_enables_import(self):
        d = self.dialog(self.bundle)
        self.assertTrue(d.parts_box.isVisibleTo(d))
        self.assertEqual(self.checks(d), {k: (True, True) for k in ("project", "dataset", "pose", "behavior")})
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.import_btn.text(), "Import")
        self.assertIn("Found the export 'd1_bundle'", d.location_label.text())

    def test_any_pick_inside_the_export_works_and_says_so(self):
        d = self.dialog(self.bundle / "frames")
        self.assertTrue(d.import_btn.isEnabled())
        self.assertIn("inside the export", d.location_label.text())

    def test_the_frames_row_is_named_for_the_media_kind(self):
        self.assertEqual(self.dialog(self.bundle)._rows["dataset"][0].text(), "Frames")

    def test_a_bad_pick_explains_itself_and_blocks_import(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        d = self.dialog(empty)
        self.assertFalse(d.parts_box.isVisibleTo(d))
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("export_info.json", d.location_label.text())

    def test_several_exports_ask_which(self):
        box = self.tmp / "many"
        shutil.copytree(self.bundle, box / "one")
        shutil.copytree(self.bundle, box / "two")
        d = self.dialog(box)
        self.assertTrue(d.candidate_combo.isVisibleTo(d))
        self.assertEqual(d.candidate_combo.count(), 2)
        self.assertFalse(d.parts_box.isVisibleTo(d))
        d._on_candidate_chosen(1)
        self.assertTrue(d.parts_box.isVisibleTo(d))
        self.assertTrue(d.import_btn.isEnabled())

    def test_unticking_everything_disables_import(self):
        d = self.dialog(self.bundle)
        for check, _ in d._rows.values():
            check.setChecked(False)
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("Tick at least one", d.problem_label.text())

    def test_missing_parts_are_disabled_with_a_reason(self):
        b = self.export_bundle(self.src, "d1", name="lite", include_project=False, include_behavior=False)
        d = self.dialog(b)
        self.assertEqual(self.checks(d)["project"], (False, False))
        self.assertEqual(self.checks(d)["behavior"], (False, False))
        self.assertEqual(self.checks(d)["pose"], (True, True))
        self.assertIn("Not in this export", d._rows["project"][1].text())

    def test_a_damaged_part_is_shown_in_red_and_the_rest_still_import(self):
        (self.bundle / "pose_labels" / "labels.h5").unlink()
        d = self.dialog(self.bundle)
        self.assertEqual(self.checks(d)["pose"], (False, False))
        self.assertIn("missing", d._rows["pose"][1].text())
        self.assertTrue(d.import_btn.isEnabled())

    def test_reimport_defaults_to_labels_for_the_dataset_you_have(self):
        bi.import_from_bundle(self.dst, bi.inspect_bundle(self.bundle),
                              bi.ImportSelection(project=True, dataset=True, dataset_name="d1"))
        self.dst.scan_projects(); self.dst.scan_datasets(); self.dst.scan_labels()
        d = self.dialog(self.bundle)
        self.assertEqual(self.checks(d), {"project": (False, True), "dataset": (False, True),
                                          "pose": (True, True), "behavior": (True, True)})
        self.assertIn("already have a dataset called 'd1'", d._rows["dataset"][1].text())
        self.assertIn("this exact project", d._rows["project"][1].text())
        self.assertEqual(d.import_btn.text(), "Next...")            # a follow-up question is coming
        self.assertTrue(d.import_btn.isEnabled())
        self.assertIn("choose which of your datasets", d.labels_note.text())

    def test_ticking_the_dataset_you_already_have_needs_a_new_name_or_replace(self):
        bi.import_from_bundle(self.dst, bi.inspect_bundle(self.bundle), bi.ImportSelection(dataset=True, dataset_name="d1"))
        self.dst.scan_datasets()
        d = self.dialog(self.bundle)
        d._rows["dataset"][0].setChecked(True)
        self.assertEqual(d.name_edit.text(), "d1_2")                # suggested unique name
        self.assertTrue(d.import_btn.isEnabled())
        d.name_edit.setText("d1")
        self.assertFalse(d.import_btn.isEnabled())
        self.assertTrue(d.dataset_replace.isVisibleTo(d))
        self.assertIn("already have a dataset called 'd1'", d.problem_label.text())
        d.dataset_replace.setChecked(True)
        self.assertTrue(d.import_btn.isEnabled())

    def test_reference_only_export_asks_for_the_folder(self):
        b = self.export_bundle(self.src, "d1", name="ref", include_media=False)
        shutil.rmtree(self.tmp / "media" / "d1")
        d = self.dialog(b)
        self.assertTrue(d.media_row.isVisibleTo(d))
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("Choose the folder", d.problem_label.text())
        supplied = self.tmp / "supplied"
        for f in self.files:
            (supplied / f).parent.mkdir(parents=True, exist_ok=True)
            (supplied / f).write_bytes(b"x")
        d.media_edit.setText(str(supplied))
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.selection.media_dir, supplied)

    def test_manual_mode_button_returns_the_advanced_code(self):
        d = self.dialog(self.bundle)
        with mock.patch.object(d, "done") as done:
            d.manual_btn.click()
        done.assert_called_once_with(import_mod.ADVANCED_RESULT)

    def test_dropping_a_folder_selects_it(self):
        from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
        from PyQt6.QtGui import QDropEvent
        d = self.dialog()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.bundle))])
        d.dropEvent(QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.contents.root, self.bundle)

    def test_import_with_a_dataset_needs_no_followup_question(self):
        d = self.dialog(self.bundle)
        with mock.patch.object(import_mod, "AttachLabelsDialog") as attach:
            d._on_import_clicked()
        attach.assert_not_called()
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)

    def test_labels_only_asks_the_followup_and_cancelling_it_keeps_this_dialog_open(self):
        self.add_dataset(self.dst, "mine", self.files, media_root=self.tmp / "dm")
        d = self.dialog(self.bundle)
        d._rows["dataset"][0].setChecked(False)
        d._rows["project"][0].setChecked(False)
        self.assertEqual(d.import_btn.text(), "Next...")
        with mock.patch.object(import_mod, "AttachLabelsDialog") as attach:
            attach.return_value.exec.return_value = QDialog.DialogCode.Rejected
            d._on_import_clicked()
            attach.assert_called_once()
        self.assertNotEqual(d.result(), QDialog.DialogCode.Accepted)   # still open
        with mock.patch.object(import_mod, "AttachLabelsDialog") as attach:
            attach.return_value.exec.return_value = QDialog.DialogCode.Accepted
            d._on_import_clicked()
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)


class AttachDialogTests(GuiCase):
    def make(self, **sel):
        contents = bi.inspect_bundle(self.bundle)
        selection = bi.ImportSelection(pose=True, behavior=True, **sel)
        d = AttachLabelsDialog(self.dst, contents, selection)
        self.addCleanup(d.deleteLater)
        return d, selection

    def radios(self, d):
        return [r.text() for r in d.findChildren(QRadioButton)]

    def test_datasets_are_ranked_by_fit_and_the_best_is_preselected(self):
        self.add_dataset(self.dst, "unrelated", ["z1.png", "z2.png"], media_root=self.tmp / "m1")
        self.add_dataset(self.dst, "fits", self.files, media_root=self.tmp / "m2")
        d, _ = self.make()
        self.assertEqual(d.dataset_combo.itemData(0), "fits")
        self.assertIn("3 of 5 labels match", d.dataset_combo.itemText(0))      # 3 pose keys (behavior keys don't apply)
        self.assertEqual(d._target(), "fits")

    def test_no_existing_labels_means_they_are_simply_added(self):
        self.add_dataset(self.dst, "fits", self.files, media_root=self.tmp / "m2")
        d, sel = self.make(target_dataset="fits")
        self.assertEqual(self.radios(d), [])
        self.assertEqual(d._policy_of["pose"](), LabelPolicy.ADD)
        self.assertTrue(d.ok_btn.isEnabled())
        d._accept()
        self.assertEqual((sel.target_dataset, sel.pose_policy), ("fits", LabelPolicy.ADD))

    def test_existing_labels_offer_merge_choices_and_default_to_the_safe_one(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_pose_labels(self.dst, "t", self.files[2:4])                    # overlaps on frame 2
        d, sel = self.make(target_dataset="t")
        texts = self.radios(d)
        self.assertTrue(any("where both have a label, keep mine" in t for t in texts))
        self.assertTrue(any("use the imported one" in t for t in texts))
        self.assertTrue(any("Replace all my labels" in t for t in texts))
        self.assertEqual(d._policy_of["pose"](), LabelPolicy.MERGE_EXISTING)     # non-destructive default
        d._accept()
        self.assertEqual(sel.pose_policy, LabelPolicy.MERGE_EXISTING)

    def test_no_overlap_offers_a_single_combine_choice(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_pose_labels(self.dst, "t", self.files[3:5])                    # no overlap with frames 0-2
        d, _ = self.make(target_dataset="t")
        self.assertEqual(sum("Combine" in t for t in self.radios(d)), 1)

    def test_incompatible_labels_cannot_be_combined_and_default_to_skip(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_pose_labels(self.dst, "t", self.files[2:4], body_parts=["nose", "ear"])
        d, _ = self.make(target_dataset="t")
        merge = [r for r in d.findChildren(QRadioButton) if r.text().startswith("Combine")]
        self.assertTrue(merge and not any(r.isEnabled() for r in merge))
        self.assertEqual(d._policy_of["pose"](), LabelPolicy.SKIP)

    def test_wrong_dataset_blocks_import(self):
        self.add_dataset(self.dst, "wrong", ["q1.png", "q2.png"], media_root=self.tmp / "m")
        d, _ = self.make(target_dataset="wrong")
        self.assertFalse(d.ok_btn.isEnabled())

    def test_unmatched_labels_offer_a_keep_option_off_by_default(self):
        self.add_dataset(self.dst, "small", self.files[1:], media_root=self.tmp / "m")   # lacks frame_0000
        d, sel = self.make(target_dataset="small")
        self.assertTrue(d.keep_unmatched.isVisibleTo(d))
        self.assertFalse(d.keep_unmatched.isChecked())
        d._accept()
        self.assertFalse(sel.keep_unmatched)

    def test_no_datasets_at_all_explains_what_to_do(self):
        d, _ = self.make()
        self.assertFalse(d.ok_btn.isEnabled())
        self.assertIn("don't have any datasets", d.findChildren(type(d.dataset_combo))[0].parent().findChildren(__import__("PyQt6.QtWidgets", fromlist=["QLabel"]).QLabel)[0].text())

    def test_changing_the_dataset_rebuilds_the_report(self):
        self.add_dataset(self.dst, "a", self.files, media_root=self.tmp / "ma")
        self.add_dataset(self.dst, "b", ["q1.png"], media_root=self.tmp / "mb")
        d, _ = self.make(target_dataset="a")
        self.assertTrue(d.ok_btn.isEnabled())
        d.dataset_combo.setCurrentIndex(d.dataset_combo.findData("b"))
        self.assertFalse(d.ok_btn.isEnabled())

    def test_end_to_end_answers_drive_a_real_merge(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_pose_labels(self.dst, "t", self.files[2:5], x0=50.0)
        contents = bi.inspect_bundle(self.bundle)
        sel = bi.ImportSelection(pose=True, target_dataset="t")
        d = AttachLabelsDialog(self.dst, contents, sel)
        self.addCleanup(d.deleteLater)
        next(r for r in d.findChildren(QRadioButton) if "use the imported one" in r.text()).setChecked(True)
        d._accept()
        result = bi.import_from_bundle(self.dst, contents, sel)
        self.assertEqual((result["pose"]["mode"], result["pose"]["overlapping"], result["pose"]["total_frames"]), ("merged", 1, 5))


class ExportDialogTests(GuiCase):
    def dialog(self, dataset="d1"):
        d = ExportAnnotationsDialog(self.src, dataset, "proj")
        self.addCleanup(d.deleteLater)
        return d

    def test_defaults_include_everything_that_exists(self):
        d = self.dialog()
        self.assertEqual((d.include_project, d.include_media, d.include_pose, d.include_behavior), (True,) * 4)
        self.assertEqual(d.name_edit.text(), "d1_bundle")

    def test_unticking_media_suggests_a_labels_name_until_the_user_types_one(self):
        d = self.dialog()
        d.include_media_checkbox.setChecked(False)
        self.assertEqual(d.name_edit.text(), "d1_labels")
        d.name_edit.textEdited.emit("mine")
        d.name_edit.setText("mine")
        d.include_media_checkbox.setChecked(True)
        self.assertEqual(d.name_edit.text(), "mine")

    def test_preview_follows_the_ticks(self):
        d = self.dialog()

        def rows():
            return [d.contents_tree.topLevelItem(i).text(0) for i in range(d.contents_tree.topLevelItemCount())]
        self.assertTrue(any(r.startswith("project/") for r in rows()))
        d.include_project_checkbox.setChecked(False)
        d.include_pose_checkbox.setChecked(False)
        self.assertFalse(any(r.startswith("project/") or r.startswith("pose_labels/") for r in rows()))
        self.assertTrue(any(r.startswith("behavior_labels/") for r in rows()))

    def test_labels_that_dont_exist_cannot_be_included(self):
        self.add_dataset(self.src, "bare", self.files, media_root=self.tmp / "bare")
        d = self.dialog("bare")
        self.assertFalse(d.include_pose_checkbox.isEnabled())
        self.assertFalse(d.include_pose or d.include_behavior)
        self.assertIn("none for this dataset", d.include_pose_checkbox.text())

    def test_export_needs_something_ticked(self):
        d = self.dialog()
        d.dest_edit.setText(str(self.tmp / "out"))
        ok = d.button_box.button(d.button_box.StandardButton.Ok)
        self.assertTrue(ok.isEnabled())
        for cb in d._include_checkboxes:
            cb.setChecked(False)
        self.assertFalse(ok.isEnabled())

    def test_labels_only_export_produces_a_small_importable_bundle(self):
        d = self.dialog()
        d.include_media_checkbox.setChecked(False)
        d.include_project_checkbox.setChecked(False)
        out = self.export_bundle(
            self.src, "d1", name=d.name_edit.text(), include_media=d.include_media, include_project=d.include_project,
            include_pose=d.include_pose, include_behavior=d.include_behavior)
        self.assertFalse((out / "frames").exists())
        self.assertFalse((out / "project").exists())
        self.assertTrue((out / "pose_labels" / "labels.h5").exists())
        c = bi.inspect_bundle(out)                       # ...and it's a valid, importable export
        self.assertTrue(c.pose.ok and c.dataset.ok and not c.project.present)
        self.assertTrue(c.needs_media_folder or c.media_dir is not None)


class ResultMessageTests(GuiCase):
    describe = staticmethod(ActionHandler._describe_import)

    def test_full_import_message(self):
        result = bi.import_from_bundle(self.dst, bi.inspect_bundle(self.bundle),
                                       bi.ImportSelection(project=True, dataset=True, pose=True, behavior=True, dataset_name="d1"))
        text = self.describe(result)
        for expected in ("Project 'proj' added", "Dataset 'd1': 5 of 5 frames copied", "Pose labels added", "Behavior labels added"):
            self.assertIn(expected, text)

    def test_merge_message_reports_overlap_and_totals(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_pose_labels(self.dst, "t", self.files[2:5], x0=50.0)
        result = bi.import_from_bundle(self.dst, bi.inspect_bundle(self.bundle),
                                       bi.ImportSelection(pose=True, target_dataset="t", pose_policy=LabelPolicy.MERGE_EXISTING))
        text = self.describe(result)
        self.assertIn("Pose labels merged into 't'", text)
        self.assertIn("1 overlapped", text)
        self.assertIn("5 frames labeled in total", text)

    def test_skipped_and_dropped_labels_are_mentioned(self):
        self.add_dataset(self.dst, "small", self.files[1:], media_root=self.tmp / "m")
        result = bi.import_from_bundle(self.dst, bi.inspect_bundle(self.bundle),
                                       bi.ImportSelection(pose=True, behavior=True, target_dataset="small",
                                                          pose_policy=LabelPolicy.ADD, behavior_policy=LabelPolicy.SKIP))
        text = self.describe(result)
        self.assertIn("1 label(s) that matched no file were skipped", text)
        self.assertIn("Behavior labels not imported", text)
