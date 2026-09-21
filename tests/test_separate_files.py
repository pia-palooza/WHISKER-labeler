"""Import from separate files: choosing existing projects/datasets from lists instead of browsing.

Covers bundle_import.contents_from_files, the rebuilt ImportDatasetDialog, and the other dialogs that
now offer your existing project / dataset (one-pick import, "labels from other software").
"""
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog

from fixtures import WorkspaceCase, frame_names
from whisker.core import bundle_import as bi
from whisker.gui.dialogs import ImportBundleDialog, ImportDatasetDialog, ImportLabelsDialog
from whisker.gui.dialogs import import_bundle_dialog as ibd
from whisker.gui.dialogs import import_dataset_dialog as idd

_app = QApplication.instance() or QApplication([])


class FilesCase(WorkspaceCase):
    """A source workspace whose files we then treat as loose files that did not come from Export."""

    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.src = self.new_workspace("src")
        self.add_project(self.src, "proj")
        self.frames = frame_names(5)
        self.add_dataset(self.src, "d1", self.frames)
        self.add_pose_labels(self.src, "d1", self.frames[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.project_json = self.src.projects.base_dir / "proj.json"
        self.manifest = self.src.datasets.base_dir / "d1" / "manifest.json"
        self.media = self.tmp / "media" / "d1"
        self.pose_h5 = self.src.pose_labels.base_dir / "d1" / "labels.h5"
        self.beh_h5 = self.src.behavior_labels.base_dir / "d1" / "labels.h5"
        self.dst = self.new_workspace("dst")

    def rescan(self):
        self.dst.scan_projects(); self.dst.scan_datasets(); self.dst.scan_labels()

    def dialog(self, ws=None, active=None):
        d = ImportDatasetDialog(ws or self.dst, None, active)
        self.addCleanup(d.deleteLater)
        return d

    def run_import(self, d):
        """What the app does after the dialog: run the selection through the shared import."""
        with mock.patch.object(idd, "AttachLabelsDialog") as attach:
            attach.return_value.exec.return_value = QDialog.DialogCode.Accepted
            d._on_import_clicked()
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)
        result = bi.import_from_bundle(d._ws, d.contents, d.selection)
        self.rescan()
        return result


# ---------------------------------------------------------------- the core description


class ContentsFromFilesTests(FilesCase):
    def test_nothing_given_means_nothing_present(self):
        c = bi.contents_from_files()
        for part in (c.project, c.dataset, c.pose, c.behavior):
            self.assertFalse(part.present)
            self.assertEqual(part.summary, "Not chosen")

    def test_everything_given(self):
        c = bi.contents_from_files(self.project_json, self.manifest, self.media, self.pose_h5, self.beh_h5)
        for part in (c.project, c.dataset, c.pose, c.behavior):
            self.assertTrue(part.present and part.ok, part)
        self.assertEqual(c.dataset_name, "d1")
        self.assertEqual(c.media_dir, self.media)
        self.assertEqual(sorted(c.pose_keys), self.frames[:3])
        self.assertEqual(c.pose_body_parts, ["nose", "tail_base"])
        self.assertEqual(c.behavior_keys, ["a.mp4", "b.mp4"])
        self.assertEqual(c.behavior_names, ["groom", "rear"])
        self.assertFalse(c.needs_media_folder)

    def test_a_dataset_file_without_its_media_folder_asks_for_it(self):
        c = bi.contents_from_files(dataset_path=self.manifest)
        self.assertTrue(c.dataset.ok)
        self.assertTrue(c.needs_media_folder)
        self.assertIn("choose the folder", c.dataset.summary)

    def test_wrong_files_say_exactly_what_is_wrong(self):
        c = bi.contents_from_files(project_path=self.manifest, dataset_path=self.project_json, media_dir=self.tmp)
        self.assertFalse(c.project.ok)
        self.assertIn("dataset info file", c.project.problem)
        self.assertFalse(c.dataset.ok)
        self.assertIn("project file", c.dataset.problem)
        c = bi.contents_from_files(dataset_path=self.manifest, media_dir=self.tmp)          # right manifest, wrong folder
        self.assertFalse(c.dataset.ok)
        self.assertIn("not in this folder", c.dataset.problem)
        c = bi.contents_from_files(pose_path=self.tmp / "nope.h5", behavior_path=self.project_json)
        self.assertIn("File not found", c.pose.problem)
        self.assertFalse(c.behavior.ok)
        self.assertIn("Could not read", c.behavior.problem)

    def test_the_result_runs_through_the_same_import_as_an_export(self):
        c = bi.contents_from_files(self.project_json, self.manifest, self.media, self.pose_h5, self.beh_h5)
        sel = bi.ImportSelection(project=True, dataset=True, pose=True, behavior=True, dataset_name="d1")
        self.assertEqual(bi.validate_selection(self.dst, c, sel), [])
        r = bi.import_from_bundle(self.dst, c, sel)
        self.rescan()
        self.assertEqual((r["num_media_copied"], self.dst.pose_labels.has_pose_labels("d1")), (5, True))
        self.assertIn("proj", self.dst.projects.keys())

    def test_existing_project_plus_new_dataset_from_files_adds_no_project(self):
        self.add_project(self.dst, "mine")
        c = bi.contents_from_files(dataset_path=self.manifest, media_dir=self.media, pose_path=self.pose_h5)
        sel = bi.ImportSelection(project=True, project_mode="existing", existing_project="mine",
                                 dataset=True, dataset_name="d1", pose=True)
        bi.import_from_bundle(self.dst, c, sel)
        self.rescan()
        self.assertEqual(list(self.dst.projects.keys()), ["mine"])
        self.assertTrue(self.dst.pose_labels.has_pose_labels("d1"))


# ----------------------------------------------------------- the rebuilt dialog


class DialogDefaultsTests(FilesCase):
    def test_an_empty_workspace_can_only_add_from_files(self):
        d = self.dialog()
        self.assertTrue(d.project_file_radio.isChecked() and d.dataset_new_radio.isChecked())
        self.assertFalse(d.project_existing_radio.isEnabled())
        self.assertFalse(d.dataset_existing_radio.isEnabled())
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("Choose the project file", d.problem_label.text())

    def test_a_workspace_with_things_defaults_to_choosing_from_lists(self):
        self.add_project(self.dst, "proj")
        self.add_project(self.dst, "proj2", body_parts=["snout"], identities=["rat1"], behaviors=["dig"])
        self.add_dataset(self.dst, "mine", self.frames, media_root=self.tmp / "dm")
        d = self.dialog(active="proj2")
        self.assertTrue(d.project_existing_radio.isChecked() and d.dataset_existing_radio.isChecked())
        self.assertEqual(d.project_combo.currentData(), "proj2")                 # the active project comes first
        self.assertEqual(d.dataset_combo.currentData(), "mine")
        self.assertFalse(d.project_edit.isEnabled())                             # nothing to browse for
        self.assertFalse(d.dataset_edit.isEnabled())
        self.assertFalse(d.import_btn.isEnabled())                               # ...but nothing to import yet
        self.assertIn("Choose something to import", d.problem_label.text())

    def test_the_active_project_is_only_used_if_it_exists(self):
        self.add_project(self.dst, "proj")
        self.assertEqual(self.dialog(active="ghost").project_combo.currentData(), "proj")


class ExistingProjectAndDatasetTests(FilesCase):
    def setUp(self):
        super().setUp()
        self.add_project(self.dst, "mine")
        self.add_dataset(self.dst, "my_data", self.frames, media_root=self.tmp / "dm")
        self.add_dataset(self.dst, "unrelated", ["z1.png", "z2.png"], media_root=self.tmp / "dz")

    def test_labels_onto_an_existing_dataset_need_no_other_file(self):
        d = self.dialog(active="mine")
        d.pose_edit.setText(str(self.pose_h5))
        self.assertIn("3 labeled frame(s)", d.pose_status.text())
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.import_btn.text(), "Next...")                          # the combine step follows
        self.assertEqual(d.problem_label.text(), "")

    def test_datasets_are_listed_best_fit_first_once_labels_are_chosen(self):
        d = self.dialog()
        self.assertEqual([d.dataset_combo.itemData(i) for i in range(d.dataset_combo.count())], ["my_data", "unrelated"])   # A-Z
        d.pose_edit.setText(str(self.pose_h5))
        self.assertEqual(d.dataset_combo.itemData(0), "my_data")
        self.assertIn("3 of 3 labels match", d.dataset_combo.itemText(0))
        self.assertEqual(d.dataset_combo.currentData(), "my_data")                 # untouched: the default follows the best fit
        idx = d.dataset_combo.findData("unrelated")
        d.dataset_combo.setCurrentIndex(idx)
        d.dataset_combo.activated.emit(idx)                                        # the user picks one themselves
        d.behavior_edit.setText(str(self.beh_h5))                                  # ...and it survives re-ranking
        self.assertEqual(d.dataset_combo.currentData(), "unrelated")

    def test_the_followup_only_asks_how_to_combine_not_which_dataset(self):
        d = self.dialog()
        d.pose_edit.setText(str(self.pose_h5))
        with mock.patch.object(idd, "AttachLabelsDialog") as attach:
            attach.return_value.exec.return_value = QDialog.DialogCode.Rejected
            d._on_import_clicked()
            self.assertEqual(attach.call_args.kwargs, {"choose_target": False})
        self.assertNotEqual(d.result(), QDialog.DialogCode.Accepted)               # cancelling the follow-up keeps this open
        self.assertIsNone(d.selection)

    def test_the_whole_thing_imports_onto_the_chosen_dataset_and_adds_no_project(self):
        d = self.dialog(active="mine")
        d.pose_edit.setText(str(self.pose_h5))
        r = self.run_import(d)
        self.assertEqual(r["pose"]["mode"], "added")
        self.assertTrue(self.dst.pose_labels.has_pose_labels("my_data"))
        self.assertFalse(self.dst.pose_labels.has_pose_labels("unrelated"))
        self.assertEqual(list(self.dst.projects.keys()), ["mine"])                # the existing project was just *used*
        self.assertTrue(any("Using your existing project 'mine'" in n for n in r["notes"]))

    def test_behavior_labels_go_onto_an_existing_video_dataset(self):
        from whisker.core.study.dataset import DatasetType
        self.add_dataset(self.dst, "my_videos", ["a.mp4", "b.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "vv")
        d = self.dialog(active="mine")
        d.behavior_edit.setText(str(self.beh_h5))
        self.assertEqual(d.dataset_combo.itemData(0), "my_videos")                # the only one the labels fit
        self.assertIn("2 of 2 labels match", d.dataset_combo.itemText(0))
        r = self.run_import(d)
        self.assertEqual(r["behavior"]["mode"], "added")
        self.assertTrue(self.dst.behavior_labels.has_behavior_labels("my_videos"))

    def test_an_existing_project_that_lacks_what_the_labels_use_is_flagged_not_blocked(self):
        self.add_project(self.dst, "narrow", body_parts=["nose"], identities=["mouse1"], behaviors=["groom"])
        d = self.dialog(active="narrow")
        d.pose_edit.setText(str(self.pose_h5))
        self.assertIn("doesn't define body parts tail_base", d.project_note.text())
        self.assertTrue(d.import_btn.isEnabled())
        d.project_combo.setCurrentIndex(d.project_combo.findData("mine"))          # one that covers them
        self.assertIn("Nothing is added", d.project_note.text())

    def test_wrong_labels_are_refused_with_a_reason(self):
        d = self.dialog()
        d.pose_edit.setText(str(self.project_json))                                # a project file, not labels
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("Could not read", d.pose_status.text())
        self.assertIn("pose labels can't be imported", d.problem_label.text())


class AddingNewPiecesTests(FilesCase):
    def test_a_new_dataset_from_files_with_an_existing_project(self):
        self.add_project(self.dst, "mine")
        d = self.dialog(active="mine")
        d.dataset_new_radio.setChecked(True)
        self.assertTrue(d.dataset_edit.isEnabled() and not d.dataset_combo.isEnabled())
        d.dataset_edit.setText(str(self.manifest))
        self.assertEqual(d.dataset_name_edit.text(), "d1")                         # taken from the file
        self.assertFalse(d.import_btn.isEnabled())                                 # still needs its media
        self.assertIn("Choose the folder", d.problem_label.text())
        d.media_edit.setText(str(self.tmp))                                        # wrong folder
        self.assertIn("not in this folder", d.media_status.text())
        self.assertFalse(d.import_btn.isEnabled())
        d.media_edit.setText(str(self.media))
        self.assertIn("All 5 frames found", d.media_status.text())
        d.pose_edit.setText(str(self.pose_h5))
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.import_btn.text(), "Import")                            # labels travel with the new dataset
        r = self.run_import(d)
        self.assertEqual(r["num_media_copied"], 5)
        self.assertTrue(self.dst.pose_labels.has_pose_labels("d1"))
        self.assertEqual(list(self.dst.projects.keys()), ["mine"])

    def test_the_new_datasets_name_avoids_one_you_already_have_and_replace_is_offered_on_request(self):
        self.add_project(self.dst, "mine")
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "dm")
        d = self.dialog()
        d.dataset_new_radio.setChecked(True)
        d.dataset_edit.setText(str(self.manifest))
        d.media_edit.setText(str(self.media))
        self.assertEqual(d.dataset_name_edit.text(), "d1_2")
        self.assertTrue(d.import_btn.isEnabled())
        d.dataset_name_edit.setText("d1")
        self.assertFalse(d.import_btn.isEnabled())
        self.assertTrue(d.dataset_replace.isVisibleTo(d))
        d.dataset_replace.setChecked(True)
        self.assertTrue(d.import_btn.isEnabled())

    def test_a_project_can_still_be_added_from_a_file_under_a_free_name(self):
        self.add_project(self.dst, "proj")
        d = self.dialog()
        d.project_file_radio.setChecked(True)
        d.project_edit.setText(str(self.project_json))
        self.assertIn("Project 'proj'", d.project_status.text())
        self.assertEqual(d.project_name_edit.text(), "proj_2")
        d.pose_edit.setText(str(self.pose_h5))
        d.dataset_existing_radio.setEnabled(False)
        d.dataset_new_radio.setChecked(True)
        d.dataset_edit.setText(str(self.manifest))
        d.media_edit.setText(str(self.media))
        self.assertTrue(d.import_btn.isEnabled())
        self.run_import(d)
        self.assertEqual(sorted(self.dst.projects.keys()), ["proj", "proj_2"])
        self.assertEqual(self.dst.projects.get("proj_2").name, "proj_2")

    def test_a_typed_name_is_not_overwritten_by_the_files_name(self):
        d = self.dialog()
        d.dataset_new_radio.setChecked(True)
        d.dataset_name_edit.textEdited.emit("my own")
        d.dataset_name_edit.setText("my own")
        d.dataset_edit.setText(str(self.manifest))
        self.assertEqual(d.dataset_name_edit.text(), "my own")


# ------------------------------------------------ other dialogs offer your existing things


class OtherDialogsTests(FilesCase):
    def test_the_one_pick_dialog_defaults_its_project_list_to_the_active_project(self):
        export = self.export_bundle(self.src, "d1")
        self.add_project(self.dst, "aaa_other", body_parts=["x"], identities=["y"], behaviors=["z"])
        self.add_project(self.dst, "my_active", body_parts=["nose", "tail_base"], identities=["mouse1"], behaviors=["groom", "rear"])
        d = ImportBundleDialog(self.dst, None, "my_active")
        self.addCleanup(d.deleteLater)
        d.set_path(str(export))
        d.project_existing_radio.setChecked(True)
        self.assertEqual(d.project_combo.currentData(), "my_active")             # not just the first alphabetically

    def test_the_button_says_what_it_does(self):
        d = ImportBundleDialog(self.dst)
        self.addCleanup(d.deleteLater)
        self.assertEqual(d.manual_btn.text(), "Import from separate files...")
        self.assertIn("existing project and dataset", d.manual_btn.toolTip())
        with mock.patch.object(d, "done") as done:
            d.manual_btn.click()
        done.assert_called_once_with(ibd.ADVANCED_RESULT)

    def test_a_folder_that_is_not_an_export_points_at_the_button_by_name(self):
        loose = self.tmp / "loose"
        loose.mkdir()
        (loose / "manifest.json").write_text("{}")
        self.assertIn("Import from separate files", bi.locate_bundle(loose).message)

    def test_labels_from_other_software_lists_your_datasets_and_your_active_project(self):
        self.add_project(self.dst, "aaa_first")
        self.add_project(self.dst, "my_active")
        self.add_dataset(self.dst, "existing_one", ["a.png"], media_root=self.tmp / "e1")
        d = ImportLabelsDialog(self.dst, None, "my_active")
        self.addCleanup(d.deleteLater)
        self.assertEqual([d.name_combo.itemText(i) for i in range(d.name_combo.count())], ["existing_one"])
        self.assertEqual(d.selected_project_name, "my_active")
        d.path_edit.setText("some/path")
        self.assertFalse(d.button_box.button(d.button_box.StandardButton.Ok).isEnabled())      # no dataset chosen yet
        d.name_combo.setCurrentIndex(0)                                            # pick one of yours
        self.assertEqual(d.selected_dataset_name, "existing_one")
        self.assertTrue(d.button_box.button(d.button_box.StandardButton.Ok).isEnabled())
        d.name_combo.setEditText("a brand new one")                                # ...or type a new name
        self.assertEqual(d.selected_dataset_name, "a brand new one")

    def test_browsing_for_labels_does_not_overwrite_a_dataset_you_already_chose(self):
        self.add_project(self.dst, "p")
        self.add_dataset(self.dst, "existing_one", ["a.png"], media_root=self.tmp / "e1")
        d = ImportLabelsDialog(self.dst, None)
        self.addCleanup(d.deleteLater)
        d.name_combo.setCurrentIndex(0)
        with mock.patch("whisker.gui.dialogs.import_labels_dialog.QFileDialog.getExistingDirectory", return_value=str(self.tmp / "labels_from_mars")):
            d._on_browse_clicked()
        self.assertEqual(d.selected_dataset_name, "existing_one")
        d2 = ImportLabelsDialog(self.dst, None)                                    # nothing chosen: the file's name is offered
        self.addCleanup(d2.deleteLater)
        with mock.patch("whisker.gui.dialogs.import_labels_dialog.QFileDialog.getExistingDirectory", return_value=str(self.tmp / "labels_from_mars")):
            d2._on_browse_clicked()
        self.assertEqual(d2.selected_dataset_name, "labels_from_mars")
