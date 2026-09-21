"""Click-through tests for the Whisker bundle import and export dialogs (headless Qt, real bundles)."""
import os
from datetime import datetime
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from bundle_helpers import BundleCase
from fixtures import frame_names
from whisker.core import whisker_bundle as wb
from whisker.core.study.dataset import DatasetType
from whisker.gui.dialogs import ExportWhiskerBundleDialog, ImportWhiskerBundleDialog
from whisker.gui.dialogs import import_whisker_bundle_dialog as import_mod
from whisker.gui.dialogs.export_whisker_bundle_dialog import suggest_filename

_app = QApplication.instance() or QApplication([])


class ImportDialogTests(BundleCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.ws = self.new_workspace("ws")

    def dialog(self, path=None):
        d = ImportWhiskerBundleDialog(self.ws, path=str(path) if path else None)
        self.addCleanup(d.deleteLater)
        self.addCleanup(d._close_bundle)           # the app always closes the dialog, which releases the zip; a test must too
        return d

    def states(self, d):
        return {k: (row.checkState(0) == Qt.CheckState.Checked, row.text(2)) for k, row in d._rows.items()}

    def test_picking_a_bundle_lists_it_and_ticks_what_is_new(self):
        d = self.dialog(self.make_bundle(project="proj"))
        self.assertEqual(self.states(d), {
            "dataset:ds": (True, "New"), "labels:ds": (True, "New"), "project:proj": (True, "New")})
        self.assertTrue(d.import_btn.isEnabled())
        self.assertFalse(d.media_row.isVisibleTo(d))               # a frame subset needs no media folder
        self.assertIn("Made by Whisker 0.8.0", d.status_label.text())

    def test_what_the_workspace_already_has_is_marked_and_left_unticked(self):
        self.add_dataset(self.ws, "ds", frame_names(2))
        d = self.dialog(self.make_bundle())
        self.assertEqual(self.states(d)["dataset:ds"], (False, "Already here (tick to replace)"))
        self.assertEqual(self.states(d)["labels:ds"], (True, "New"))

    def test_ticking_something_that_exists_says_it_will_be_replaced_and_asks_first(self):
        self.add_dataset(self.ws, "ds", frame_names(2))
        d = self.dialog(self.make_bundle())
        d._rows["dataset:ds"].setCheckState(0, Qt.CheckState.Checked)
        self.assertIn("will be replaced", d.summary_label.text())
        with mock.patch.object(import_mod.QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as q:
            d.accept()
        q.assert_called_once()
        self.assertEqual(d.result(), 0)                                # declining leaves the dialog open
        with mock.patch.object(import_mod.QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            d.accept()
        self.assertEqual(d.result(), 1)

    def test_no_question_when_nothing_is_replaced(self):
        d = self.dialog(self.make_bundle())
        with mock.patch.object(import_mod.QMessageBox, "question") as q:
            d.accept()
        q.assert_not_called()
        self.assertEqual(d.result(), 1)

    def test_videos_ask_for_a_folder_and_import_waits_for_a_usable_one(self):
        d = self.dialog(self.make_bundle(dataset="vids", type_="VIDEO_COLLECTION", files=("v1.mp4",)))
        self.assertTrue(d.media_row.isVisibleTo(d))
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("Choose the folder", d.summary_label.text())
        d.media_edit.setText(str(self.tmp / "no_such_folder"))
        self.assertFalse(d.import_btn.isEnabled())
        folder = self.tmp / "videos"
        folder.mkdir()
        d.media_edit.setText(str(folder))
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.media_root, folder)

    def test_unticking_the_videos_takes_the_folder_question_away(self):
        d = self.dialog(self.make_bundle(dataset="vids", type_="VIDEO_COLLECTION", files=("v1.mp4",)))
        d._rows["dataset:vids"].setCheckState(0, Qt.CheckState.Unchecked)
        self.assertFalse(d.media_row.isVisibleTo(d))
        # Its labels are still ticked but the dataset isn't coming, so they have nowhere to go: that is said, not silently dropped.
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("nowhere to go", d.summary_label.text())
        d._rows["labels:vids"].setCheckState(0, Qt.CheckState.Unchecked)
        self.assertIn("Nothing is ticked", d.summary_label.text())

    def test_a_bad_file_says_why_and_offers_nothing(self):
        junk = self.tmp / "junk.zip"
        junk.write_bytes(b"not a zip")
        d = self.dialog(junk)
        self.assertEqual(d.tree.topLevelItemCount(), 0)
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("isn't a readable zip", d.status_label.text())
        self.assertIsNone(d.bundle_path)

    def test_a_newer_bundle_is_explained(self):
        d = self.dialog(self.make_bundle(extra_manifest={"format_version": 9}))
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("newer", d.status_label.text())

    def test_the_choice_survives_closing_and_the_zip_is_released(self):
        """The job reopens the file on its own thread; the dialog must not keep it locked (Windows) or forget the choice."""
        path = self.make_bundle(project="proj")
        d = self.dialog(path)
        d._rows["project:proj"].setCheckState(0, Qt.CheckState.Unchecked)
        d.accept()
        self.assertEqual((d.bundle_path, d.chosen), (path, frozenset({"dataset:ds", "labels:ds"})))
        path.unlink()                                                  # would fail if the dialog still held it open

    def test_choosing_another_bundle_replaces_the_list(self):
        d = self.dialog(self.make_bundle("one.zip", dataset="one"))
        d.set_path(str(self.make_bundle("two.zip", dataset="two", labels=False)))
        self.assertEqual(set(d._rows), {"dataset:two"})


class ExportDialogTests(BundleCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.ws = self.new_workspace("src")
        self.add_project(self.ws, "proj")
        self.add_project(self.ws, "other")
        self.add_dataset(self.ws, "d1", frame_names(3), DatasetType.FRAME_SUBSET)
        self.add_dataset(self.ws, "d2", ["v.mp4"], DatasetType.VIDEO_COLLECTION)
        self.add_pose_labels(self.ws, "d1", frame_names(2))
        run = self.ws.base_dir / "workflows/pose_estimation/predictions/run1/d1"
        run.mkdir(parents=True)
        (run / "predictions.h5").write_bytes(b"p" * 50)

    def dialog(self, preselect=None, active="proj"):
        d = ExportWhiskerBundleDialog(self.ws, active, preselect)
        self.addCleanup(d.deleteLater)
        d._default_dir = self.tmp
        return d

    def tick(self, widget, name, on=True):
        for i in range(widget.count()):
            if widget.item(i).data(Qt.ItemDataRole.UserRole) == name:
                widget.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                return
        self.fail(f"{name} isn't listed")

    def test_the_dataset_being_looked_at_is_ticked_and_the_active_project_comes_along(self):
        d = self.dialog(preselect="d1")
        self.assertEqual((d.selected_datasets(), d.selected_projects()), (["d1"], ["proj"]))

    def test_nothing_ticked_means_nothing_to_export(self):
        d = self.dialog()
        self.assertFalse(d.export_btn.isEnabled())
        self.assertIn("at least one dataset", d.summary_label.text())

    def test_export_is_ready_once_something_is_ticked_and_a_folder_is_named(self):
        d = self.dialog(preselect="d1")
        d.dest_edit.setText(str(self.tmp / "out.zip"))
        self.assertTrue(d.export_btn.isEnabled())
        self.assertEqual(d.destination, self.tmp / "out.zip")
        self.assertEqual({e[0].split("/")[0] for e in d.plan.entries}, {"datasets", "workflows", "projects"})
        d.dest_edit.setText(str(self.tmp / "no_such_folder" / "out.zip"))
        self.assertFalse(d.export_btn.isEnabled())

    def test_the_file_name_follows_the_choice_until_the_user_types_one(self):
        d = self.dialog()
        self.tick(d.datasets_list, "d1")
        self.assertEqual(Path(d.dest_edit.text()).name, suggest_filename(["d1"]))
        self.tick(d.datasets_list, "d2")
        self.assertIn("2_datasets", d.dest_edit.text())
        d.dest_edit.textEdited.emit("mine.zip")                        # the user typed
        d.dest_edit.setText(str(self.tmp / "mine.zip"))
        self.tick(d.datasets_list, "d2", on=False)
        self.assertEqual(Path(d.dest_edit.text()).name, "mine.zip")

    def test_a_missing_dot_zip_is_added(self):
        d = self.dialog(preselect="d1")
        d.dest_edit.setText(str(self.tmp / "bundle"))
        self.assertEqual(d.destination, self.tmp / "bundle.zip")

    def test_prediction_runs_are_offered_for_the_ticked_datasets_and_are_off_by_default(self):
        d = self.dialog()
        self.assertFalse(d.runs_box.isVisibleTo(d))
        self.tick(d.datasets_list, "d1")
        self.assertTrue(d.runs_box.isVisibleTo(d))
        self.assertEqual(d.selected_runs(), [])
        self.tick(d.runs_list, "pose_estimation/run1")
        self.assertEqual(d.selected_runs(), [("pose_estimation", "run1")])
        d.dest_edit.setText(str(self.tmp / "out.zip"))
        self.assertTrue(any("predictions/run1" in e[0] for e in d.plan.entries))
        self.tick(d.datasets_list, "d2")                               # still ticked after the list is rebuilt
        self.assertEqual(d.selected_runs(), [("pose_estimation", "run1")])

    def test_unreachable_media_are_warned_about_but_do_not_block_the_export(self):
        (Path(self.ws.datasets.get("d1").base_data_path) / frame_names(3)[1]).unlink()
        d = self.dialog(preselect="d1")
        d.dest_edit.setText(str(self.tmp / "out.zip"))
        self.assertIn("can't be found", d.summary_label.text())
        self.assertTrue(d.export_btn.isEnabled())
        self.assertEqual(d.plan.media_missing, 1)

    def test_accepting_without_a_plan_explains_instead_of_closing(self):
        d = self.dialog()
        with mock.patch("whisker.gui.dialogs.export_whisker_bundle_dialog.QMessageBox") as box:
            d.accept()
        box.warning.assert_called_once()
        self.assertEqual(d.result(), 0)

    def test_the_result_is_ready_for_the_export_job(self):
        d = self.dialog(preselect="d1")
        d.dest_edit.setText(str(self.tmp / "out.zip"))
        d.accept()
        result = wb.export_bundle(self.ws, d.plan, d.destination)
        self.assertEqual((result["problems"], result["datasets"]), ([], ["d1"]))
        self.assertTrue((self.tmp / "out.zip").is_file())

    # -- what will be in the bundle -----------------------------------------------------------------
    def contents(self, d):
        """The 'What will be in the bundle' tree as {top-level title: [(child title, files text, size text)]}."""
        tree, out = d.contents_tree, {}
        for i in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(i)
            out[top.text(0)] = [(top.child(j).text(0), top.child(j).text(1), top.child(j).text(2)) for j in range(top.childCount())]
        return out

    def test_nothing_ticked_says_what_to_do_instead_of_showing_an_empty_box(self):
        d = self.dialog()
        self.assertEqual(list(self.contents(d)), ["Tick a dataset above to see everything that will be exported."])

    def test_ticking_a_dataset_lists_its_manifest_media_and_labels_underneath(self):
        d = self.dialog()
        self.tick(d.datasets_list, "d1")
        contents = self.contents(d)
        self.assertEqual([title for title, _f, _s in contents["d1"]],
                         ["Dataset manifest", "Media — 3 frames", "Labels — pose estimation"])
        by_title = {title: (files, size) for title, files, size in contents["d1"]}
        self.assertEqual(by_title["Dataset manifest"][0], "manifest.json")
        self.assertEqual(by_title["Labels — pose estimation"][0], "labels.h5, metadata.json")
        self.assertTrue(all(size for _f, size in by_title.values()))
        self.assertTrue(d.contents_tree.topLevelItem(1).text(2))                   # the dataset's total
        self.assertIn("Bundle description", contents)

    def test_the_active_project_is_listed_and_a_dataset_without_labels_says_so(self):
        d = self.dialog(active="proj")
        self.tick(d.datasets_list, "d2")
        contents = self.contents(d)
        self.assertEqual(contents["Projects"], [("Project — proj", "proj.json", contents["Projects"][0][2])])
        labels = next(row for row in contents["d2"] if row[0] == "Labels")
        self.assertIn("none for this dataset", labels[1])

    def test_what_exists_but_is_not_ticked_is_listed_as_not_included(self):
        d = self.dialog(active=None)
        self.tick(d.datasets_list, "d1")
        not_included = [title for title, _f, _s in self.contents(d)["Not included"]]
        self.assertEqual(len(not_included), 2)
        self.assertTrue(any("run1" in t and "tick it above" in t for t in not_included))
        self.assertTrue(any("Project definitions" in t for t in not_included))

    def test_ticking_a_prediction_run_or_project_moves_it_into_the_bundle(self):
        d = self.dialog(active=None)
        self.tick(d.datasets_list, "d1")
        self.tick(d.runs_list, "pose_estimation/run1")
        self.tick(d.projects_list, "other")
        contents = self.contents(d)
        self.assertIn("Prediction results — run1", [t for t, _f, _s in contents["d1"]])
        self.assertIn("Project — other", [t for t, _f, _s in contents["Projects"]])
        self.assertNotIn("Not included", contents)

    def test_unticking_the_dataset_clears_it_again(self):
        d = self.dialog(preselect="d1")
        self.assertIn("d1", self.contents(d))
        self.tick(d.datasets_list, "d1", on=False)
        self.assertEqual(list(self.contents(d)), ["Tick a dataset above to see everything that will be exported."])

    def test_media_that_cannot_be_found_are_flagged_in_the_list_too(self):
        (Path(self.ws.datasets.get("d1").base_data_path) / frame_names(3)[1]).unlink()
        d = self.dialog(preselect="d1")
        media = next(row for row in self.contents(d)["d1"] if row[0].startswith("Media"))
        self.assertEqual(media[0], "Media — 2 frames")
        self.assertIn("1 not found on disk", media[1])

    def test_every_file_in_the_plan_is_accounted_for_in_the_list(self):
        d = self.dialog(preselect="d1")
        self.tick(d.runs_list, "pose_estimation/run1")
        shown = sum(g.bytes for g in d.plan.groups())
        self.assertEqual(shown, d.plan.total_bytes)


class Filenames(BundleCase):
    def test_the_suggested_name_follows_whiskers_style(self):
        """Brackets are turned into underscores, as in the bundle WHISKER made for this dataset."""
        self.assertEqual(
            suggest_filename(["20260803_Pharm_JI_trainingdataset [Manual Samples]"], datetime(2026, 9, 21)),
            "whisker_bundle_20260803_Pharm_JI_trainingdataset _Manual Samples__20260921.zip")
        self.assertEqual(suggest_filename(["a", "b", "c"], datetime(2026, 9, 21)), "whisker_bundle_3_datasets_20260921.zip")
        self.assertEqual(suggest_filename(['we/ird:na*me?'], datetime(2026, 9, 21)), "whisker_bundle_we_ird_na_me__20260921.zip")
