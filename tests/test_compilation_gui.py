"""Click-through tests for the compilation dialogs (headless Qt, real workspaces and packages)."""
import os
import shutil
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from fixtures import WorkspaceCase, frame_names
from whisker.core import bundle_import as bi
from whisker.core import compilation as comp
from whisker.core.compilation import CompilationItem
from whisker.core.study.dataset import DatasetType
from whisker.gui.dialogs import (
    COMPILATION_RESULT, ExportCompilationDialog, ImportBundleDialog, ImportCompilationDialog,
)
from whisker.gui.dialogs import import_compilation_dialog as icd
from whisker.gui.widgets.data_explorer.action_handler import ActionHandler

_app = QApplication.instance() or QApplication([])
CHECKED, UNCHECKED = Qt.CheckState.Checked, Qt.CheckState.Unchecked


class GuiCase(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.src = self.new_workspace("src")
        self.add_project(self.src, "proj")
        self.add_project(self.src, "proj2", body_parts=["snout", "tail"], identities=["rat1"], behaviors=["dig"])
        self.frames = frame_names(5)
        self.add_dataset(self.src, "d1", self.frames)
        self.add_pose_labels(self.src, "d1", self.frames[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.add_dataset(self.src, "vids", ["a.mp4", "b.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "vmedia")
        self.add_behavior_labels(self.src, "vids", ["a.mp4", "b.mp4"])
        self.add_dataset(self.src, "rats", frame_names(3), media_root=self.tmp / "rats")
        self.add_pose_labels(self.src, "rats", frame_names(3)[:2], body_parts=["snout", "tail"], individuals=["rat1"])
        self.dst = self.new_workspace("dst")

    def package(self, items=None, name="pack") -> Path:
        items = items or [CompilationItem("d1", "proj"), CompilationItem("vids", "proj"), CompilationItem("rats", "proj2")]
        plans = comp.build_plans(self.src, items)
        return Path(comp.export_compilation(items, plans, self.tmp / "out", name)["compilation_dir"])

    def rescan(self):
        self.dst.scan_projects(); self.dst.scan_datasets(); self.dst.scan_labels()


# ----------------------------------------------------------------- export


class ExportDialogTests(GuiCase):
    def dialog(self, preselect=None, ws=None, default_project=None):
        d = ExportCompilationDialog(ws or self.src, default_project, preselect)
        self.addCleanup(d.deleteLater)
        return d

    def row(self, d, name):
        return next(r for r in d._rows if r["name"] == name)

    def test_lists_every_dataset_and_ticks_none_by_default(self):
        d = self.dialog()
        self.assertEqual([r["name"] for r in d._rows], ["d1", "rats", "vids"])
        self.assertEqual(d.items(), [])
        self.assertFalse(d.ok_btn.isEnabled())

    def test_the_selected_dataset_is_preticked(self):
        d = self.dialog(preselect="d1")
        self.assertEqual([i.dataset_name for i in d.items()], ["d1"])
        self.assertTrue(d.ok_btn.isEnabled())

    def test_ticking_a_dataset_turns_on_only_the_parts_it_has(self):
        d = self.dialog()
        vids = self.row(d, "vids")
        self.assertIsNone(vids["media"].data(Qt.ItemDataRole.CheckStateRole))       # no checkbox while unticked
        vids["include"].setCheckState(CHECKED)
        self.assertEqual(vids["media"].checkState(), CHECKED)
        self.assertEqual(vids["behavior"].checkState(), CHECKED)
        self.assertIsNone(vids["pose"].data(Qt.ItemDataRole.CheckStateRole))        # vids has no pose labels
        item = d.items()[0]
        self.assertEqual((item.include_media, item.include_pose, item.include_behavior), (True, False, True))

    def test_parts_can_be_unticked_per_dataset(self):
        d = self.dialog(preselect="d1")
        d1 = self.row(d, "d1")
        d1["media"].setCheckState(UNCHECKED)
        d1["behavior"].setCheckState(UNCHECKED)
        item = d.items()[0]
        self.assertEqual((item.include_media, item.include_pose, item.include_behavior), (False, True, False))

    def test_unticking_a_dataset_clears_its_parts(self):
        d = self.dialog(preselect="d1")
        self.row(d, "d1")["include"].setCheckState(UNCHECKED)
        self.assertEqual(d.items(), [])
        self.assertFalse(self.row(d, "d1")["combo"].isEnabled())

    def test_the_project_is_guessed_from_the_labels(self):
        d = self.dialog(default_project="proj")
        self.assertEqual(self.row(d, "d1")["combo"].currentText(), "proj")
        self.assertEqual(self.row(d, "rats")["combo"].currentText(), "proj2")       # snout/tail/rat1 only fit proj2
        self.row(d, "rats")["include"].setCheckState(CHECKED)
        self.assertEqual(d.items()[0].project_name, "proj2")

    def test_tick_all_and_untick_all(self):
        d = self.dialog()
        d._tick_all(True)
        self.assertEqual([i.dataset_name for i in d.items()], ["d1", "rats", "vids"])
        d._tick_all(False)
        self.assertEqual(d.items(), [])

    def test_the_summary_counts_files_that_will_be_copied(self):
        d = self.dialog()
        d._tick_all(True)
        self.assertIn("3 dataset(s) ticked", d.summary.text())
        self.assertIn("10 media file(s)", d.summary.text())
        for r in d._rows:
            r["media"].setCheckState(UNCHECKED)
        self.assertNotIn("media file", d.summary.text())

    def test_the_name_is_checked(self):
        d = self.dialog(preselect="d1")
        d.dest_edit.setText(str(self.tmp / "o"))
        self.assertTrue(d.ok_btn.isEnabled())
        d.name_edit.setText("what?")
        self.assertFalse(d.ok_btn.isEnabled())
        self.assertIn("can't contain", d.notice.text())
        d.name_edit.setText("")
        self.assertFalse(d.ok_btn.isEnabled())
        d.name_edit.setText("good name")
        self.assertTrue(d.ok_btn.isEnabled())
        self.assertTrue(d.full_path.text().endswith("good name"))

    def test_a_workspace_with_no_projects_explains_itself(self):
        ws = self.new_workspace("noproj")
        self.add_dataset(ws, "x", ["a.png"], media_root=self.tmp / "x")
        d = self.dialog(preselect="x", ws=ws)
        self.assertIn("no projects", d.notice.text())
        self.assertFalse(d.ok_btn.isEnabled())

    def test_what_the_dialog_returns_exports_and_imports_back(self):
        d = self.dialog(preselect="d1")
        d._tick_all(True)
        self.row(d, "vids")["media"].setCheckState(UNCHECKED)
        d.dest_edit.setText(str(self.tmp / "out"))
        d.name_edit.setText("from-dialog")
        items = d.items()
        r = comp.export_compilation(items, comp.build_plans(self.src, items), d.destination_dir, d.name)
        root = Path(r["compilation_dir"])
        c = comp.inspect_compilation(root)
        self.assertEqual([e.name for e in c.entries], ["d1", "rats", "vids"])
        self.assertFalse(c.entry("vids").contents.media_included)
        self.assertTrue(c.entry("d1").contents.media_included)


# ------------------------------------------------------- import: detection


class ImportBundleDialogTests(GuiCase):
    def dialog(self, path):
        d = ImportBundleDialog(self.dst)
        self.addCleanup(d.deleteLater)
        d.set_path(str(path))
        return d

    def test_a_compilation_is_recognised_and_offers_to_choose_datasets(self):
        root = self.package()
        d = self.dialog(root)
        self.assertEqual(d.compilation_root, root)
        self.assertIn("compilation 'pack'", d.location_label.text())
        self.assertIn("3 dataset(s)", d.location_label.text())
        self.assertEqual(d.import_btn.text(), "Choose datasets...")
        self.assertTrue(d.import_btn.isEnabled())
        self.assertFalse(d.parts_box.isVisibleTo(d))

    def test_clicking_it_asks_the_caller_to_open_the_compilation_dialog(self):
        d = self.dialog(self.package())
        with mock.patch.object(d, "done") as done:
            d._on_import_clicked()
        done.assert_called_once_with(COMPILATION_RESULT)

    def test_picking_an_inner_dataset_stays_a_single_import(self):
        root = self.package()
        d = self.dialog(root / "d1")
        self.assertIsNone(d.compilation_root)
        self.assertEqual(d.import_btn.text(), "Import")
        self.assertTrue(d.parts_box.isVisibleTo(d))
        self.assertEqual(d.contents.dataset_name, "d1")

    def test_a_folder_holding_a_compilation_finds_it(self):
        root = self.package()
        d = self.dialog(root.parent)
        self.assertEqual(d.compilation_root, root)

    def test_moving_on_from_a_compilation_restores_the_button(self):
        d = self.dialog(self.package())
        nothing = self.tmp / "nothing_here"
        nothing.mkdir()
        d.set_path(str(nothing))
        self.assertIsNone(d.compilation_root)
        self.assertEqual(d.import_btn.text(), "Import")
        self.assertFalse(d.import_btn.isEnabled())

    def test_a_folder_of_two_compilations_asks_which(self):
        a = self.package(name="a")
        shutil.copytree(a, self.tmp / "out" / "b")
        d = self.dialog(self.tmp / "out")
        self.assertTrue(d.candidate_combo.isVisibleTo(d))
        d._on_candidate_chosen(d.candidate_combo.findData(str(a)))
        self.assertEqual(d.compilation_root, a)


# ------------------------------------------------------- import: the table


class ImportCompilationDialogTests(GuiCase):
    def dialog(self, root, ws=None):
        d = ImportCompilationDialog(ws or self.dst, root)
        self.addCleanup(d.deleteLater)
        return d

    def row(self, d, name):
        return next(r for r in d._rows if r["entry"].name == name)

    def cell(self, d, name, key):
        return self.row(d, name)["cells"][key]

    def test_a_fresh_workspace_ticks_everything_and_is_ready(self):
        d = self.dialog(self.package())
        self.assertEqual([r["entry"].name for r in d._rows], ["d1", "vids", "rats"])
        for name, expected in {"d1": (1, 1, 1), "vids": (1, 0, 1), "rats": (1, 1, 0)}.items():
            got = tuple(int(d._ticked(self.cell(d, name, k))) for k in ("media", "pose", "behavior"))
            self.assertEqual(got, expected, name)
            self.assertEqual(self.row(d, name)["use"].currentData(), "", name)          # all "add as new"
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.problem_label.text(), "")
        self.assertTrue(d.projects_check.isChecked())
        self.assertEqual(sorted(d._project_rows), ["proj", "proj2"])
        for name in ("proj", "proj2"):
            pr = d._project_rows[name]
            self.assertEqual((pr["mode"].currentData(), pr["name"].text(), pr["name"].isEnabled()), ("new", name, True))
        self.assertFalse(d.policy_box.isVisibleTo(d))
    def test_the_header_names_the_compilation(self):
        d = self.dialog(self.package(name="my pack"))
        self.assertIn("my pack", d.header.text())
        self.assertIn("3 dataset(s)", d.header.text())

    def test_datasets_you_have_start_labels_only_with_a_match_report(self):
        self.add_dataset(self.dst, "d1", self.frames[1:], media_root=self.tmp / "m")        # lacks frame_0000
        d = self.dialog(self.package())
        self.assertFalse(d._ticked(self.cell(d, "d1", "media")))
        self.assertTrue(d._ticked(self.cell(d, "d1", "pose")))
        self.assertFalse(bool(self.row(d, "d1")["as_item"].flags() & Qt.ItemFlag.ItemIsEditable))   # name irrelevant now
        notes = self.row(d, "d1")["notes"].text()
        self.assertIn("Pose: 2/3 frames match (1 skipped)", notes)
        self.assertIn("Behavior: none of 2 videos match", notes)
        self.assertTrue(d.import_btn.isEnabled())

    def test_the_combine_choice_only_appears_when_existing_labels_are_involved(self):
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "m")
        root = self.package()
        d = self.dialog(root)
        self.assertFalse(d.policy_box.isVisibleTo(d))                 # d1 has no labels yet
        self.add_pose_labels(self.dst, "d1", self.frames[2:4], x0=50.0)
        d2 = self.dialog(root)
        self.assertTrue(d2.policy_box.isVisibleTo(d2))
        self.assertEqual(d2.selection.existing_labels_policy, comp.LabelPolicy.MERGE_EXISTING)   # the safe default
        d2._policy_radios[comp.LabelPolicy.REPLACE].setChecked(True)
        self.assertEqual(d2.selection.existing_labels_policy, comp.LabelPolicy.REPLACE)

    def test_a_name_clash_is_flagged_until_renamed(self):
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "m")
        d = self.dialog(self.package())
        row = self.row(d, "d1")
        self.assertEqual(row["use"].currentData(), "d1")                          # starts on "use mine"
        row["use"].setCurrentIndex(0)                                              # the user chooses "add as new"
        self.assertTrue(d._ticked(self.cell(d, "d1", "media")))                    # media tick back on
        self.assertEqual(row["as_item"].text(), "d1_2")                            # a free name was suggested
        self.assertTrue(d.import_btn.isEnabled())
        row["as_item"].setText("d1")
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("already have a dataset called 'd1'", row["notes"].text())
        self.assertIn("'d1':", d.problem_label.text())
        row["as_item"].setText("brand new")
        self.assertTrue(d.import_btn.isEnabled())
    def test_two_rows_cannot_share_a_new_name(self):
        d = self.dialog(self.package())
        self.row(d, "vids")["as_item"].setText("d1")
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("both be imported as 'd1'", d.problem_label.text())

    def test_labels_for_a_dataset_you_dont_have_need_a_dataset_to_go_on(self):
        d = self.dialog(self.package())
        self.cell(d, "d1", "media").setCheckState(UNCHECKED)                       # neither add-as-new nor use-existing
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("These labels need a dataset", self.row(d, "d1")["notes"].text())
    def test_datasets_you_have_start_on_use_existing_with_the_media_switched_off(self):
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "m")
        d = self.dialog(self.package())
        row = self.row(d, "d1")
        self.assertEqual(row["use"].currentData(), "d1")
        self.assertFalse(bool(row["cells"]["media"].flags() & Qt.ItemFlag.ItemIsUserCheckable))    # nothing is copied
        self.assertFalse(bool(row["as_item"].flags() & Qt.ItemFlag.ItemIsEditable))
        self.assertEqual(self.row(d, "vids")["use"].currentData(), "")             # no such dataset here: add as new
        row["use"].setCurrentIndex(0)
        self.assertTrue(d._ticked(row["cells"]["media"]))
        self.assertTrue(bool(row["as_item"].flags() & Qt.ItemFlag.ItemIsEditable))
        row["use"].setCurrentIndex(row["use"].findData("d1"))
        self.assertFalse(d._ticked(row["cells"]["media"]))
        self.assertEqual(d.selection.items["d1"].target_dataset, "d1")

    def test_a_row_can_use_a_dataset_of_a_different_name(self):
        self.add_dataset(self.dst, "my_own_name", self.frames, media_root=self.tmp / "m")
        root = self.package()
        d = self.dialog(root)
        row = self.row(d, "d1")
        self.assertEqual(row["use"].currentData(), "")                             # different name: not preselected
        choices = [row["use"].itemData(i) for i in range(row["use"].count())]
        self.assertEqual(choices[:2], ["", "my_own_name"])                         # best fit is listed first
        row["use"].setCurrentIndex(row["use"].findData("my_own_name"))
        self.assertEqual(d.selection.items["d1"].target_dataset, "my_own_name")
        self.assertIn("Pose: 3/3 frames match", row["notes"].text())
        d._tick_all(False)
        row["cells"]["pose"].setCheckState(CHECKED)
        self.assertTrue(d.import_btn.isEnabled())
        comp.import_compilation(self.dst, d.contents, d.selection)
        self.rescan()
        self.assertTrue(self.dst.pose_labels.has_pose_labels("my_own_name"))
        self.assertNotIn("d1", self.dst.datasets.keys())

    def test_projects_you_have_default_to_use_existing_and_can_be_switched(self):
        self.add_project(self.dst, "proj")
        d = self.dialog(self.package())
        pr, pr2 = d._project_rows["proj"], d._project_rows["proj2"]
        self.assertEqual(pr["mode"].currentData(), "existing:proj")
        self.assertFalse(pr["name"].isEnabled())
        self.assertIn("Nothing is added", pr["note"].text())
        self.assertEqual(pr2["mode"].currentData(), "new")
        pr["mode"].setCurrentIndex(0)                                              # switch to "add as new"
        self.assertTrue(pr["name"].isEnabled())
        pr["name"].setText("proj")                                                 # ...but that name is taken
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("already have a project called 'proj'", d.problem_label.text())
        pr["name"].setText("proj_copy")
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.selection.project_choices["proj"], ("new", "proj_copy"))

    def test_a_project_can_be_mapped_onto_one_of_mine_and_gaps_are_flagged(self):
        self.add_project(self.dst, "mine", body_parts=["nose"], identities=["mouse1"], behaviors=["groom"])
        root = self.package()
        d = self.dialog(root)
        pr = d._project_rows["proj"]
        pr["mode"].setCurrentIndex(pr["mode"].findData("existing:mine"))
        self.assertEqual(d.selection.project_choices["proj"], ("existing", "mine"))
        self.assertIn("doesn't define", pr["note"].text())
        self.assertIn("tail_base", pr["note"].text())
        self.assertTrue(d.import_btn.isEnabled())                                  # a heads-up, not a blocker
        comp.import_compilation(self.dst, d.contents, d.selection)
        self.dst.scan_projects()
        self.assertEqual(sorted(self.dst.projects.keys()), ["mine", "proj2"])

    def test_unticking_the_projects_gate_skips_them_all(self):
        d = self.dialog(self.package())
        d.projects_check.setChecked(False)
        self.assertFalse(d.selection.import_projects)
        self.assertFalse(d.projects_box.isEnabled())
        comp.import_compilation(self.dst, d.contents, d.selection)
        self.dst.scan_projects()
        self.assertEqual(list(self.dst.projects.keys()), [])

    def test_locating_media_is_only_offered_while_adding_as_new(self):
        items = [CompilationItem("d1", "proj", include_media=False)]
        root = self.package(items, name="refonly")
        shutil.rmtree(self.tmp / "media" / "d1")
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "m")
        d = self.dialog(root)
        row = self.row(d, "d1")
        self.assertEqual(row["use"].currentData(), "d1")
        with mock.patch.object(icd.QFileDialog, "getExistingDirectory", return_value=str(self.tmp)) as dialog:
            d._on_double_click(d._rows.index(row), icd._NOTES)
        dialog.assert_not_called()                                                 # using mine: no media are needed

    def test_a_broken_entry_shows_why_and_cannot_be_ticked(self):
        root = self.package()
        shutil.rmtree(root / "vids")
        d = self.dialog(root)
        row = self.row(d, "vids")
        self.assertIn("missing", row["notes"].text())
        self.assertFalse(bool(row["cells"]["media"].flags() & Qt.ItemFlag.ItemIsUserCheckable))
        self.assertTrue(d.import_btn.isEnabled())                       # the rest still import

    def test_untick_everything_leaves_only_the_projects_then_nothing(self):
        d = self.dialog(self.package())
        d._tick_all(False)
        self.assertTrue(d.import_btn.isEnabled())                       # projects alone is still something
        d.projects_check.setChecked(False)
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("Tick at least one", d.problem_label.text())

    def test_tick_everything_never_means_replacing_a_dataset_you_have(self):
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "m")
        d = self.dialog(self.package())
        d._tick_all(False)
        d._tick_all(True)
        self.assertFalse(d._ticked(self.cell(d, "d1", "media")))
        self.assertTrue(d._ticked(self.cell(d, "vids", "media")))

    def test_a_dataset_whose_media_are_missing_can_be_located(self):
        items = [CompilationItem("d1", "proj", include_media=False)]
        root = self.package(items, name="refonly")
        shutil.rmtree(self.tmp / "media" / "d1")
        d = self.dialog(root)
        row = self.row(d, "d1")
        self.assertIn("double-click here to locate", row["notes"].text())
        self.assertFalse(d.import_btn.isEnabled())
        supplied = self.tmp / "supplied"
        for f in self.frames:
            (supplied / f).parent.mkdir(parents=True, exist_ok=True)
            (supplied / f).write_bytes(b"x")
        wrong = self.tmp
        with mock.patch.object(icd.QFileDialog, "getExistingDirectory", return_value=str(wrong)):
            d._on_double_click(d._rows.index(row), icd._NOTES)
        self.assertFalse(d.import_btn.isEnabled())
        self.assertIn("not in this folder", row["notes"].text())
        with mock.patch.object(icd.QFileDialog, "getExistingDirectory", return_value=str(supplied)):
            d._on_double_click(d._rows.index(row), icd._NOTES)
        self.assertTrue(d.import_btn.isEnabled())
        self.assertEqual(d.selection.items["d1"].media_dir, supplied)

    def test_a_damaged_compilation_explains_itself(self):
        root = self.package()
        (root / "compilation_info.json").write_text("{ nope")
        d = self.dialog(root)
        self.assertIn("damaged", d.header.text())
        self.assertFalse(d.import_btn.isEnabled())

    def test_what_the_dialog_selects_imports_correctly(self):
        root = self.package()
        d = self.dialog(root)
        self.cell(d, "vids", "behavior").setCheckState(UNCHECKED)
        self.row(d, "rats")["as_item"].setText("renamed rats")
        result = comp.import_compilation(self.dst, d.contents, d.selection)
        self.rescan()
        self.assertEqual(sorted(self.dst.datasets.keys()), ["d1", "renamed rats", "vids"])
        self.assertFalse(self.dst.behavior_labels.has_behavior_labels("vids"))
        self.assertTrue(self.dst.pose_labels.has_pose_labels("renamed rats"))
        self.assertTrue(all(x["error"] is None for x in result["datasets"]))


class ResultMessageTests(GuiCase):
    describe = staticmethod(ActionHandler._describe_compilation_import)

    def import_all(self, ws=None, sel=None):
        root = self.package()
        contents = comp.inspect_compilation(root)
        sel = sel or comp.default_compilation_selection(ws or self.dst, contents)
        return comp.import_compilation(ws or self.dst, contents, sel)

    def test_summarises_projects_and_each_dataset(self):
        text = self.describe(self.import_all())
        self.assertIn("3 of 3 dataset(s) imported.", text)
        self.assertIn("Projects added: proj, proj2.", text)
        self.assertIn("• d1", text)
        self.assertIn("Dataset 'd1': 5 of 5 frames copied.", text)
        self.assertIn("Pose labels added", text)

    def test_reports_failures_by_name(self):
        result = {"projects": [], "cancelled": False, "datasets": [
            {"name": "ok", "result": {"dataset_name": "ok", "notes": []}, "error": None},
            {"name": "bad one", "result": None, "error": "disk full"}]}
        text = self.describe(result)
        self.assertIn("1 of 2 dataset(s) imported.", text)
        self.assertIn("✗ bad one: disk full", text)

    def test_reports_cancellation(self):
        text = self.describe({"projects": [], "datasets": [], "cancelled": True})
        self.assertIn("Import cancelled", text)

    def test_projects_you_use_are_reported_as_such(self):
        self.add_project(self.dst, "proj")
        contents = comp.inspect_compilation(self.package())
        sel = comp.default_compilation_selection(self.dst, contents)              # proj exists -> use it; proj2 is new
        text = self.describe(comp.import_compilation(self.dst, contents, sel))
        self.assertIn("Projects added: proj2.", text)
        self.assertIn("Using your existing project(s): 'proj'.", text)

    def test_a_project_mapped_to_one_of_yours_or_renamed_is_described(self):
        self.add_project(self.dst, "mine")
        contents = comp.inspect_compilation(self.package())
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.project_choices["proj"] = ("existing", "mine")
        sel.project_choices["proj2"] = ("new", "rat_project")
        text = self.describe(comp.import_compilation(self.dst, contents, sel))
        self.assertIn("Using your existing project(s): 'mine' (for 'proj').", text)
        self.assertIn("Projects added: rat_project.", text)
