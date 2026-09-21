"""An export must contain everything the import tool needs, or refuse to be made.

Covers: media copied by default, checks before anything is written (missing files, free space),
copy verification, cleanup on failure, reading the finished package back, multi-arena and
frame-subset round trips, and the wording the user sees.
"""
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from fixtures import WorkspaceCase, frame_names, make_behavior_dataset
from whisker.core import bundle as fmt
from whisker.core import bundle_import as bi
from whisker.core import compilation as comp
from whisker.core.bundle_import import ImportSelection
from whisker.core.compilation import CompilationItem
from whisker.core.study.dataset import Dataset, DatasetType, MultiArenaConfig
from whisker.gui.dialogs import ExportAnnotationsDialog, ExportCompilationDialog
from whisker.gui.widgets.data_explorer.action_handler import ActionHandler

_app = QApplication.instance() or QApplication([])


class Case(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.src = self.new_workspace("src")
        self.add_project(self.src, "proj")
        self.files = frame_names(5)
        self.add_dataset(self.src, "d1", self.files)
        self.add_pose_labels(self.src, "d1", self.files[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.add_dataset(self.src, "d2", frame_names(3), media_root=self.tmp / "d2media")
        self.dst = self.new_workspace("dst")
        self.out = self.tmp / "exports"

    def plan(self, name="d1"):
        return fmt.build_export_plan(self.src, name, "proj")

    def export(self, name="d1", target="pkg", **kw):
        return fmt.export_annotation_bundle(self.plan(name), self.out / target, **kw)


# ---------------------------------------------------------- the default: everything copied


class DefaultsTests(Case):
    def test_by_default_every_file_is_copied_into_the_package_and_it_checks_out(self):
        r = self.export()
        self.assertEqual((r["media_included"], r["num_media"], r["num_media_copied"], r["num_missing"]), (True, 5, 5, 0))
        self.assertEqual(r["problems"], [])
        pkg = self.out / "pkg"
        self.assertEqual(sorted(p.name for p in (pkg / "frames").iterdir()), self.files)
        for needed in ("export_info.json", "README.txt", "project/proj.json", "dataset/manifest.json",
                       "pose_labels/labels.h5", "pose_labels/metadata.json", "behavior_labels/labels.h5"):
            self.assertTrue((pkg / needed).is_file(), needed)
        self.assertTrue(json.loads((pkg / "export_info.json").read_text())["dataset"]["media_included"])

    def test_a_default_export_imports_completely_into_an_empty_workspace(self):
        self.export()
        c = bi.inspect_bundle(self.out / "pkg")
        self.assertTrue(all(p.ok for p in (c.project, c.dataset, c.pose, c.behavior)))
        self.assertFalse(c.needs_media_folder)                     # nothing to go and find
        r = bi.import_from_bundle(self.dst, c, bi.default_selection(self.dst, c))
        self.dst.scan_projects(); self.dst.scan_datasets(); self.dst.scan_labels()
        self.assertEqual((r["num_media"], r["num_media_copied"], r["num_missing"]), (5, 5, 0))
        ds = self.dst.datasets.get("d1")
        self.assertTrue(all((Path(ds.base_data_path) / f).is_file() for f in self.files))

    def test_the_single_export_dialog_defaults_to_copying_for_both_kinds_of_dataset(self):
        self.add_dataset(self.src, "vids", ["a.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "vm")
        for name in ("d1", "vids"):
            d = ExportAnnotationsDialog(self.src, name, "proj")
            self.addCleanup(d.deleteLater)
            self.assertTrue(d.include_media, name)
            self.assertTrue(d.include_project and d.include_media, name)
            self.assertFalse(d.media_warning.isVisibleTo(d), name)
            self.assertIn("recommended", d.include_media_checkbox.text())

    def test_unticking_media_warns_that_the_recipient_must_find_the_files(self):
        d = ExportAnnotationsDialog(self.src, "d1", "proj")
        self.addCleanup(d.deleteLater)
        d.include_media_checkbox.setChecked(False)
        self.assertTrue(d.media_warning.isVisibleTo(d))
        self.assertIn("find them themselves", d.media_warning.text())
        d.include_media_checkbox.setChecked(True)
        self.assertFalse(d.media_warning.isVisibleTo(d))

    def test_the_compilation_dialog_defaults_to_copying_and_warns_when_it_wont(self):
        d = ExportCompilationDialog(self.src, "proj")
        self.addCleanup(d.deleteLater)
        d._tick_all(True)
        self.assertTrue(all(i.include_media for i in d.items()))
        self.assertFalse(d.media_note.isVisibleTo(d))
        next(r for r in d._rows if r["name"] == "d2")["media"].setCheckState(Qt.CheckState.Unchecked)
        self.assertTrue(d.media_note.isVisibleTo(d))
        self.assertIn("d2", d.media_note.text())
        self.assertIn("find the files themselves", d.media_note.text())


# ------------------------------------------------------------ checks before writing


class PreflightTests(Case):
    def test_a_missing_file_stops_the_export_before_anything_is_written(self):
        (self.tmp / "media" / "d1" / "frame_0002.png").unlink()
        with self.assertRaises(fmt.BundleError) as cm:
            self.export()
        msg = str(cm.exception)
        for expected in ("1 of 5 frames", "frame_0002.png", "complete package can't be made", "export without the frames"):
            self.assertIn(expected, msg)
        self.assertFalse((self.out / "pkg").exists())

    def test_a_failed_check_leaves_an_existing_package_untouched(self):
        self.export()
        (self.out / "pkg" / "marker.txt").write_text("the earlier package")
        (self.tmp / "media" / "d1" / "frame_0001.png").unlink()
        with self.assertRaises(fmt.BundleError):
            self.export(overwrite=True)
        self.assertEqual((self.out / "pkg" / "marker.txt").read_text(), "the earlier package")

    def test_many_missing_files_are_summarised(self):
        for f in self.files[:4]:
            (self.tmp / "media" / "d1" / f).unlink()
        with self.assertRaises(fmt.BundleError) as cm:
            self.export()
        self.assertIn("4 of 5 frames", str(cm.exception))
        self.assertIn("and 1 more", str(cm.exception))

    def test_missing_files_can_be_allowed_and_the_result_then_says_the_package_is_incomplete(self):
        (self.tmp / "media" / "d1" / "frame_0002.png").unlink()
        r = self.export(allow_missing=True)
        self.assertEqual((r["num_media_copied"], r["num_missing"]), (4, 1))
        self.assertTrue(any("can't be imported" in p for p in r["problems"]))

    def test_a_reference_only_export_needs_no_files_on_disk(self):
        shutil.rmtree(self.tmp / "media" / "d1")
        r = self.export(include_media=False)
        self.assertEqual((r["media_included"], r["problems"]), (False, []))

    def test_not_enough_free_space_is_refused_up_front(self):
        with mock.patch.object(fmt.shutil, "disk_usage", return_value=SimpleNamespace(total=100, used=99, free=1024)):
            with self.assertRaises(fmt.BundleError) as cm:
                self.export()
        self.assertIn("isn't enough free space", str(cm.exception))
        self.assertIn("free up some space", str(cm.exception).lower())
        self.assertFalse((self.out / "pkg").exists())

    def test_free_space_is_not_checked_when_no_media_is_copied(self):
        with mock.patch.object(fmt.shutil, "disk_usage", return_value=SimpleNamespace(total=100, used=99, free=1)):
            self.assertEqual(self.export(include_media=False)["problems"], [])

    def test_the_space_needed_scales_with_the_files(self):
        big = self.tmp / "media" / "d1" / "frame_0000.png"
        big.write_bytes(b"x" * (60 * 1024 * 1024))
        free = SimpleNamespace(total=1, used=1, free=70 * 1024 * 1024)               # room for slack, not for 60 MB + slack
        with mock.patch.object(fmt.shutil, "disk_usage", return_value=free):
            with self.assertRaises(fmt.BundleError) as cm:
                self.export()
        self.assertIn("about 8", str(cm.exception).replace("about 81", "about 8"))    # ~81 MB required

    def test_sizes_read_nicely(self):
        self.assertEqual([fmt.format_bytes(n) for n in (0, 512, 1536, 5 * 1024 ** 3)], ["0 B", "512 B", "1.5 KB", "5.0 GB"])


# ------------------------------------------------------ verifying while and after copying


class IntegrityTests(Case):
    def test_a_short_copy_is_caught_and_nothing_is_left_behind(self):
        real = shutil.copy2

        def truncating(src, dst, *a, **k):
            real(src, dst, *a, **k)
            Path(dst).write_bytes(Path(dst).read_bytes()[:4])            # a copy that silently lost data
        with mock.patch.object(fmt.shutil, "copy2", truncating):
            with self.assertRaises(fmt.BundleError) as cm:
                self.export()
        self.assertIn("Could not copy", str(cm.exception))
        self.assertIn("4 bytes", str(cm.exception))
        self.assertFalse((self.out / "pkg").exists())

    def test_a_failure_part_way_removes_the_partial_package(self):
        real = shutil.copy2
        calls = {"n": 0}

        def flaky(src, dst, *a, **k):
            calls["n"] += 1
            if "frame_" in str(src) and calls["n"] > 4:
                raise OSError("device is full")
            return real(src, dst, *a, **k)
        with mock.patch.object(fmt.shutil, "copy2", flaky):
            with self.assertRaises(fmt.BundleError) as cm:
                self.export()
        self.assertIn("device is full", str(cm.exception))
        self.assertFalse((self.out / "pkg").exists())

    def test_cancelling_removes_the_partial_package(self):
        state = {"n": 0}

        def cancel():
            state["n"] += 1
            return state["n"] > 9
        with self.assertRaises(fmt.BundleError):
            self.export(cancel_cb=cancel)
        self.assertFalse((self.out / "pkg").exists())

    def test_verify_export_passes_a_good_package_and_reads_it_like_the_importer(self):
        self.export()
        self.assertEqual(bi.verify_export(self.out / "pkg"), [])

    def test_verify_export_finds_what_is_wrong(self):
        self.export()
        pkg = self.out / "pkg"
        (pkg / "frames" / "frame_0001.png").unlink()
        (pkg / "pose_labels" / "labels.h5").unlink()
        problems = bi.verify_export(pkg)
        self.assertTrue(any("dataset can't be imported" in p and "1 of 5" in p for p in problems), problems)
        self.assertTrue(any("pose labels" in p for p in problems), problems)
        (pkg / "export_info.json").write_text("{ nope")
        self.assertTrue(any("damaged" in p for p in bi.verify_export(pkg)))


# ------------------------------------------------------------- compilations


class CompilationCompletenessTests(Case):
    def items(self):
        return [CompilationItem("d1", "proj"), CompilationItem("d2", "proj")]

    def build(self, **kw):
        items = self.items()
        return comp.export_compilation(items, comp.build_plans(self.src, items), self.out, "pack", **kw)

    def test_a_good_compilation_reports_no_problems(self):
        r = self.build()
        self.assertEqual(r["problems"], [])
        self.assertEqual((r["num_media"], r["num_media_copied"]), (8, 8))
        for name in ("d1", "d2"):
            self.assertEqual(bi.verify_export(self.out / "pack" / name), [])

    def test_a_missing_file_in_any_dataset_stops_the_whole_thing_and_names_it(self):
        (self.tmp / "d2media" / "frame_0001.png").unlink()
        with self.assertRaises(fmt.BundleError) as cm:
            self.build()
        self.assertIn("'d2':", str(cm.exception))
        self.assertNotIn("'d1':", str(cm.exception))
        self.assertFalse((self.out / "pack").exists())                 # not even a half-made package

    def test_a_failed_check_does_not_replace_an_existing_compilation(self):
        self.build()
        (self.out / "pack" / "marker.txt").write_text("earlier")
        (self.tmp / "d2media" / "frame_0001.png").unlink()
        with self.assertRaises(fmt.BundleError):
            self.build(overwrite=True)
        self.assertEqual((self.out / "pack" / "marker.txt").read_text(), "earlier")

    def test_space_is_checked_for_the_whole_compilation(self):
        with mock.patch.object(fmt.shutil, "disk_usage", return_value=SimpleNamespace(total=1, used=1, free=1024)):
            with self.assertRaises(fmt.BundleError) as cm:
                self.build()
        self.assertIn("isn't enough free space", str(cm.exception))
        self.assertFalse((self.out / "pack").exists())

    def test_problems_found_when_reading_back_are_reported_per_dataset(self):
        real = bi.verify_export

        def verify(root):
            return ["the labels can't be read"] if Path(root).name == "d2" else real(root)
        with mock.patch.object(bi, "verify_export", verify):
            r = self.build()
        self.assertEqual(r["problems"], ["'d2': the labels can't be read"])


# ------------------------------------------------- special datasets round-trip completely


class SpecialDatasetTests(Case):
    def test_a_multi_arena_dataset_survives_export_and_import(self):
        cfg = MultiArenaConfig(box_width=10, box_height=12, placements={"clip.mp4": [(0, 0), (20, 5)]})
        root = self.tmp / "arenas"
        root.mkdir()
        (root / "clip.mp4").write_bytes(b"video")
        self.src.add_dataset("arenas", Dataset(name="arenas", type=DatasetType.VIDEO_COLLECTION,
                                               base_data_path=str(root), files=["clip.mp4"], multi_arena=cfg))
        make_behavior_dataset(["clip_arena0", "clip_arena1"]).to_file(self.src.behavior_labels.base_dir / "arenas" / "labels.h5")
        self.src.scan_labels()
        r = fmt.export_annotation_bundle(self.plan("arenas"), self.out / "arenas_pkg")
        self.assertEqual(r["problems"], [])
        self.assertTrue(json.loads((self.out / "arenas_pkg" / "export_info.json").read_text())["dataset"]["multi_arena"])
        c = bi.inspect_bundle(self.out / "arenas_pkg")
        sel = bi.default_selection(self.dst, c)
        bi.import_from_bundle(self.dst, c, sel)
        self.dst.scan_datasets(); self.dst.scan_labels()
        got = self.dst.datasets.get("arenas")
        self.assertTrue(got.is_multi_arena)
        self.assertEqual((got.multi_arena.box_width, got.multi_arena.box_height), (10, 12))
        self.assertEqual({k: [tuple(p) for p in v] for k, v in got.multi_arena.placements.items()},
                         {"clip.mp4": [(0, 0), (20, 5)]})
        report = bi.analyze_behavior_labels(self.dst, "arenas", self.dst.behavior_labels.base_dir / "arenas" / "labels.h5")
        self.assertEqual((sorted(report.matched), report.unmatched), (["clip_arena0", "clip_arena1"], []))

    def test_a_frame_subset_with_nested_folders_and_arena_boxes_survives(self):
        files = ["clip_arena0/frame_0001.png", "clip_arena0/frame_0002.png", "clip_arena1/frame_0001.png"]
        boxes = {"clip_arena0": (0, 0, 10, 10), "clip_arena1": (20, 0, 10, 10)}
        root = self.tmp / "subset"
        for f in files:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_bytes(b"png")
        self.src.add_dataset("subset", Dataset(name="subset", type=DatasetType.FRAME_SUBSET,
                                               base_data_path=str(root), files=files, arena_boxes=boxes))
        self.add_pose_labels(self.src, "subset", files[:2])
        r = fmt.export_annotation_bundle(self.plan("subset"), self.out / "subset_pkg")
        self.assertEqual(r["problems"], [])
        for f in files:
            self.assertTrue((self.out / "subset_pkg" / "frames" / f).is_file(), f)
        c = bi.inspect_bundle(self.out / "subset_pkg")
        bi.import_from_bundle(self.dst, c, bi.default_selection(self.dst, c))
        self.dst.scan_datasets(); self.dst.scan_labels()
        got = self.dst.datasets.get("subset")
        self.assertEqual(sorted(got.files), sorted(files))
        self.assertEqual({k: tuple(v) for k, v in got.arena_boxes.items()}, boxes)
        self.assertEqual(len(bi.analyze_pose_labels(self.dst, "subset", c.pose.path).matched), 2)


# ------------------------------------------------------------ what the user is told


class MessageTests(Case):
    def test_a_complete_export_says_it_was_checked(self):
        text, complete = ActionHandler._describe_export(self.export(), "d1")
        self.assertTrue(complete)
        self.assertIn("Frames copied: 5/5", text)
        self.assertIn("Checked: everything the import tool needs is in the package", text)

    def test_a_reference_only_export_says_the_recipient_will_be_asked(self):
        text, complete = ActionHandler._describe_export(self.export(include_media=False), "d1")
        self.assertTrue(complete)
        self.assertIn("referenced, not copied", text)
        self.assertIn("asked where the frames are", text)

    def test_problems_are_listed_and_the_export_is_not_called_complete(self):
        result = {"media_kind": "frames", "bundle_dir": "x", "media_included": True, "num_media": 5,
                  "num_media_copied": 4, "problems": ["The dataset can't be imported from this package: 1 of 5 missing"]}
        text, complete = ActionHandler._describe_export(result, "d1")
        self.assertFalse(complete)
        self.assertIn("would NOT import completely", text)
        self.assertIn("• The dataset can't be imported", text)
        self.assertNotIn("Checked:", text)

    def test_compilation_messages(self):
        items = [CompilationItem("d1", "proj", include_media=False), CompilationItem("d2", "proj")]
        r = comp.export_compilation(items, comp.build_plans(self.src, items), self.out, "pack")
        text, complete = ActionHandler._describe_compilation_export(r)
        self.assertTrue(complete)
        self.assertIn("Exported 2 dataset(s)", text)
        self.assertIn("Not copied for: d1", text)
        self.assertIn("Checked: everything the import tool needs is in every dataset's folder", text)
        r["problems"] = ["'d2': the labels can't be read"]
        text, complete = ActionHandler._describe_compilation_export(r)
        self.assertFalse(complete)
        self.assertIn("• 'd2': the labels can't be read", text)
