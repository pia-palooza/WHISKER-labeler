"""Tests for the piece-by-piece checks and writers in whisker.core.manual_import.

These back every import (bundle_import composes them); the end-to-end behavior is covered in
test_bundle_import.py, test_separate_files.py and test_compilation.py.
"""
import json
from pathlib import Path

from fixtures import WorkspaceCase, frame_names
from whisker.core import manual_import as mi


class ManualImportTests(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.src = self.new_workspace("src")
        self.add_project(self.src)
        self.files = frame_names(4)
        self.add_dataset(self.src, "d1", self.files)
        self.add_pose_labels(self.src, "d1", self.files[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.bundle = self.export_bundle(self.src, "d1")
        self.dst = self.new_workspace("dst")

    def _pieces(self):
        b = self.bundle
        _, project = mi.check_project_file(b / "project" / "proj.json")
        _, dataset = mi.check_dataset_file(b / "dataset" / "manifest.json")
        return b, project, dataset

    def _rescan(self):
        self.dst.scan_projects(); self.dst.scan_datasets(); self.dst.scan_labels()

    # ------------------------------------------------------------------ install_dataset

    def test_install_dataset_copies_the_media_and_points_the_manifest_at_the_copy(self):
        b, _project, dataset = self._pieces()
        copied, missing = mi.install_dataset(self.dst, dataset, "d1", b / "frames")
        self._rescan()
        self.assertEqual((copied, missing), (4, []))
        ds = self.dst.datasets.get("d1")
        self.assertEqual(sorted(ds.files), sorted(self.files))
        self.assertIn(str(self.dst.base_dir.resolve()), str(Path(ds.base_data_path).resolve()))
        self.assertTrue(all((Path(ds.base_data_path) / f).exists() for f in self.files))

    def test_install_dataset_refuses_an_existing_dataset_unless_told_to_replace_it(self):
        b, _project, dataset = self._pieces()
        mi.install_dataset(self.dst, dataset, "d1", b / "frames")
        with self.assertRaises(FileExistsError):
            mi.install_dataset(self.dst, dataset, "d1", b / "frames")
        copied, _missing = mi.install_dataset(self.dst, dataset, "d1", b / "frames", overwrite=True)
        self.assertEqual(copied, 4)

    def test_install_dataset_reports_files_it_could_not_find(self):
        b, _project, dataset = self._pieces()
        (b / "frames" / "frame_0001.png").unlink()
        copied, missing = mi.install_dataset(self.dst, dataset, "d1", b / "frames")
        self.assertEqual((copied, missing), (3, ["frame_0001.png"]))

    # ------------------------------------------------------------------ labels and projects

    def test_a_dataset_installed_under_a_new_name_records_that_name_in_its_pose_metadata(self):
        b = self.bundle
        self.assertTrue(mi.install_pose_labels(self.dst, "renamed", b / "pose_labels" / "labels.h5"))
        meta = json.loads((self.dst.pose_labels.base_dir / "renamed" / "metadata.json").read_text())
        self.assertEqual(meta["dataset_name"], "renamed")
        self.assertEqual(sorted(meta["frame_indices"]), self.files[:3])

    def test_labels_are_not_replaced_unless_told_to(self):
        b = self.bundle
        h5 = b / "pose_labels" / "labels.h5"
        self.assertTrue(mi.install_pose_labels(self.dst, "d1", h5))
        self.assertFalse(mi.install_pose_labels(self.dst, "d1", h5))                    # already there: left alone
        self.assertTrue(mi.install_pose_labels(self.dst, "d1", h5, overwrite=True))
        beh = b / "behavior_labels" / "labels.h5"
        self.assertTrue(mi.install_behavior_labels(self.dst, "d1", beh))
        self.assertFalse(mi.install_behavior_labels(self.dst, "d1", beh))
        self.assertTrue(mi.install_behavior_labels(self.dst, "d1", beh, overwrite=True))

    def test_install_project_keeps_an_existing_one_and_can_save_under_another_name(self):
        b, project, _dataset = self._pieces()
        source = b / "project" / "proj.json"
        self.assertTrue(mi.install_project(self.dst, project, source))
        self.assertFalse(mi.install_project(self.dst, project, source))                # exists: untouched
        self.assertTrue(mi.install_project(self.dst, project, source, overwrite=True))
        self.assertTrue(mi.install_project(self.dst, project, source, name="copy"))
        self._rescan()
        self.assertEqual(sorted(self.dst.projects.keys()), ["copy", "proj"])
        self.assertEqual(self.dst.projects.get("copy").name, "copy")                   # the name inside changed too

    # ------------------------------------------------------------------ the checks

    def test_component_checks_explain_wrong_picks(self):
        b = self.bundle
        self.assertIn("dataset info file", mi.check_project_file(b / "dataset" / "manifest.json")[0].message)
        self.assertIn("project file", mi.check_dataset_file(b / "project" / "proj.json")[0].message)
        _, _, dataset = self._pieces()
        check, missing = mi.check_media_folder(b / "project", dataset)
        self.assertFalse(check.ok)
        self.assertEqual(len(missing), 4)
        self.assertTrue(mi.check_media_folder(b / "frames", dataset)[0].ok)
