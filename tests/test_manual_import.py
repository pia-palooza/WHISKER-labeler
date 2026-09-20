"""Characterization tests for the per-piece import (whisker.core.manual_import)."""
import json
from pathlib import Path

from fixtures import WorkspaceCase, frame_names
from whisker.core import manual_import as mi
from whisker.core.study.dataset import DatasetType


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

    def _import(self, name="d1", **kw):
        b, project, dataset = self._pieces()
        return mi.import_dataset_from_components(
            self.dst, name, project, b / "project" / "proj.json", dataset, b / "frames",
            pose_labels_path=b / "pose_labels" / "labels.h5",
            pose_metadata_path=b / "pose_labels" / "metadata.json",
            behavior_labels_path=b / "behavior_labels" / "labels.h5", **kw)

    def _rescan(self):
        self.dst.scan_projects(); self.dst.scan_datasets(); self.dst.scan_labels()

    def test_round_trip_installs_everything(self):
        result = self._import()
        self._rescan()
        self.assertEqual((result["num_media"], result["num_media_copied"], result["num_missing"]), (4, 4, 0))
        self.assertTrue(result["pose_imported"] and result["behavior_imported"] and result["project_installed"])
        self.assertIsNotNone(self.dst.projects.get("proj"))
        ds = self.dst.datasets.get("d1")
        self.assertEqual(sorted(ds.files), sorted(self.files))
        # media were copied *into* the workspace, and the manifest points at the copy
        self.assertTrue(str(self.dst.base_dir.resolve()) in str(Path(ds.base_data_path).resolve()))
        self.assertTrue(all((Path(ds.base_data_path) / f).exists() for f in self.files))
        self.assertTrue(self.dst.pose_labels.has_pose_labels("d1"))
        self.assertTrue(self.dst.behavior_labels.has_behavior_labels("d1"))

    def test_existing_dataset_is_refused_without_overwrite(self):
        self._import()
        with self.assertRaises(FileExistsError):
            self._import()

    def test_overwrite_replaces(self):
        self._import()
        self.assertEqual(self._import(overwrite=True)["num_media_copied"], 4)

    def test_imported_under_a_new_name_records_that_name_in_pose_metadata(self):
        self._import(name="renamed")
        meta = json.loads((self.dst.pose_labels.base_dir / "renamed" / "metadata.json").read_text())
        self.assertEqual(meta["dataset_name"], "renamed")

    def test_component_checks_explain_wrong_picks(self):
        b = self.bundle
        self.assertIn("dataset info file", mi.check_project_file(b / "dataset" / "manifest.json")[0].message)
        self.assertIn("project file", mi.check_dataset_file(b / "project" / "proj.json")[0].message)
        _, _, dataset = self._pieces()
        check, missing = mi.check_media_folder(b / "project", dataset)
        self.assertFalse(check.ok)
        self.assertEqual(len(missing), 4)
        self.assertTrue(mi.check_media_folder(b / "frames", dataset)[0].ok)
