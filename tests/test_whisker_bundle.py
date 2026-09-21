"""Whisker bundles: reading, importing and exporting the ZIP format shared with WHISKER.

Covers: the manifest rules (format, version, unknown fields), refusing hostile archives, reading a real bundle
made by WHISKER, importing without ever leaving a half-done workspace (corrupt, truncated, cancelled, failing
part-way), where media go for each dataset type, replacing what exists, exporting a bundle WHISKER can read, and
that a bundle survives the round trip byte for byte.
"""
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from bundle_helpers import BundleCase, ds_entry, ds_manifest, manifest_for, sha, tree, write_zip
from fixtures import WorkspaceCase, frame_names
from whisker.core import whisker_bundle as wb
from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.workspace import Workspace
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset

# A real bundle made by WHISKER. The tests that need it are skipped where it isn't available.
SAMPLE = Path(r"C:\Users\piaa\Downloads\whisker_bundle_20260803_Pharm_JI_trainingdataset _Manual Samples__20260921.zip")
SAMPLE_DATASET = "20260803_Pharm_JI_trainingdataset [Manual Samples]"
# Real bundles from WHISKER with videos: a small multi-arena one, and a 13.7 GB one (opt in with WHISKER_TEST_BIG_BUNDLE=1).
ARENA_SAMPLE = Path(r"C:\Users\piaa\Downloads\whisker_bundle_TST_Tshz1_TrainClips3_20260921.zip")
ARENA_DATASET = "TST_Tshz1_TrainClips3"
BIG_SAMPLE = Path(r"C:\Users\piaa\Downloads\whisker_bundle_PJA_JI_MDMA_JI_TrimmedTrainingvideos_20260921.zip")
BIG_DATASET = "PJA_JI_MDMA_JI_TrimmedTrainingvideos"


# --------------------------------------------------------------------------- #
# The manifest and untrusted names
# --------------------------------------------------------------------------- #


class ManifestRules(BundleCase):
    def test_reads_a_valid_bundle(self):
        b = self.open(self.make_bundle(project="proj"))
        self.assertEqual(b.manifest.format_version, 1)
        self.assertEqual([d.name for d in b.manifest.datasets], ["ds"])
        self.assertEqual(b.manifest.datasets[0].type, DatasetType.FRAME_SUBSET)
        self.assertEqual(b.manifest.projects, ["proj"])

    def test_unknown_manifest_fields_are_ignored(self):
        p = self.make_bundle(extra_manifest={"from_the_future": {"x": 1}, "datasets": [
            {**ds_entry("ds", files=["a/f1.png", "a/f2.png"]), "new_field": [1, 2]}]})
        self.assertEqual(self.open(p).manifest.datasets[0].name, "ds")

    def test_a_newer_format_version_is_refused(self):
        with self.assertRaisesRegex(wb.BundleError, "newer"):
            wb.WhiskerBundle.open(self.make_bundle(extra_manifest={"format_version": 2}))

    def test_a_zip_that_is_not_a_whisker_bundle_is_refused(self):
        p = self.tmp / "other.zip"
        write_zip(p, {"readme.txt": b"hi"})
        with self.assertRaisesRegex(wb.BundleError, "no whisker_bundle.json"):
            wb.WhiskerBundle.open(p)
        write_zip(p, {}, manifest={"format": "something-else", "format_version": 1})
        with self.assertRaisesRegex(wb.BundleError, "isn't a Whisker bundle"):
            wb.WhiskerBundle.open(p)

    def test_a_file_that_is_not_a_zip_is_refused_clearly(self):
        p = self.tmp / "junk.zip"
        p.write_bytes(b"this is not a zip file")
        with self.assertRaisesRegex(wb.BundleError, "isn't a readable zip"):
            wb.WhiskerBundle.open(p)
        with self.assertRaisesRegex(wb.BundleError, "doesn't exist"):
            wb.WhiskerBundle.open(self.tmp / "nope.zip")


class HostileArchives(BundleCase):
    BAD_NAMES = ["../evil.txt", "datasets/../../evil.txt", "/abs/evil.txt", "C:/evil.txt", "datasets\\..\\evil.txt",
                 "datasets/x/media/../../../evil", "datasets//x", "datasets/./x", "a/b:stream", "bad\x01name"]

    def test_unsafe_entry_names_refuse_the_whole_bundle(self):
        for bad in self.BAD_NAMES:
            with self.subTest(bad):
                p = self.tmp / "evil.zip"
                write_zip(p, {bad: b"x"}, manifest_for())
                with self.assertRaises(wb.BundleError):
                    wb.WhiskerBundle.open(p)

    def test_unsafe_names_in_the_manifest_are_refused(self):
        for bad in ("../x", "a/b", "C:x", " lead"):
            with self.subTest(bad):
                p = self.tmp / "evil.zip"
                write_zip(p, {}, manifest_for(ds_entry(bad)))
                with self.assertRaises(wb.BundleError):
                    wb.WhiskerBundle.open(p)

    def test_unsafe_file_names_in_a_datasets_file_list_are_refused(self):
        p = self.tmp / "evil.zip"
        write_zip(p, {"datasets/ds/manifest.json": ds_manifest("ds", "FRAME_SUBSET", ["../../escape.png"])},
                  manifest_for(ds_entry("ds", files=["../../escape.png"])))
        b = self.open(p)
        with self.assertRaises(wb.BundleError):
            b.media_files("ds", b.dataset("ds"))

    def test_links_inside_the_archive_are_refused(self):
        p = self.tmp / "link.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(wb.MANIFEST_NAME, json.dumps(manifest_for()))
            info = zipfile.ZipInfo("datasets/x")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(wb.BundleError, "link"):
            wb.WhiskerBundle.open(p)


# --------------------------------------------------------------------------- #
# A real bundle made by WHISKER
# --------------------------------------------------------------------------- #


@unittest.skipUnless(SAMPLE.exists(), "the example bundle from WHISKER isn't available")
class RealBundle(unittest.TestCase):
    """Imports the example bundle once, into a throwaway workspace, and checks it from several angles."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        (cls.tmp / "ws").mkdir()
        cls.ws = Workspace(cls.tmp / "ws")
        cls.bundle = wb.WhiskerBundle.open(SAMPLE)
        cls.items = wb.plan_import(cls.ws, cls.bundle)
        cls.chosen = wb.default_choices(cls.items)
        cls.result = wb.import_bundle(cls.ws, cls.bundle, cls.chosen)
        # What the app does once an import has finished.
        cls.ws.scan_datasets(); cls.ws.scan_projects(); cls.ws.scan_labels(); cls.ws.scan_predictions()

    @classmethod
    def tearDownClass(cls):
        cls.bundle.close()
        for h in [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and str(cls.tmp) in h.baseFilename]:
            logging.getLogger().removeHandler(h)
            h.close()
        logging.disable(logging.NOTSET)
        cls._tmp.cleanup()

    def test_the_manifest_is_read_without_extracting_anything(self):
        m = self.bundle.manifest
        self.assertEqual((m.format_version, m.whisker_version), (1, "0.8.0"))
        d = m.datasets[0]
        self.assertEqual((d.name, d.type, d.file_count, d.multi_arena), (SAMPLE_DATASET, DatasetType.FRAME_SUBSET, 544, False))
        self.assertEqual(d.label_workflows, ["pose_estimation"])
        self.assertEqual(m.projects, ["Boris_MDMA"])
        self.assertEqual([(r.workflow, r.run_name) for r in m.prediction_runs], [("pose_estimation", "PharmJI_Wpose_852F_20260830_v5")])

    def test_every_listed_frame_is_in_the_archive(self):
        present, absent = self.bundle.media_files(SAMPLE_DATASET, self.bundle.dataset(SAMPLE_DATASET))
        self.assertEqual((len(present), absent), (544, []))

    def test_the_offer_lists_the_dataset_its_labels_the_run_and_the_project(self):
        self.assertEqual([(i.kind, i.exists) for i in self.items], [("dataset", False), ("labels", False), ("run", False), ("project", False)])
        self.assertEqual(self.chosen, {i.key for i in self.items})

    def test_the_dataset_lands_in_the_workspace_with_its_frames(self):
        ds_dir = self.ws.base_dir / "datasets" / SAMPLE_DATASET
        self.assertEqual(len(list((ds_dir / "data").rglob("*.png"))), 544)
        self.assertEqual(self.result["datasets"], [SAMPLE_DATASET])

    def test_the_manifests_media_location_is_rewritten_for_this_machine(self):
        ds = Dataset.from_json((self.ws.base_dir / "datasets" / SAMPLE_DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(ds.base_data_path), (self.ws.base_dir / "datasets" / SAMPLE_DATASET / "data").resolve())
        self.assertNotIn("D:\\workspace", ds.base_data_path)
        self.assertEqual(len(ds.files), 544)
        self.assertTrue(all((Path(ds.base_data_path) / f).is_file() for f in ds.files))

    def test_labels_predictions_and_project_are_copied_unchanged(self):
        with zipfile.ZipFile(SAMPLE) as z:
            for entry in (f"workflows/pose_estimation/labels/{SAMPLE_DATASET}/labels.h5",
                          f"workflows/pose_estimation/labels/{SAMPLE_DATASET}/metadata.json",
                          f"workflows/pose_estimation/predictions/PharmJI_Wpose_852F_20260830_v5/{SAMPLE_DATASET}/predictions.h5",
                          "projects/Boris_MDMA.json"):
                self.assertEqual(hashlib.sha256(z.read(entry)).hexdigest(), sha(self.ws.base_dir / entry), entry)

    def test_the_labeler_can_read_the_imported_pose_labels_and_the_workspace_sees_everything(self):
        self.assertIsNotNone(self.ws.datasets.get(SAMPLE_DATASET))
        self.assertIsNotNone(self.ws.projects.get("Boris_MDMA"))
        self.assertTrue(self.ws.pose_labels.has_pose_labels(SAMPLE_DATASET))
        pose = PoseDataset.from_file(self.ws.pose_labels.base_dir / SAMPLE_DATASET / "labels.h5")
        with zipfile.ZipFile(SAMPLE) as z:
            listed = json.loads(z.read(f"workflows/pose_estimation/labels/{SAMPLE_DATASET}/metadata.json"))["frame_indices"]
        self.assertEqual(set(pose.frame_indices), set(listed))      # not every frame of the dataset is labeled, and none is lost
        self.assertEqual(pose.individuals, ["SmallerMouse", "LargerMouse"])

    def test_nothing_is_left_behind_and_a_second_look_finds_it_all_present(self):
        self.assertEqual(tree(self.ws.base_dir)["<staging>"], [])
        again = wb.plan_import(self.ws, self.bundle)
        self.assertTrue(all(i.exists for i in again))
        self.assertEqual(wb.default_choices(again), set())          # nothing is offered again unless the user ticks it

    def test_exporting_it_again_gives_the_same_files_and_the_same_bytes(self):
        """The other direction: what the labeler writes is what WHISKER wrote."""
        runs = wb.list_prediction_runs(self.ws, [SAMPLE_DATASET])
        plan = wb.build_export_plan(self.ws, [SAMPLE_DATASET], ["Boris_MDMA"], [(r.workflow, r.run_name) for r in runs])
        out = self.tmp / "again.zip"
        result = wb.export_bundle(self.ws, plan, out)
        self.assertEqual(result["problems"], [])
        with zipfile.ZipFile(SAMPLE) as a, zipfile.ZipFile(out) as b:
            ai = {i.filename: i for i in a.infolist()}
            bi = {i.filename: i for i in b.infolist()}
            self.assertEqual(set(ai), set(bi))
            # Every file is the same size, checksum and storage method, except the two descriptions that are
            # rewritten on the way through (the bundle's own, and the dataset's, whose media location changes).
            rewritten = {wb.MANIFEST_NAME, f"datasets/{SAMPLE_DATASET}/manifest.json"}
            self.assertEqual({n: (i.file_size, i.CRC, i.compress_type) for n, i in ai.items() if n not in rewritten},
                             {n: (i.file_size, i.CRC, i.compress_type) for n, i in bi.items() if n not in rewritten})
            ma, mb = json.loads(a.read(wb.MANIFEST_NAME)), json.loads(b.read(wb.MANIFEST_NAME))
            dsa = json.loads(a.read(f"datasets/{SAMPLE_DATASET}/manifest.json"))
            dsb = json.loads(b.read(f"datasets/{SAMPLE_DATASET}/manifest.json"))
        self.assertEqual({k: v for k, v in dsa.items() if k != "base_data_path"}, {k: v for k, v in dsb.items() if k != "base_data_path"})
        for key in ("format", "format_version", "projects", "prediction_runs", "file_count"):
            self.assertEqual(ma[key], mb[key], key)
        da, db = ma["datasets"][0], mb["datasets"][0]
        for key in set(da) - {"original_base_data_path"}:
            self.assertEqual(da[key], db[key], key)


class RealVideoBundleCase(unittest.TestCase):
    """Imports a real WHISKER bundle of videos once, into throwaway folders, and lets subclasses examine it."""

    BUNDLE: Path
    DATASET: str

    @classmethod
    def setUpClass(cls):
        if not getattr(cls, "BUNDLE", None):
            raise unittest.SkipTest("base class")
        if not cls.BUNDLE.exists():
            raise unittest.SkipTest(f"{cls.BUNDLE.name} isn't available")
        logging.disable(logging.CRITICAL)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        (cls.tmp / "ws").mkdir()
        cls.media_root = cls.tmp / "videos"
        cls.media_root.mkdir()
        cls.ws = Workspace(cls.tmp / "ws")
        cls.bundle = wb.WhiskerBundle.open(cls.BUNDLE)
        cls.items = wb.plan_import(cls.ws, cls.bundle)
        cls.result = wb.import_bundle(cls.ws, cls.bundle, wb.default_choices(cls.items), cls.media_root)
        cls.ws.scan_datasets(); cls.ws.scan_projects(); cls.ws.scan_labels(); cls.ws.scan_predictions()
        with zipfile.ZipFile(cls.BUNDLE) as z:
            cls.sizes = {i.filename: i.file_size for i in z.infolist()}

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "_tmp"):
            return
        cls.bundle.close()
        for h in [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and str(cls.tmp) in h.baseFilename]:
            logging.getLogger().removeHandler(h)
            h.close()
        logging.disable(logging.NOTSET)
        cls._tmp.cleanup()

    def dataset(self) -> Dataset:
        return self.ws.datasets.get(self.DATASET)

    def test_the_manifest_describes_a_video_collection_stored_outside_the_workspace(self):
        d = self.bundle.manifest.dataset(self.DATASET)
        self.assertEqual((d.type, d.media_layout, d.keeps_media_in_workspace), (DatasetType.VIDEO_COLLECTION, "external", False))
        self.assertEqual(d.missing_media, [])

    def test_videos_need_a_folder_and_are_refused_without_one(self):
        item = next(i for i in self.items if i.kind == "dataset")
        self.assertTrue(item.requires_media_folder)
        with self.assertRaisesRegex(wb.BundleError, "Choose the folder"):
            wb.import_bundle(self.ws, self.bundle, {item.key}, None)

    def test_every_video_arrives_at_its_full_size_in_the_chosen_folder(self):
        folder = self.media_root / self.DATASET
        files = self.dataset().files
        self.assertTrue(files)
        for f in files:
            self.assertEqual((folder / f).stat().st_size, self.sizes[f"datasets/{self.DATASET}/media/{f}"], f)
        self.assertEqual(Path(self.dataset().base_data_path), folder.resolve())

    def test_the_workspace_reaches_every_video_through_its_data_folder(self):
        data = self.ws.base_dir / "datasets" / self.DATASET / "data"
        self.assertTrue(all((data / f).is_file() for f in self.dataset().files))

    def test_behavior_labels_and_the_project_are_copied_unchanged(self):
        with zipfile.ZipFile(self.BUNDLE) as z:
            entry = f"workflows/behavior_classification/labels/{self.DATASET}/labels.h5"
            self.assertEqual(hashlib.sha256(z.read(entry)).hexdigest(), sha(self.ws.base_dir / entry), entry)
            project = next(n for n in z.namelist() if n.startswith("projects/"))
            self.assertEqual(hashlib.sha256(z.read(project)).hexdigest(), sha(self.ws.base_dir / project))
        self.assertTrue(self.ws.behavior_labels.has_behavior_labels(self.DATASET))

    def test_nothing_is_left_behind(self):
        self.assertEqual(tree(self.ws.base_dir)["<staging>"], [])
        self.assertEqual([p.name for p in self.media_root.iterdir()], [self.DATASET])

    def reexport(self, name="again.zip"):
        runs = wb.list_prediction_runs(self.ws, [self.DATASET])
        plan = wb.build_export_plan(self.ws, [self.DATASET], self.bundle.manifest.projects, [(r.workflow, r.run_name) for r in runs])
        out = self.tmp / name
        result = wb.export_bundle(self.ws, plan, out)
        self.assertEqual((result["problems"], result["num_missing"]), ([], 0))
        return out

    def assert_round_trip(self, out: Path):
        """Same files, and every one the same size, checksum and storage method, except the two descriptions that are
        rewritten on the way through."""
        rewritten = {wb.MANIFEST_NAME, f"datasets/{self.DATASET}/manifest.json"}
        with zipfile.ZipFile(self.BUNDLE) as a, zipfile.ZipFile(out) as b:
            self.assertEqual(set(a.namelist()), set(b.namelist()))
            sig = lambda z, n: (z.getinfo(n).CRC, z.getinfo(n).file_size, z.getinfo(n).compress_type)
            self.assertEqual([n for n in a.namelist() if n not in rewritten and sig(a, n) != sig(b, n)], [])
            ma, mb = json.loads(a.read(wb.MANIFEST_NAME)), json.loads(b.read(wb.MANIFEST_NAME))
            dsa = json.loads(a.read(f"datasets/{self.DATASET}/manifest.json"))
            dsb = json.loads(b.read(f"datasets/{self.DATASET}/manifest.json"))
        self.assertEqual({k: v for k, v in dsa.items() if k != "base_data_path"}, {k: v for k, v in dsb.items() if k != "base_data_path"})
        for key in ("format", "format_version", "projects", "prediction_runs", "file_count"):
            self.assertEqual(ma[key], mb[key], key)
        da, db = ma["datasets"][0], mb["datasets"][0]
        self.assertEqual({k: v for k, v in da.items() if k != "original_base_data_path"},
                         {k: v for k, v in db.items() if k != "original_base_data_path"})


class RealMultiArenaVideoBundle(RealVideoBundleCase):
    """A real WHISKER bundle of videos with several arenas per video, and behavior labels keyed by arena."""

    BUNDLE, DATASET = ARENA_SAMPLE, ARENA_DATASET

    def test_the_arena_layout_survives_unchanged(self):
        ds = self.dataset()
        self.assertTrue(ds.is_multi_arena)
        raw = self.bundle.read_json(f"datasets/{self.DATASET}/manifest.json")
        on_disk = json.loads((self.ws.base_dir / "datasets" / self.DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["multi_arena"], raw["multi_arena"])        # box size and every placement
        self.assertEqual(on_disk["files"], raw["files"])
        self.assertEqual((ds.multi_arena.box_width, ds.multi_arena.box_height), (120, 230))
        self.assertEqual(len(ds.multi_arena.arena_units()), sum(len(v) for v in raw["multi_arena"]["placements"].values()))

    def test_behavior_labels_are_keyed_by_arena_and_each_key_is_a_real_arena_of_the_dataset(self):
        beh = BehaviorDataset.from_file(self.ws.base_dir / f"workflows/behavior_classification/labels/{self.DATASET}/labels.h5")
        keys = set(beh.bouts["video_key"])
        self.assertTrue(keys)
        self.assertTrue(all("_arena" in k for k in keys))
        self.assertTrue(all(self.dataset().resolve_arena_stem(k) is not None for k in keys), keys)

    def test_a_bouts_end_frame_is_never_before_its_start(self):
        bouts = BehaviorDataset.from_file(self.ws.base_dir / f"workflows/behavior_classification/labels/{self.DATASET}/labels.h5").bouts
        self.assertTrue((bouts["end_frame"] >= bouts["start_frame"]).all())

    def test_exporting_it_again_gives_the_bundle_back(self):
        self.assert_round_trip(self.reexport())


@unittest.skipUnless(os.environ.get("WHISKER_TEST_BIG_BUNDLE"), "set WHISKER_TEST_BIG_BUNDLE=1 to run the 13.7 GB bundle (a few minutes, ~30 GB of disk)")
class RealBigVideoBundle(RealVideoBundleCase):
    """A real 13.7 GB ZIP64 bundle: 21 videos, behavior labels, behavior predictions and a pose prediction folder per video."""

    BUNDLE, DATASET = BIG_SAMPLE, BIG_DATASET

    def test_predictions_arrive_as_they_were_in_the_bundle(self):
        predictions = [n for n in self.sizes if "/predictions/" in n]
        self.assertGreater(len(predictions), 40)
        for name in predictions:
            self.assertEqual((self.ws.base_dir / name).stat().st_size, self.sizes[name], name)
        run = self.ws.base_dir / "workflows/pose_estimation/predictions/PharmJI_Wpose_852F_20260830_v5" / self.DATASET
        first = sorted(d for d in run.iterdir() if d.is_dir())[0]
        self.assertTrue(PoseDataset.from_file(first / "predictions.h5").frame_indices)

    def test_exporting_it_again_gives_the_bundle_back(self):
        self.assert_round_trip(self.reexport())


# --------------------------------------------------------------------------- #
# Importing
# --------------------------------------------------------------------------- #


class Importing(BundleCase):
    def setUp(self):
        super().setUp()
        self.ws = self.new_workspace("ws")

    def do_import(self, path, chosen=None, media_root=None, **kw):
        b = self.open(path)
        items = wb.plan_import(self.ws, b)
        return wb.import_bundle(self.ws, b, wb.default_choices(items) if chosen is None else chosen, media_root, **kw)

    def test_a_frame_subset_goes_inside_the_workspace(self):
        r = self.do_import(self.make_bundle(project="proj"))
        base = self.ws.base_dir
        self.assertEqual(r["datasets"], ["ds"])
        self.assertEqual((base / "datasets/ds/data/a/f1.png").read_bytes(), b"media:a/f1.png" * 10)
        self.assertTrue((base / "workflows/pose_estimation/labels/ds/labels.h5").is_file())
        self.assertTrue((base / "projects/proj.json").is_file())
        ds = Dataset.from_json((base / "datasets/ds/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(ds.base_data_path), (base / "datasets/ds/data").resolve())

    def test_videos_go_to_the_chosen_folder_and_are_linked_from_the_workspace(self):
        media_root = self.tmp / "videos"
        media_root.mkdir()
        p = self.make_bundle(dataset="vids", type_="VIDEO_COLLECTION", files=("v1.mp4", "sub/v2.mp4"))
        self.do_import(p, media_root=media_root)
        self.assertEqual((media_root / "vids/sub/v2.mp4").read_bytes(), b"media:sub/v2.mp4" * 10)
        self.assertEqual((self.ws.base_dir / "datasets/vids/data/sub/v2.mp4").read_bytes(), b"media:sub/v2.mp4" * 10)
        ds = Dataset.from_json((self.ws.base_dir / "datasets/vids/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(ds.base_data_path), (media_root / "vids").resolve())

    def test_image_collections_are_placed_like_videos(self):
        media_root = self.tmp / "imgs"
        media_root.mkdir()
        self.do_import(self.make_bundle(dataset="imgs", type_="IMAGE_COLLECTION", files=("i1.png",)), media_root=media_root)
        self.assertTrue((media_root / "imgs/i1.png").is_file())

    def test_videos_need_a_folder_and_it_must_not_already_hold_that_dataset(self):
        p = self.make_bundle(dataset="vids", type_="VIDEO_COLLECTION", files=("v1.mp4",))
        with self.assertRaisesRegex(wb.BundleError, "Choose the folder"):
            self.do_import(p)
        media_root = self.tmp / "videos"
        (media_root / "vids").mkdir(parents=True)
        (media_root / "vids" / "originals.mp4").write_bytes(b"precious")
        with self.assertRaisesRegex(wb.BundleError, "never"):
            self.do_import(p, media_root=media_root)
        self.assertEqual((media_root / "vids" / "originals.mp4").read_bytes(), b"precious")

    def test_only_what_is_ticked_is_imported(self):
        self.do_import(self.make_bundle(project="proj"), chosen={wb.dataset_key("ds")})
        base = self.ws.base_dir
        self.assertTrue((base / "datasets/ds").is_dir())
        self.assertFalse((base / "workflows/pose_estimation/labels/ds").exists())
        self.assertFalse((base / "projects/proj.json").exists())

    def test_labels_alone_attach_to_a_dataset_the_workspace_already_has(self):
        self.add_dataset(self.ws, "ds", frame_names(2))
        self.do_import(self.make_bundle(), chosen={wb.labels_key("ds")})
        self.assertTrue((self.ws.base_dir / "workflows/pose_estimation/labels/ds/labels.h5").is_file())

    def test_labels_for_a_dataset_the_workspace_lacks_are_refused(self):
        b = self.open(self.make_bundle())
        problems = wb.validate_choices(self.ws, b, {wb.labels_key("ds")}, None)
        self.assertTrue(any("nowhere to go" in p for p in problems))

    def test_existing_things_are_marked_and_not_offered_by_default(self):
        self.do_import(self.make_bundle(project="proj"))
        b = self.open(self.tmp / "b.zip")
        items = wb.plan_import(self.ws, b)
        self.assertTrue(all(i.exists for i in items))
        self.assertEqual(wb.default_choices(items), set())

    def test_replacing_a_dataset_replaces_it_and_keeps_its_external_media(self):
        media_root = self.tmp / "videos"
        media_root.mkdir()
        old_media = self.tmp / "old_media"
        old_media.mkdir()
        self.add_dataset(self.ws, "vids", ["v1.mp4"], DatasetType.VIDEO_COLLECTION, media_root=old_media)
        (old_media / "v1.mp4").write_bytes(b"the originals")
        p = self.make_bundle(dataset="vids", type_="VIDEO_COLLECTION", files=("v1.mp4",), labels=False)
        self.do_import(p, chosen={wb.dataset_key("vids")}, media_root=media_root)
        ds = Dataset.from_json((self.ws.base_dir / "datasets/vids/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(ds.base_data_path), (media_root / "vids").resolve())
        self.assertEqual((old_media / "v1.mp4").read_bytes(), b"the originals")      # not the bundle's to delete

    def test_progress_is_reported_and_finishes_at_100(self):
        seen = []
        self.do_import(self.make_bundle(), progress_cb=lambda m, p: seen.append(p))
        self.assertEqual((seen[-1], seen == sorted(seen)), (100, True))


class ImportLeavesNothingHalfDone(BundleCase):
    """A corrupt, truncated, cancelled or failing import leaves the workspace exactly as it was."""

    def setUp(self):
        super().setUp()
        self.ws = self.new_workspace("ws")
        self.add_dataset(self.ws, "keep", frame_names(2))
        self.before = tree(self.ws.base_dir)

    def attempt(self, path, chosen=None, expect=wb.BundleError, **kw):
        b = self.open(path)
        chosen = chosen if chosen is not None else wb.default_choices(wb.plan_import(self.ws, b))
        with self.assertRaises(expect) as cm:
            wb.import_bundle(self.ws, b, chosen, **kw)
        self.assertEqual(tree(self.ws.base_dir), self.before)
        return cm.exception

    def corrupt(self, path, needle: bytes):
        data = bytearray(Path(path).read_bytes())
        i = bytes(data).index(needle)
        data[i] ^= 0xFF
        Path(path).write_bytes(bytes(data))

    def test_a_damaged_file_stops_the_whole_import_by_its_checksum(self):
        p = self.make_bundle()
        self.corrupt(p, b"media:a/f2.png")
        err = self.attempt(p)
        self.assertIn("damaged", str(err))

    def test_a_truncated_bundle_is_refused_up_front(self):
        p = self.make_bundle()
        p.write_bytes(p.read_bytes()[:-200])
        with self.assertRaises(wb.BundleError):
            wb.WhiskerBundle.open(p)
        self.assertEqual(tree(self.ws.base_dir), self.before)

    def test_cancelling_part_way_leaves_the_workspace_alone(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 3

        err = self.attempt(self.make_bundle(files=[f"a/f{i}.png" for i in range(20)]), cancel_cb=cancel)
        self.assertIn("cancel", str(err).lower())

    def test_a_failure_while_moving_things_into_place_is_undone(self):
        """Replace an existing dataset and its labels, then fail on the last move: the old ones must come back."""
        self.add_dataset(self.ws, "ds", ["old.png"])
        self.add_pose_labels(self.ws, "ds", ["old.png"])
        self.before = tree(self.ws.base_dir)
        old_manifest = (self.ws.base_dir / "datasets/ds/manifest.json").read_bytes()
        p = self.make_bundle(project="proj")
        real = wb._Journal.place
        n = {"c": 0}

        def flaky(self_, staged, final, replace=True):
            n["c"] += 1
            if n["c"] == 3:
                raise OSError("disk went away")
            return real(self_, staged, final, replace)

        b = self.open(p)
        chosen = {i.key for i in wb.plan_import(self.ws, b)}
        with mock.patch.object(wb._Journal, "place", flaky), self.assertRaisesRegex(wb.BundleError, "disk went away"):
            wb.import_bundle(self.ws, b, chosen)
        self.assertEqual(tree(self.ws.base_dir), self.before)
        self.assertEqual((self.ws.base_dir / "datasets/ds/manifest.json").read_bytes(), old_manifest)

    def test_a_folder_another_program_briefly_holds_open_is_retried(self):
        """Windows refuses to rename a folder while antivirus or the indexer has one of its new files open."""
        real, calls = os.replace, {"n": 0}

        def busy_twice(src, dst):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError(5, "Access is denied")
            return real(src, dst)

        a, b = self.tmp / "a", self.tmp / "b"
        a.mkdir()
        with mock.patch.object(wb.os, "replace", busy_twice), mock.patch.object(wb.time, "sleep"):
            wb._replace(a, b)
        self.assertEqual(calls["n"], 3)
        self.assertTrue(b.is_dir())

    def test_a_folder_that_stays_locked_fails_cleanly_and_leaves_the_workspace_alone(self):
        b = self.open(self.make_bundle())
        chosen = wb.default_choices(wb.plan_import(self.ws, b))
        with mock.patch.object(wb.os, "replace", side_effect=PermissionError(5, "Access is denied")),                 mock.patch.object(wb.time, "sleep"):
            with self.assertRaisesRegex(wb.BundleError, "Nothing was changed"):
                wb.import_bundle(self.ws, b, chosen)
        self.assertEqual(tree(self.ws.base_dir), self.before)

    @unittest.skipUnless(os.name == "nt", "the 259-character limit is a Windows one")
    def test_paths_too_long_for_windows_are_refused_up_front_when_long_paths_are_off(self):
        long_rel = "/".join(["d" * 60] * 4) + "/frame.png"             # 4 x 60 characters deep: over the limit anywhere
        p = self.tmp / "deep.zip"
        write_zip(p, {"datasets/ds/manifest.json": ds_manifest("ds", "FRAME_SUBSET", [long_rel]),
                      f"datasets/ds/media/{long_rel}": b"x"},
                  manifest_for(ds_entry("ds", files=[long_rel])))
        b = self.open(p)
        chosen = wb.default_choices(wb.plan_import(self.ws, b))
        with mock.patch.object(wb, "_windows_long_paths_enabled", return_value=False):
            with self.assertRaisesRegex(wb.BundleError, "Windows allows 259"):
                wb.import_bundle(self.ws, b, chosen)
        self.assertEqual(tree(self.ws.base_dir), self.before)

    def test_a_dataset_whose_files_are_missing_from_the_bundle_still_imports_and_says_so(self):
        p = self.tmp / "partial.zip"
        files = ["a.png", "b.png", "c.png"]
        write_zip(p, {"datasets/ds/manifest.json": ds_manifest("ds", "FRAME_SUBSET", files),
                      "datasets/ds/media/a.png": b"A"},
                  manifest_for(ds_entry("ds", files=files, missing_media=["b.png", "c.png"])))
        b = self.open(p)
        r = wb.import_bundle(self.ws, b, wb.default_choices(wb.plan_import(self.ws, b)))
        self.assertEqual(r["missing"], {"ds": 2})
        self.assertTrue(any("weren't in the bundle" in n for n in r["notes"]))


# --------------------------------------------------------------------------- #
# Exporting
# --------------------------------------------------------------------------- #


class Exporting(BundleCase):
    def setUp(self):
        super().setUp()
        self.ws = self.new_workspace("src")
        self.add_project(self.ws, "proj")
        self.files = frame_names(3, "clip")
        self.add_dataset(self.ws, "d1", self.files)
        self.add_pose_labels(self.ws, "d1", self.files[:2])
        self.add_behavior_labels(self.ws, "d1", ["a", "b"])
        run = self.ws.base_dir / "workflows/pose_estimation/predictions/run1/d1"
        run.mkdir(parents=True)
        (run / "predictions.h5").write_bytes(b"p" * 100)
        (run / "metadata.json").write_text('{"k": 1}')

    def export(self, names=("d1",), projects=("proj",), runs=(("pose_estimation", "run1"),), **kw):
        plan = wb.build_export_plan(self.ws, list(names), list(projects), list(runs))
        out = self.tmp / kw.pop("name", "out.zip")
        return wb.export_bundle(self.ws, plan, out, **kw), out

    def test_the_plan_lists_everything_going_into_the_bundle_in_the_order_a_person_thinks_of_it(self):
        plan = wb.build_export_plan(self.ws, ["d1"], ["proj"], [("pose_estimation", "run1")])
        listed = [(g.dataset, g.kind, g.title, [f for f, _n in g.files]) for g in plan.groups()]
        self.assertEqual(listed, [
            ("d1", "manifest", "Dataset manifest", ["manifest.json"]),
            ("d1", "media", "Media — 3 images", self.files),
            ("d1", "labels", "Labels — behavior classification", ["labels.h5"]),
            ("d1", "labels", "Labels — pose estimation", ["labels.h5", "metadata.json"]),
            ("d1", "predictions", "Prediction results — run1", ["metadata.json", "predictions.h5"]),
            ("", "project", "Project — proj", ["proj.json"]),
        ])
        self.assertEqual(sum(g.bytes for g in plan.groups()), plan.total_bytes)         # nothing is unaccounted for
        self.assertEqual(sorted(n for n, _p, _c in plan.entries), sorted(plan.sizes))

    def test_a_dataset_with_no_labels_says_so_rather_than_saying_nothing(self):
        self.add_dataset(self.ws, "bare", ["v.mp4"], DatasetType.VIDEO_COLLECTION)
        groups = [g for g in wb.build_export_plan(self.ws, ["bare"]).groups() if g.kind == "labels"]
        self.assertEqual([(g.title, g.files, g.note) for g in groups], [("Labels", [], "none for this dataset")])

    def test_unreachable_media_show_up_as_a_problem_on_the_media_line(self):
        (Path(self.ws.datasets.get("d1").base_data_path) / self.files[1]).unlink()
        media = next(g for g in wb.build_export_plan(self.ws, ["d1"]).groups() if g.kind == "media")
        self.assertEqual((len(media.files), media.problem), (2, True))
        self.assertIn("1 not found on disk", media.note)

    def test_writes_the_layout_whisker_reads(self):
        r, out = self.export()
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        self.assertEqual(names[0], wb.MANIFEST_NAME)
        self.assertEqual(sorted(names[1:]), sorted([
            "datasets/d1/manifest.json",
            *[f"datasets/d1/media/{f}" for f in self.files],
            "workflows/pose_estimation/labels/d1/labels.h5", "workflows/pose_estimation/labels/d1/metadata.json",
            "workflows/behavior_classification/labels/d1/labels.h5",
            "workflows/pose_estimation/predictions/run1/d1/predictions.h5",
            "workflows/pose_estimation/predictions/run1/d1/metadata.json",
            "projects/proj.json"]))
        self.assertEqual(r["problems"], [])

    def test_media_and_hdf5_are_stored_and_json_is_deflated(self):
        _, out = self.export()
        with zipfile.ZipFile(out) as z:
            kinds = {i.filename: i.compress_type for i in z.infolist()}
        for name, kind in kinds.items():
            expected = zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED
            self.assertEqual(kind, expected, name)

    def test_the_manifest_describes_the_bundle_the_way_whisker_does(self):
        _, out = self.export()
        with zipfile.ZipFile(out) as z:
            m = json.loads(z.read(wb.MANIFEST_NAME))
            rest = [i for i in z.infolist() if i.filename != wb.MANIFEST_NAME]
            media = [i for i in rest if "/media/" in i.filename]
        self.assertEqual((m["format"], m["format_version"]), ("whisker-bundle", 1))
        self.assertEqual((m["file_count"], m["total_bytes"]), (len(rest), sum(i.file_size for i in rest)))
        d = m["datasets"][0]
        self.assertEqual((d["name"], d["type"], d["file_count"], d["media_bytes"], d["missing_media"], d["multi_arena"]),
                         ("d1", "IMAGE_COLLECTION", 3, sum(i.file_size for i in media), [], False))
        self.assertEqual(d["label_workflows"], ["behavior_classification", "pose_estimation"])
        self.assertEqual(m["projects"], ["proj"])
        self.assertEqual(m["prediction_runs"], [{"workflow": "pose_estimation", "run_name": "run1", "datasets": ["d1"]}])

    def test_it_reads_back_through_the_importer_and_arrives_byte_for_byte(self):
        _, out = self.export()
        dst = self.new_workspace("dst")
        media_root = self.tmp / "imgs"
        media_root.mkdir()
        with wb.WhiskerBundle.open(out) as b:
            wb.import_bundle(dst, b, wb.default_choices(wb.plan_import(dst, b)), media_root)
        for rel in ("workflows/pose_estimation/labels/d1/labels.h5", "workflows/pose_estimation/labels/d1/metadata.json",
                    "workflows/behavior_classification/labels/d1/labels.h5", "projects/proj.json",
                    "workflows/pose_estimation/predictions/run1/d1/predictions.h5"):
            self.assertEqual(sha(self.ws.base_dir / rel), sha(dst.base_dir / rel), rel)
        for f in self.files:
            self.assertEqual((media_root / "d1" / f).read_bytes(), b"x" * 16)

    def test_labels_are_always_included_and_predictions_and_projects_only_when_chosen(self):
        _, out = self.export(projects=(), runs=())
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        self.assertFalse(any(n.startswith(("projects/", "workflows/pose_estimation/predictions")) for n in names))
        self.assertTrue(any("labels/d1/labels.h5" in n for n in names))

    def test_media_that_cannot_be_found_are_listed_and_the_bundle_is_still_sound(self):
        (Path(self.ws.datasets.get("d1").base_data_path) / self.files[1]).unlink()
        r, out = self.export()
        self.assertEqual((r["num_missing"], r["problems"]), (1, []))
        with zipfile.ZipFile(out) as z:
            d = json.loads(z.read(wb.MANIFEST_NAME))["datasets"][0]
        self.assertEqual((d["missing_media"], d["file_count"]), ([self.files[1]], 3))

    def test_a_stale_base_data_path_is_no_problem_when_the_workspace_holds_the_media(self):
        ds = self.ws.datasets.get("d1")
        data = self.ws.base_dir / "datasets/d1/data"
        for f in self.files:
            (data / f).parent.mkdir(parents=True, exist_ok=True)
            (data / f).write_bytes(b"in the workspace")
        ds.base_data_path = "D:\\gone\\for\\good"
        r, out = self.export()
        self.assertEqual(r["num_missing"], 0)
        with zipfile.ZipFile(out) as z:
            self.assertEqual(z.read(f"datasets/d1/media/{self.files[0]}"), b"in the workspace")

    def test_cancelling_leaves_nothing_behind(self):
        plan = wb.build_export_plan(self.ws, ["d1"], ["proj"])
        out = self.tmp / "cancelled.zip"
        with self.assertRaisesRegex(wb.BundleError, "cancel"):
            wb.export_bundle(self.ws, plan, out, cancel_cb=lambda: True)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("cancelled*")), [])

    def test_an_existing_bundle_is_only_replaced_when_asked(self):
        _, out = self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        r, _ = self.export(overwrite=True)
        self.assertEqual(r["problems"], [])

    def test_a_failed_export_does_not_replace_a_good_existing_bundle(self):
        _, out = self.export()
        good = out.read_bytes()
        plan = wb.build_export_plan(self.ws, ["d1"], ["proj"])
        with mock.patch.object(wb, "verify_bundle", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                wb.export_bundle(self.ws, plan, out, overwrite=True)
        self.assertEqual(out.read_bytes(), good)
        self.assertFalse(out.with_name(out.name + ".part").exists())

    def test_choosing_nothing_or_something_unknown_is_refused(self):
        with self.assertRaisesRegex(wb.BundleError, "at least one"):
            wb.build_export_plan(self.ws, [])
        with self.assertRaisesRegex(wb.BundleError, "isn't in this workspace"):
            wb.build_export_plan(self.ws, ["nope"])
        with self.assertRaisesRegex(wb.BundleError, "Project 'nope'"):
            wb.build_export_plan(self.ws, ["d1"], ["nope"])

    def test_verifying_a_bundle_finds_a_damaged_file(self):
        _, out = self.export()
        data = bytearray(out.read_bytes())
        data[bytes(data).index(b"p" * 100)] ^= 0xFF
        out.write_bytes(bytes(data))
        self.assertTrue(wb.verify_bundle(out))


if __name__ == "__main__":
    unittest.main()
