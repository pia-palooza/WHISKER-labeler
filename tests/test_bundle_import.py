"""Tests for the one-pick bundle import (whisker.core.bundle_import)."""
import json
import shutil
from pathlib import Path
from unittest import mock

from fixtures import (
    BEHAVIORS, BODY_PARTS, WorkspaceCase, frame_names, make_behavior_dataset, make_pose_dataset,
)
from whisker.core import bundle_import as bi
from whisker.core.bundle_import import ImportSelection, LabelPolicy
from whisker.core.study.dataset import DatasetType
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset


class BundleCase(WorkspaceCase):
    """A source workspace with one labeled dataset, exported to ``self.bundle``."""

    def setUp(self):
        super().setUp()
        self.src = self.new_workspace("src")
        self.add_project(self.src)
        self.files = frame_names(5)
        self.add_dataset(self.src, "d1", self.files)
        self.add_pose_labels(self.src, "d1", self.files[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.bundle = self.export_bundle(self.src, "d1")
        self.dst = self.new_workspace("dst")

    def rescan(self, ws=None):
        ws = ws or self.dst
        ws.scan_projects(); ws.scan_datasets(); ws.scan_labels()

    def pose_of(self, ws, name):
        return PoseDataset.from_file(ws.pose_labels.base_dir / name / "labels.h5")

    def behavior_of(self, ws, name):
        return BehaviorDataset.from_file(ws.behavior_labels.base_dir / name / "labels.h5")

    def run_import(self, sel, contents=None, ws=None):
        ws = ws or self.dst
        contents = contents or bi.inspect_bundle(self.bundle)
        result = bi.import_from_bundle(ws, contents, sel)
        self.rescan(ws)
        return result


# ----------------------------------------------------------------- locating


class LocateTests(BundleCase):
    def test_finds_the_export_from_every_kind_of_pick(self):
        inside_file = self.bundle / "dataset" / "manifest.json"
        for pick in (self.bundle, self.bundle / "export_info.json", inside_file,
                     self.bundle / "frames", self.bundle / "pose_labels"):
            loc = bi.locate_bundle(pick)
            self.assertEqual(loc.root, self.bundle, f"picked {pick}")

    def test_finds_an_export_nested_one_level_deeper_as_unzipping_makes(self):
        outer = self.tmp / "download" / "bundle_from_email"
        outer.parent.mkdir()
        shutil.copytree(self.bundle, outer / "d1_bundle")
        loc = bi.locate_bundle(outer)
        self.assertEqual(loc.root, outer / "d1_bundle")
        self.assertIn("inside the folder you picked", loc.message)

    def test_several_exports_are_offered_not_guessed(self):
        box = self.tmp / "many"
        shutil.copytree(self.bundle, box / "one")
        shutil.copytree(self.bundle, box / "two")
        loc = bi.locate_bundle(box)
        self.assertIsNone(loc.root)
        self.assertEqual([p.name for p in loc.candidates], ["one", "two"])

    def test_pick_inside_the_export_says_so(self):
        loc = bi.locate_bundle(self.bundle / "frames")
        self.assertIn("inside the export", loc.message)

    def test_failure_says_what_was_searched_and_what_to_do(self):
        empty = self.tmp / "nothing_here"
        empty.mkdir()
        loc = bi.locate_bundle(empty)
        self.assertFalse(loc.found)
        for expected in ("export_info.json", "levels down", "levels up", "README.txt"):
            self.assertIn(expected, loc.message)

    def test_hand_made_json_folder_points_to_manual_mode(self):
        (self.tmp / "loose").mkdir()
        (self.tmp / "loose" / "manifest.json").write_text("{}")
        self.assertIn("Import from separate files", bi.locate_bundle(self.tmp / "loose").message)

    def test_missing_path(self):
        self.assertIn("doesn't exist", bi.locate_bundle(self.tmp / "nope").message)

    def test_does_not_wander_into_media_folders(self):
        # A decoy export description buried in the media folder must not be found.
        decoy = self.tmp / "box" / "frames" / "x"
        decoy.mkdir(parents=True)
        (decoy / "export_info.json").write_text("{}")
        self.assertFalse(bi.locate_bundle(self.tmp / "box").found)


# ---------------------------------------------------------------- inspecting


class InspectTests(BundleCase):
    def test_full_bundle_reports_every_part(self):
        c = bi.inspect_bundle(self.bundle)
        for part in (c.project, c.dataset, c.pose, c.behavior):
            self.assertTrue(part.present and part.ok, part)
        self.assertIn("5 frames", c.dataset.summary)
        self.assertIn("3 labeled frame", c.pose.summary)
        self.assertIn("2 labeled video", c.behavior.summary)
        self.assertEqual(c.dataset_name, "d1")
        self.assertEqual(c.media_dir, self.bundle / "frames")
        self.assertFalse(c.needs_media_folder)

    def test_parts_left_out_of_the_export_are_absent_not_broken(self):
        b = self.export_bundle(self.src, "d1", name="labels_only", include_project=False, include_media=False,
                               include_behavior=False)
        c = bi.inspect_bundle(b)
        self.assertFalse(c.project.present)
        self.assertFalse(c.behavior.present)
        self.assertTrue(c.pose.ok)
        self.assertEqual(c.project.summary, "Not in this export")

    def test_reference_only_uses_the_original_location_when_it_still_exists(self):
        b = self.export_bundle(self.src, "d1", name="ref", include_media=False)
        c = bi.inspect_bundle(b)
        self.assertFalse(c.media_included)
        self.assertEqual(Path(c.media_dir), Path(self.tmp / "media" / "d1"))
        self.assertIn("original location", c.dataset.summary)

    def test_reference_only_asks_for_the_folder_when_the_original_is_gone(self):
        b = self.export_bundle(self.src, "d1", name="ref", include_media=False)
        shutil.rmtree(self.tmp / "media" / "d1")
        c = bi.inspect_bundle(b)
        self.assertTrue(c.needs_media_folder)
        self.assertTrue(c.dataset.ok)
        self.assertIn("asked where they are", c.dataset.summary)

    def test_a_listed_but_missing_label_file_is_reported_not_crashed_on(self):
        (self.bundle / "pose_labels" / "labels.h5").unlink()
        c = bi.inspect_bundle(self.bundle)
        self.assertTrue(c.pose.present)
        self.assertFalse(c.pose.ok)
        self.assertIn("missing", c.pose.problem)
        self.assertTrue(c.behavior.ok)                   # the other parts are unaffected

    def test_missing_media_in_an_included_export_is_reported(self):
        (self.bundle / "frames" / "frame_0001.png").unlink()
        c = bi.inspect_bundle(self.bundle)
        self.assertFalse(c.dataset.ok)
        self.assertIn("1 of 5", c.dataset.problem)

    def test_damaged_description_gives_a_clear_error(self):
        (self.bundle / "export_info.json").write_text("{ not json")
        with self.assertRaises(bi.BundleImportError) as cm:
            bi.inspect_bundle(self.bundle)
        self.assertIn("damaged", str(cm.exception))

    def test_newer_format_version_warns(self):
        info = json.loads((self.bundle / "export_info.json").read_text())
        info["bundle_format_version"] = "9.0"
        (self.bundle / "export_info.json").write_text(json.dumps(info))
        self.assertTrue(any("newer version" in w for w in bi.inspect_bundle(self.bundle).warnings))


# -------------------------------------------------------- importing subsets


class SubsetImportTests(BundleCase):
    def sel(self, **kw):
        base = dict(dataset_name="d1")
        base.update(kw)
        return ImportSelection(**base)

    def test_everything(self):
        r = self.run_import(self.sel(project=True, dataset=True, pose=True, behavior=True))
        self.assertTrue(r["project_installed"])
        self.assertEqual(r["num_media_copied"], 5)
        self.assertIsNotNone(self.dst.projects.get("proj"))
        self.assertEqual(len(self.pose_of(self.dst, "d1").frame_indices), 3)
        self.assertEqual(len(self.behavior_of(self.dst, "d1").bouts), 2)

    def test_project_only(self):
        self.run_import(self.sel(project=True))
        self.assertIsNotNone(self.dst.projects.get("proj"))
        self.assertEqual(self.dst.datasets.keys(), [])
        self.assertFalse(self.dst.pose_labels.has_pose_labels("d1"))

    def test_videos_or_frames_only(self):
        self.run_import(self.sel(dataset=True))
        self.assertEqual(len(self.dst.datasets.get("d1").files), 5)
        self.assertFalse(self.dst.pose_labels.has_pose_labels("d1"))
        self.assertFalse(self.dst.behavior_labels.has_behavior_labels("d1"))
        self.assertEqual(list(self.dst.projects.keys()), [])

    def test_dataset_plus_only_the_behavior_labels(self):
        self.run_import(self.sel(dataset=True, behavior=True))
        self.assertTrue(self.dst.behavior_labels.has_behavior_labels("d1"))
        self.assertFalse(self.dst.pose_labels.has_pose_labels("d1"))

    def test_dataset_can_be_imported_under_another_name_and_labels_follow(self):
        self.run_import(self.sel(dataset=True, pose=True, dataset_name="renamed"))
        self.assertEqual(self.dst.datasets.keys(), ["renamed"])
        meta = json.loads((self.dst.pose_labels.base_dir / "renamed" / "metadata.json").read_text())
        self.assertEqual(meta["dataset_name"], "renamed")

    def test_labels_only_attach_to_an_existing_dataset(self):
        self.add_dataset(self.dst, "mine", self.files, media_root=self.tmp / "dstmedia")
        r = self.run_import(self.sel(pose=True, behavior=False, target_dataset="mine"))
        self.assertEqual(r["pose"]["mode"], "added")
        self.assertEqual(len(self.pose_of(self.dst, "mine").frame_indices), 3)
        self.assertNotIn("d1", self.dst.datasets.keys())

    def test_nothing_ticked_is_refused(self):
        self.assertEqual(bi.validate_selection(self.dst, bi.inspect_bundle(self.bundle), self.sel()),
                         ["Tick at least one thing to import."])

    def test_reference_only_needs_a_media_folder_and_validates_it(self):
        b = self.export_bundle(self.src, "d1", name="ref", include_media=False)
        shutil.rmtree(self.tmp / "media" / "d1")
        c = bi.inspect_bundle(b)
        sel = ImportSelection(dataset=True, dataset_name="d1")
        self.assertIn("Choose the folder", bi.validate_selection(self.dst, c, sel)[0])
        sel.media_dir = self.tmp                          # wrong folder
        self.assertIn("not in this folder", bi.validate_selection(self.dst, c, sel)[0])
        # supplying the right folder makes it importable
        supplied = self.tmp / "supplied"
        for f in self.files:
            (supplied / f).parent.mkdir(parents=True, exist_ok=True)
            (supplied / f).write_bytes(b"x")
        sel.media_dir = supplied
        self.assertEqual(bi.validate_selection(self.dst, c, sel), [])
        self.assertEqual(bi.import_from_bundle(self.dst, c, sel)["num_media_copied"], 5)

    def test_existing_dataset_name_blocks_until_renamed_or_replaced(self):
        self.run_import(self.sel(dataset=True))
        c = bi.inspect_bundle(self.bundle)
        problems = bi.validate_selection(self.dst, c, self.sel(dataset=True))
        self.assertIn("already have a dataset called 'd1'", problems[0])
        self.assertEqual(bi.validate_selection(self.dst, c, self.sel(dataset=True, overwrite_dataset=True)), [])
        self.assertEqual(bi.validate_selection(self.dst, c, self.sel(dataset=True, dataset_name="d1_2")), [])

    def test_a_new_project_can_be_saved_under_another_name(self):
        self.run_import(self.sel(project=True, project_name="renamed_proj"))
        self.assertEqual(sorted(self.dst.projects.keys()), ["renamed_proj"])
        p = self.dst.projects.get("renamed_proj")
        self.assertEqual(p.name, "renamed_proj")                      # the name *inside* the file changed too
        self.assertEqual((p.body_parts, p.behaviors), (BODY_PARTS, BEHAVIORS))

    def test_a_taken_project_name_needs_another_name_or_replace(self):
        self.run_import(self.sel(project=True))
        c = bi.inspect_bundle(self.bundle)
        problems = bi.validate_selection(self.dst, c, self.sel(project=True))
        self.assertIn("already have a project called 'proj'", problems[0])
        self.assertEqual(bi.validate_selection(self.dst, c, self.sel(project=True, project_name="proj_2")), [])
        self.assertEqual(bi.validate_selection(self.dst, c, self.sel(project=True, overwrite_project=True)), [])
        (self.dst.projects.base_dir / "proj.json").write_text(
            self.dst.projects.get("proj").model_copy(update={"behaviors": ["x"]}).model_dump_json())
        self.rescan()
        self.run_import(self.sel(project=True, overwrite_project=True))
        self.assertEqual(self.dst.projects.get("proj").behaviors, BEHAVIORS)

    def test_bad_project_names_are_refused(self):
        c = bi.inspect_bundle(self.bundle)
        for bad in ("a/b", "what?", "trail."):
            problems = bi.validate_selection(self.dst, c, self.sel(project=True, project_name=bad))
            self.assertTrue(problems and "project name can't" in problems[0], bad)
        # surrounding spaces are trimmed, not refused
        self.assertEqual(bi.validate_selection(self.dst, c, self.sel(project=True, project_name="  padded  ")), [])
        self.run_import(self.sel(project=True, project_name="  padded  "))
        self.assertEqual(list(self.dst.projects.keys()), ["padded"])

    def test_using_an_existing_project_installs_nothing_and_says_what_it_lacks(self):
        self.add_project(self.dst, "mine", body_parts=["nose"], identities=["mouse1"], behaviors=["groom"])
        r = self.run_import(self.sel(project=True, project_mode="existing", existing_project="mine",
                                     dataset=True, pose=True, behavior=True))
        self.assertEqual(sorted(self.dst.projects.keys()), ["mine"])          # the export's project was NOT added
        self.assertEqual((r["project_name"], r["project_installed"], r["project_existing"]), ("mine", False, True))
        notes = " ".join(r["notes"])
        self.assertIn("Using your existing project 'mine'", notes)
        self.assertIn("body parts tail_base", notes)
        self.assertIn("behaviors rear", notes)
        self.assertTrue(self.dst.pose_labels.has_pose_labels("d1"))             # the rest imported as normal

    def test_using_a_project_that_covers_the_labels_says_nothing_about_gaps(self):
        self.add_project(self.dst, "mine")                                     # same body parts / identities / behaviors
        r = self.run_import(self.sel(project=True, project_mode="existing", existing_project="mine", dataset=True, pose=True))
        self.assertEqual(r["notes"], ["Using your existing project 'mine'."])

    def test_an_existing_project_on_its_own_imports_nothing_and_says_so(self):
        self.add_project(self.dst, "mine")
        c = bi.inspect_bundle(self.bundle)
        problems = bi.validate_selection(self.dst, c, self.sel(project=True, project_mode="existing", existing_project="mine"))
        self.assertIn("imports nothing", problems[0])

    def test_an_existing_project_must_actually_exist(self):
        c = bi.inspect_bundle(self.bundle)
        for missing in ("", "ghost"):
            problems = bi.validate_selection(self.dst, c, self.sel(project=True, project_mode="existing",
                                                                   existing_project=missing, dataset=True))
            self.assertIn("Choose which of your projects to use.", problems)

    def test_project_gaps(self):
        self.add_project(self.dst, "mine", body_parts=["nose", "tail_base"], identities=["other"], behaviors=["groom", "rear"])
        c = bi.inspect_bundle(self.bundle)
        self.assertEqual(bi.project_gaps(self.dst, c, "mine"), ["identities mouse1"])
        self.assertEqual(bi.project_gaps(self.dst, c, "mine", pose=False), [])
        self.assertEqual(bi.project_gaps(self.dst, c, "nope"), [])

    def test_unique_project_names(self):
        self.add_project(self.dst, "proj")
        self.add_project(self.dst, "proj_2")
        self.assertEqual(bi.unique_project_name(self.dst, "proj"), "proj_3")
        self.assertEqual(bi.unique_project_name(self.dst, "fresh"), "fresh")


# --------------------------------------------------------- default selection


class DefaultSelectionTests(BundleCase):
    def test_fresh_workspace_ticks_everything(self):
        sel = bi.default_selection(self.dst, bi.inspect_bundle(self.bundle))
        self.assertTrue(sel.project and sel.dataset and sel.pose and sel.behavior)
        self.assertEqual(sel.dataset_name, "d1")
        self.assertEqual(bi.validate_selection(self.dst, bi.inspect_bundle(self.bundle), sel), [])

    def test_reimporting_defaults_to_using_what_you_already_have(self):
        self.run_import(ImportSelection(project=True, dataset=True, dataset_name="d1"))
        c = bi.inspect_bundle(self.bundle)
        sel = bi.default_selection(self.dst, c)
        # the project: use mine (offered, ticked), with a free name ready in case they'd rather add a copy
        self.assertTrue(sel.project)
        self.assertEqual((sel.project_mode, sel.existing_project, sel.project_name), ("existing", "proj", "proj_2"))
        # the dataset: labels attach to mine, its media aren't copied again
        self.assertFalse(sel.dataset)
        self.assertTrue(sel.pose and sel.behavior)
        self.assertEqual(sel.target_dataset, "d1")
        self.assertEqual(sel.dataset_name, "d1_2")                      # if they choose to add it as new
        self.assertEqual(bi.project_relation(self.dst, c), "identical")

    def test_a_fresh_workspace_defaults_to_adding_the_project_as_new(self):
        sel = bi.default_selection(self.dst, bi.inspect_bundle(self.bundle))
        self.assertEqual((sel.project, sel.project_mode, sel.project_name, sel.existing_project), (True, "new", "proj", ""))

    def test_project_relation_detects_a_different_definition(self):
        self.run_import(ImportSelection(project=True))
        (self.dst.projects.base_dir / "proj.json").write_text(
            self.dst.projects.get("proj").model_copy(update={"identities": ["other"]}).model_dump_json())
        self.rescan()
        self.assertEqual(bi.project_relation(self.dst, bi.inspect_bundle(self.bundle)), "different")

    def test_best_fitting_dataset_is_suggested_for_the_labels(self):
        # "d1" exists here but holds different files, so it's the wrong home for these labels;
        # "fits" has the right files and should be preselected instead.
        self.add_dataset(self.dst, "d1", [f"zzz_{i}.png" for i in range(5)], media_root=self.tmp / "m1")
        self.add_dataset(self.dst, "fits", self.files, media_root=self.tmp / "m2")
        c = bi.inspect_bundle(self.bundle)
        self.assertEqual(bi.rank_target_datasets(self.dst, c)[0][0], "fits")
        sel = bi.default_selection(self.dst, c)
        self.assertFalse(sel.dataset)
        self.assertEqual(sel.target_dataset, "fits")

    def test_no_preselection_when_nothing_fits(self):
        self.add_dataset(self.dst, "d1", [f"zzz_{i}.png" for i in range(5)], media_root=self.tmp / "m1")
        sel = bi.default_selection(self.dst, bi.inspect_bundle(self.bundle))
        self.assertEqual(sel.target_dataset, "d1")        # falls back to the same-named dataset


# -------------------------------------------------------------- mismatch report


class PoseReportTests(BundleCase):
    def setUp(self):
        super().setUp()
        self.h5 = self.bundle / "pose_labels" / "labels.h5"            # labels frame_0000..0002

    def report(self, dataset, **kw):
        return bi.analyze_pose_labels(self.dst, dataset, self.h5, **kw)

    def test_all_match(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        r = self.report("t")
        self.assertEqual((r.total, len(r.matched), len(r.unmatched)), (3, 3, 0))
        self.assertEqual(r.unlabeled_files, 2)                         # frames 3 and 4 have no label
        self.assertFalse(r.has_existing)

    def test_some_labels_have_no_file(self):
        self.add_dataset(self.dst, "t", self.files[1:], media_root=self.tmp / "m")   # lacks frame_0000
        r = self.report("t")
        self.assertEqual((len(r.matched), r.unmatched), (2, ["frame_0000.png"]))

    def test_nothing_matches_is_flagged(self):
        self.add_dataset(self.dst, "t", ["other_1.png", "other_2.png"], media_root=self.tmp / "m")
        self.assertTrue(self.report("t").nothing_matches)

    def test_path_separators_are_normalised_and_keys_follow_the_datasets_spelling(self):
        self.add_dataset(self.dst, "t", [f"sub\\{f}" for f in self.files], media_root=self.tmp / "m")
        # dataset says sub\frame_0000.png, labels say frame_0000.png -> no match ...
        self.assertTrue(self.report("t").nothing_matches)
        # ... but a slash-vs-backslash difference alone does match, and maps to the dataset's spelling
        labels = make_pose_dataset([f"sub/{f}" for f in self.files[:2]])
        labels.to_file(self.tmp / "slashes.h5")
        r = bi.analyze_pose_labels(self.dst, "t", self.tmp / "slashes.h5")
        self.assertEqual(len(r.matched), 2)
        self.assertEqual(r.key_map["sub/frame_0000.png"], "sub\\frame_0000.png")

    def test_overlap_and_incompatibility_are_reported_against_existing_labels(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_pose_labels(self.dst, "t", self.files[2:4])           # existing: frames 2,3
        r = self.report("t")
        self.assertEqual((r.existing_total, r.overlapping), (2, ["frame_0002.png"]))
        self.assertTrue(r.can_merge)
        self.add_pose_labels(self.dst, "t", self.files[2:4], body_parts=["nose", "ear"])
        r = self.report("t")
        self.assertFalse(r.can_merge)
        self.assertIn("body parts differ", r.problems[0])

    def test_warns_when_no_project_has_the_body_parts(self):
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.assertTrue(any("No project" in w for w in self.report("t").warnings))
        self.add_project(self.dst)
        self.assertEqual(self.report("t").warnings, [])


class BehaviorReportTests(BundleCase):
    h5 = property(lambda self: self.bundle / "behavior_labels" / "labels.h5")   # keys a.mp4, b.mp4

    def test_matches_by_video_file_name(self):
        self.add_dataset(self.dst, "v", ["a.mp4", "b.mp4", "c.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "v")
        r = bi.analyze_behavior_labels(self.dst, "v", self.h5)
        self.assertEqual((sorted(r.matched), r.unmatched, r.unlabeled_files), (["a.mp4", "b.mp4"], [], 1))

    def test_a_bare_stem_matches_and_is_normalised_to_the_file_name(self):
        make_behavior_dataset(["a"]).to_file(self.tmp / "stem.h5")
        self.add_dataset(self.dst, "v", ["sub/a.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "v")
        r = bi.analyze_behavior_labels(self.dst, "v", self.tmp / "stem.h5")
        self.assertEqual(r.key_map, {"a": "a.mp4"})

    def test_multi_arena_keys_match(self):
        from whisker.core.study.dataset import Dataset, MultiArenaConfig
        cfg = MultiArenaConfig(box_width=10, box_height=10, placements={"clip.mp4": [(0, 0), (20, 0)]})
        (self.tmp / "va").mkdir()
        (self.tmp / "va" / "clip.mp4").write_bytes(b"x")
        self.dst.add_dataset("arenas", Dataset(name="arenas", type=DatasetType.VIDEO_COLLECTION,
                                               base_data_path=str(self.tmp / "va"), files=["clip.mp4"], multi_arena=cfg))
        make_behavior_dataset(["clip_arena0", "clip_arena1", "clip_arena9"]).to_file(self.tmp / "ar.h5")
        r = bi.analyze_behavior_labels(self.dst, "arenas", self.tmp / "ar.h5")
        self.assertEqual((sorted(r.matched), r.unmatched, r.unlabeled_files), (["clip_arena0", "clip_arena1"], ["clip_arena9"], 0))

    def test_unrelated_dataset_matches_nothing(self):
        self.add_dataset(self.dst, "v", ["z.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "v")
        self.assertTrue(bi.analyze_behavior_labels(self.dst, "v", self.h5).nothing_matches)


# ------------------------------------------------------------------ merging


class MergeTests(BundleCase):
    def setUp(self):
        super().setUp()
        self.add_dataset(self.dst, "t", self.files, media_root=self.tmp / "m")
        self.add_project(self.dst)
        self.pose_h5 = self.bundle / "pose_labels" / "labels.h5"       # frames 0,1,2 at x0=10

    def x_of(self, ds, frame, part="nose"):
        return float(ds.keypoint_data.xs((frame, "mouse1", part))["x"])

    def test_pose_merge_imported_wins_on_overlap(self):
        self.add_pose_labels(self.dst, "t", self.files[2:5], x0=50.0)   # existing: 2,3,4 at x0=50
        r = bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.MERGE_IMPORTED)
        merged = self.pose_of(self.dst, "t")
        self.assertEqual((r["mode"], r["overlapping"], len(merged.frame_indices)), ("merged", 1, 5))
        self.assertEqual(self.x_of(merged, "frame_0002.png"), 10.0)   # imported won
        self.assertEqual(self.x_of(merged, "frame_0003.png"), 50.0)   # existing untouched
        self.assertEqual(self.x_of(merged, "frame_0000.png"), 10.0)   # newly added

    def test_pose_merge_existing_wins_on_overlap(self):
        self.add_pose_labels(self.dst, "t", self.files[2:5], x0=50.0)
        bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.MERGE_EXISTING)
        merged = self.pose_of(self.dst, "t")
        self.assertEqual(len(merged.frame_indices), 5)
        self.assertEqual(self.x_of(merged, "frame_0002.png"), 50.0)   # existing kept
        self.assertEqual(self.x_of(merged, "frame_0001.png"), 10.0)

    def test_pose_merge_keeps_metadata_in_step_with_the_labels(self):
        self.add_pose_labels(self.dst, "t", self.files[2:5], x0=50.0)
        bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.MERGE_IMPORTED)
        meta = json.loads((self.dst.pose_labels.base_dir / "t" / "metadata.json").read_text())
        self.assertEqual(meta["dataset_name"], "t")
        self.assertEqual(sorted(meta["frame_indices"]), sorted(self.files))
        self.rescan()
        self.assertEqual(self.dst.pose_labels.get_pose_labeled_image_keys_from_summary("t"), set(self.files))

    def test_labeled_frame_cache_is_refreshed_after_a_merge(self):
        # The app caches the labeled-frame list per dataset (it drives the check marks in the
        # data explorer). Read it first so it's cached, then merge and rescan like the GUI does.
        self.add_pose_labels(self.dst, "t", self.files[3:5], x0=50.0)
        self.assertEqual(self.dst.pose_labels.get_pose_labeled_image_keys_from_summary("t"), set(self.files[3:5]))
        bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.MERGE_IMPORTED)
        self.rescan()
        self.assertEqual(self.dst.pose_labels.get_pose_labeled_image_keys_from_summary("t"), set(self.files))

    def test_pose_replace_discards_existing(self):
        self.add_pose_labels(self.dst, "t", self.files[3:5], x0=50.0)
        bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.REPLACE)
        self.assertEqual(sorted(self.pose_of(self.dst, "t").frame_indices), self.files[:3])

    def test_pose_skip_changes_nothing(self):
        self.add_pose_labels(self.dst, "t", self.files[3:5], x0=50.0)
        self.assertFalse(bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.SKIP)["applied"])
        self.assertEqual(sorted(self.pose_of(self.dst, "t").frame_indices), self.files[3:5])

    def test_pose_merge_refuses_incompatible_body_parts(self):
        self.add_pose_labels(self.dst, "t", self.files[3:5], body_parts=["nose", "ear"])
        with self.assertRaises(bi.BundleImportError) as cm:
            bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.MERGE_IMPORTED)
        self.assertIn("body parts differ", str(cm.exception))
        self.assertEqual(sorted(self.pose_of(self.dst, "t").frame_indices), self.files[3:5])   # untouched

    def test_unmatched_labels_are_dropped_by_default_and_kept_on_request(self):
        self.add_dataset(self.dst, "small", self.files[1:], media_root=self.tmp / "m2")   # lacks frame_0000
        r = bi.apply_pose_labels(self.dst, "small", self.pose_h5, LabelPolicy.ADD)
        self.assertEqual((r["imported_frames"], r["dropped_unmatched"]), (2, 1))
        self.assertNotIn("frame_0000.png", self.pose_of(self.dst, "small").frame_indices)
        r = bi.apply_pose_labels(self.dst, "small", self.pose_h5, LabelPolicy.REPLACE, keep_unmatched=True)
        self.assertIn("frame_0000.png", self.pose_of(self.dst, "small").frame_indices)

    def test_importing_into_the_wrong_dataset_is_refused(self):
        self.add_dataset(self.dst, "wrong", ["q1.png", "q2.png"], media_root=self.tmp / "m3")
        with self.assertRaises(bi.BundleImportError) as cm:
            bi.apply_pose_labels(self.dst, "wrong", self.pose_h5, LabelPolicy.ADD)
        self.assertIn("wrong dataset", str(cm.exception))
        self.assertFalse((self.dst.pose_labels.base_dir / "wrong").exists())      # nothing left behind

    # -- behavior --

    def beh_h5(self):
        return self.bundle / "behavior_labels" / "labels.h5"          # a.mp4, b.mp4 both start at frame 0

    def setup_videos(self):
        self.add_dataset(self.dst, "v", ["a.mp4", "b.mp4", "c.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "vm")
        beh = make_behavior_dataset(["a.mp4", "c.mp4"], behaviors=["groom", "sniff"], start=100)
        beh.to_file(self.dst.behavior_labels.base_dir / "v" / "labels.h5")
        self.rescan()

    def bouts_by_video(self, ds):
        return {r.video_key: r.start_frame for r in ds.bouts.itertuples()}

    def test_behavior_merge_imported_wins(self):
        self.setup_videos()
        r = bi.apply_behavior_labels(self.dst, "v", self.beh_h5(), LabelPolicy.MERGE_IMPORTED)
        merged = self.behavior_of(self.dst, "v")
        self.assertEqual(self.bouts_by_video(merged), {"a.mp4": 0, "b.mp4": 0, "c.mp4": 100})
        self.assertEqual((r["mode"], r["overlapping"]), ("merged", 1))

    def test_behavior_merge_existing_wins(self):
        self.setup_videos()
        bi.apply_behavior_labels(self.dst, "v", self.beh_h5(), LabelPolicy.MERGE_EXISTING)
        self.assertEqual(self.bouts_by_video(self.behavior_of(self.dst, "v")), {"a.mp4": 100, "b.mp4": 0, "c.mp4": 100})

    def test_behavior_merge_combines_the_behavior_lists(self):
        self.setup_videos()                                             # existing: groom, sniff; imported: groom, rear
        bi.apply_behavior_labels(self.dst, "v", self.beh_h5(), LabelPolicy.MERGE_IMPORTED)
        self.assertEqual(self.behavior_of(self.dst, "v").behaviors, ["groom", "sniff", "rear"])

    def test_behavior_replace(self):
        self.setup_videos()
        bi.apply_behavior_labels(self.dst, "v", self.beh_h5(), LabelPolicy.REPLACE)
        self.assertEqual(self.bouts_by_video(self.behavior_of(self.dst, "v")), {"a.mp4": 0, "b.mp4": 0})

    def test_behavior_only_bouts_for_files_in_the_dataset_are_kept(self):
        self.add_dataset(self.dst, "v", ["a.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "vm")
        r = bi.apply_behavior_labels(self.dst, "v", self.beh_h5(), LabelPolicy.ADD)
        self.assertEqual((r["imported_videos"], r["dropped_unmatched"]), (1, 1))

    # -- safety --

    def test_a_failed_write_restores_the_previous_labels(self):
        self.add_pose_labels(self.dst, "t", self.files[3:5], x0=50.0)
        before = (self.dst.pose_labels.base_dir / "t" / "labels.h5").read_bytes()
        with mock.patch.object(type(self.dst.pose_labels), "write_poses_file", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.REPLACE)
        self.assertEqual((self.dst.pose_labels.base_dir / "t" / "labels.h5").read_bytes(), before)
        self.assertFalse(any(p.name.endswith(".import-backup") for p in self.dst.pose_labels.base_dir.iterdir()))

    def test_a_failed_first_write_leaves_no_half_made_folder(self):
        with mock.patch.object(type(self.dst.pose_labels), "write_poses_file", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                bi.apply_pose_labels(self.dst, "t", self.pose_h5, LabelPolicy.ADD)
        self.assertFalse((self.dst.pose_labels.base_dir / "t").exists())
