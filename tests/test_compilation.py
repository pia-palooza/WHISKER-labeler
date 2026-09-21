"""Tests for compilations: several datasets in one package (whisker.core.compilation)."""
import json
import shutil
from pathlib import Path
from unittest import mock

from fixtures import WorkspaceCase, frame_names, make_behavior_dataset
from whisker.core import bundle as fmt
from whisker.core import bundle_import as bi
from whisker.core import compilation as comp
from whisker.core import manual_import as mi
from whisker.core.bundle_import import ImportSelection, LabelPolicy
from whisker.core.compilation import CompilationItem
from whisker.core.study.dataset import DatasetType
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset


class CompCase(WorkspaceCase):
    """A source workspace with three differently-shaped datasets."""

    def setUp(self):
        super().setUp()
        self.src = self.new_workspace("src")
        self.add_project(self.src, "proj")
        self.add_project(self.src, "proj2", body_parts=["snout", "tail"], identities=["rat1"], behaviors=["dig"])
        self.frames = frame_names(5)
        self.add_dataset(self.src, "d1", self.frames)
        self.add_pose_labels(self.src, "d1", self.frames[:3])
        self.add_behavior_labels(self.src, "d1", ["a.mp4", "b.mp4"])
        self.add_dataset(self.src, "vids", ["a.mp4", "b.mp4"], DatasetType.VIDEO_COLLECTION, self.tmp / "vmedia")
        self.add_behavior_labels(self.src, "vids", ["a.mp4", "b.mp4"])
        self.add_dataset(self.src, "Odd [Name]", frame_names(3), media_root=self.tmp / "odd")
        self.add_pose_labels(self.src, "Odd [Name]", frame_names(3)[:2], body_parts=["snout", "tail"], individuals=["rat1"])
        self.dst = self.new_workspace("dst")

    # -- helpers ---------------------------------------------------------
    def items(self, **over):
        base = {
            "d1": CompilationItem("d1", "proj"),
            "vids": CompilationItem("vids", "proj"),
            "Odd [Name]": CompilationItem("Odd [Name]", "proj2"),
        }
        for k, v in over.items():
            base[k.replace("_", " ") if k not in base else k] = v
        return list(base.values())

    def export(self, items=None, name="pack", **kw):
        items = items or self.items()
        plans = comp.build_plans(self.src, items)
        return Path(comp.export_compilation(items, plans, self.tmp / "out", name, **kw)["compilation_dir"])

    def rescan(self, ws=None):
        ws = ws or self.dst
        ws.scan_projects(); ws.scan_datasets(); ws.scan_labels()

    def do_import(self, root, sel=None, ws=None, **kw):
        ws = ws or self.dst
        contents = comp.inspect_compilation(root)
        sel = sel or comp.default_compilation_selection(ws, contents)
        result = comp.import_compilation(ws, contents, sel, **kw)
        self.rescan(ws)
        return result, contents, sel


# ------------------------------------------------------------------- export


class ExportTests(CompCase):
    def test_layout_and_description(self):
        root = self.export()
        self.assertTrue((root / "compilation_info.json").is_file() and (root / "README.txt").is_file())
        info = json.loads((root / "compilation_info.json").read_text())
        self.assertEqual([d["name"] for d in info["datasets"]], ["d1", "vids", "Odd [Name]"])
        self.assertEqual(info["projects"], ["proj", "proj2"])
        d1 = info["datasets"][0]
        self.assertEqual((d1["folder"], d1["type"], d1["project"], d1["num_media"]), ("d1", "IMAGE_COLLECTION", "proj", 5))
        self.assertEqual((d1["pose_labels"], d1["num_labeled_frames"], d1["behavior_labels"]), (True, 3, True))
        odd = info["datasets"][2]
        self.assertEqual((odd["pose_labels"], odd["behavior_labels"], odd["folder"]), (True, False, "Odd [Name]"))

    def test_every_inner_folder_is_a_standard_export_that_imports_on_its_own(self):
        root = self.export()
        for name in ("d1", "vids", "Odd [Name]"):
            c = bi.inspect_bundle(root / name)
            self.assertTrue(c.dataset.ok and c.project.ok, name)
        # ...and it's byte-for-byte the same shape as a single-dataset export (full WHISKER compatibility)
        single = self.export_bundle(self.src, "d1", name="single")
        shape = lambda base: sorted(str(p.relative_to(base)).replace("\\", "/") for p in base.rglob("*") if p.is_file())
        self.assertEqual(shape(root / "d1"), shape(single))
        a = json.loads((root / "d1" / "export_info.json").read_text())
        b = json.loads((single / "export_info.json").read_text())
        for volatile in ("created_at",):
            a.pop(volatile), b.pop(volatile)
        self.assertEqual(a, b)

    def test_per_dataset_choices_are_honoured(self):
        items = [CompilationItem("d1", "proj", include_media=True, include_pose=False),
                 CompilationItem("vids", "proj", include_media=False)]
        root = self.export(items)
        self.assertFalse((root / "d1" / "pose_labels").exists())
        self.assertTrue((root / "d1" / "frames" / "frame_0000.png").exists())
        self.assertFalse((root / "vids" / "videos").exists())
        self.assertFalse(bi.inspect_bundle(root / "vids").media_included)
        info = json.loads((root / "compilation_info.json").read_text())
        self.assertEqual([(d["media_included"], d["pose_labels"]) for d in info["datasets"]], [(True, False), (False, False)])

    def test_bad_requests_are_refused_with_the_dataset_named(self):
        with self.assertRaises(fmt.BundleError):
            comp.build_plans(self.src, [])
        with self.assertRaises(fmt.BundleError):
            comp.build_plans(self.src, [CompilationItem("d1", "proj"), CompilationItem("d1", "proj")])
        with self.assertRaises(fmt.BundleError) as cm:
            comp.build_plans(self.src, [CompilationItem("nope", "proj")])
        self.assertIn("'nope'", str(cm.exception))
        with self.assertRaises(fmt.BundleError):
            comp.build_plans(self.src, [CompilationItem("d1", "no such project")])

    def test_names_that_cannot_be_folders_are_refused(self):
        items = self.items()
        plans = comp.build_plans(self.src, items)
        for bad in ("", "   ", "a/b", "what?", "trail.", "CON"):
            with self.assertRaises(fmt.BundleError, msg=bad):
                comp.export_compilation(items, plans, self.tmp / "out", bad)
        # surrounding spaces are simply trimmed
        r = comp.export_compilation(items, plans, self.tmp / "out", "  padded  ")
        self.assertEqual(Path(r["compilation_dir"]).name, "padded")

    def test_existing_destination_needs_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        (self.tmp / "out" / "pack" / "stale.txt").write_text("old")
        self.export(overwrite=True)
        self.assertFalse((self.tmp / "out" / "pack" / "stale.txt").exists())

    def test_cancelling_leaves_nothing_behind(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 4                       # let the first dataset start, then cancel
        items = self.items()
        plans = comp.build_plans(self.src, items)
        with self.assertRaises(fmt.BundleError):
            comp.export_compilation(items, plans, self.tmp / "out", "pack", cancel_cb=cancel)
        self.assertFalse((self.tmp / "out" / "pack").exists())

    def test_a_failure_part_way_removes_the_partial_folder(self):
        items = self.items()
        plans = comp.build_plans(self.src, items)
        real = fmt.export_annotation_bundle
        state = {"n": 0}

        def flaky(*a, **k):
            state["n"] += 1
            if state["n"] == 2:
                raise OSError("disk full")
            return real(*a, **k)
        with mock.patch.object(fmt, "export_annotation_bundle", flaky):
            with self.assertRaises(OSError):
                comp.export_compilation(items, plans, self.tmp / "out", "pack")
        self.assertFalse((self.tmp / "out" / "pack").exists())

    def test_progress_only_moves_forward_and_finishes(self):
        seen = []
        items = self.items()
        comp.export_compilation(items, comp.build_plans(self.src, items), self.tmp / "out", "pack",
                                progress_cb=lambda m, p: seen.append(p))
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 100)
        self.assertGreater(len(seen), 5)

    def test_folder_names_are_legal_and_unique(self):
        taken = set()
        got = [comp.safe_folder_name(n, taken) for n in ("a:b", "a?b", "A_B", "con", "trail. ", "")]
        self.assertEqual(got, ["a_b", "a_b_2", "A_B_3", "con_", "trail", "dataset"])

    def test_the_result_totals_what_was_copied(self):
        items = self.items()
        r = comp.export_compilation(items, comp.build_plans(self.src, items), self.tmp / "out", "pack")
        self.assertEqual((r["num_datasets"], r["num_media"], r["num_media_copied"], r["num_missing"]), (3, 10, 10, 0))


class GuessProjectTests(CompCase):
    def test_picks_the_project_that_fits_the_labels(self):
        self.assertEqual(comp.guess_project(self.src, "d1"), "proj")
        self.assertEqual(comp.guess_project(self.src, "Odd [Name]"), "proj2")     # snout/tail/rat1 only fit proj2

    def test_prefers_the_active_project_when_it_fits_or_nothing_is_labeled(self):
        self.add_dataset(self.src, "bare", ["x.png"], media_root=self.tmp / "b")
        self.assertEqual(comp.guess_project(self.src, "bare", preferred="proj2"), "proj2")
        self.assertEqual(comp.guess_project(self.src, "d1", preferred="proj2"), "proj")   # proj2 doesn't fit d1

    def test_no_projects(self):
        self.assertIsNone(comp.guess_project(self.dst, "anything"))


# ------------------------------------------------------- locating / inspecting


class LocateTests(CompCase):
    def test_a_compilation_is_found_from_any_pick(self):
        root = self.export()
        for pick in (root, root / "compilation_info.json", root / "README.txt"):
            loc = bi.locate_bundle(pick)
            self.assertEqual((loc.root, loc.kind, loc.is_compilation), (root, "compilation", True), pick)

    def test_a_folder_containing_a_compilation(self):
        root = self.export()
        loc = bi.locate_bundle(root.parent)
        self.assertEqual((loc.root, loc.kind), (root, "compilation"))
        self.assertIn("compilation", loc.message)

    def test_an_inner_export_picked_directly_stays_a_single_import(self):
        root = self.export()
        for pick in (root / "d1", root / "d1" / "frames", root / "d1" / "export_info.json"):
            loc = bi.locate_bundle(pick)
            self.assertEqual((loc.root, loc.kind), (root / "d1", "bundle"), pick)

    def test_the_inner_exports_are_not_listed_when_searching_a_parent(self):
        root = self.export()
        below, _ = bi._search_down(root.parent)
        self.assertEqual(below, [root])

    def test_several_compilations_are_offered_not_guessed(self):
        a = self.export(name="a")
        shutil.copytree(a, self.tmp / "out" / "b")
        loc = bi.locate_bundle(self.tmp / "out")
        self.assertIsNone(loc.root)
        self.assertEqual(sorted(p.name for p in loc.candidates), ["a", "b"])

    def test_something_inside_a_compilation_but_not_an_inner_export(self):
        root = self.export()
        (root / "notes").mkdir()
        loc = bi.locate_bundle(root / "notes")
        self.assertEqual((loc.root, loc.kind), (root, "compilation"))
        self.assertIn("inside the compilation", loc.message)


class InspectTests(CompCase):
    def test_entries_and_projects(self):
        c = comp.inspect_compilation(self.export())
        self.assertEqual([e.name for e in c.entries], ["d1", "vids", "Odd [Name]"])
        self.assertTrue(all(e.ok for e in c.entries))
        self.assertEqual(c.project_names, ["proj", "proj2"])
        self.assertEqual(c.name, "pack")
        self.assertTrue(c.entry("vids").contents.behavior.ok)

    def test_a_missing_or_damaged_inner_export_is_reported_and_the_rest_still_load(self):
        root = self.export()
        shutil.rmtree(root / "vids")
        (root / "Odd [Name]" / "export_info.json").write_text("{ nope")
        c = comp.inspect_compilation(root)
        by = {e.name: e for e in c.entries}
        self.assertTrue(by["d1"].ok)
        self.assertFalse(by["vids"].ok)
        self.assertIn("missing", by["vids"].problem)
        self.assertFalse(by["Odd [Name]"].ok)
        self.assertIn("damaged", by["Odd [Name]"].problem)

    def test_a_damaged_description_gives_a_clear_error(self):
        root = self.export()
        (root / "compilation_info.json").write_text("{ nope")
        with self.assertRaises(bi.BundleImportError) as cm:
            comp.inspect_compilation(root)
        self.assertIn("damaged", str(cm.exception))
        (root / "compilation_info.json").write_text('{"datasets": "x"}')
        with self.assertRaises(bi.BundleImportError):
            comp.inspect_compilation(root)

    def test_folders_not_listed_in_the_description_are_ignored(self):
        root = self.export()
        shutil.copytree(root / "d1", root / "stray")
        self.assertEqual([e.name for e in comp.inspect_compilation(root).entries], ["d1", "vids", "Odd [Name]"])

    def test_newer_format_warns(self):
        root = self.export()
        info = json.loads((root / "compilation_info.json").read_text())
        info["compilation_format_version"] = "9.0"
        (root / "compilation_info.json").write_text(json.dumps(info))
        self.assertTrue(any("newer version" in w for w in comp.inspect_compilation(root).warnings))


# --------------------------------------------------------------------- import


class ImportTests(CompCase):
    def test_everything_into_a_fresh_workspace(self):
        root = self.export()
        result, contents, sel = self.do_import(root)
        self.assertEqual(sorted(self.dst.datasets.keys()), ["Odd [Name]", "d1", "vids"])
        self.assertEqual(sorted(self.dst.projects.keys()), ["proj", "proj2"])
        self.assertEqual(sorted(p["name"] for p in result["projects"]), ["proj", "proj2"])   # each once, though proj is shared
        self.assertTrue(all(p["installed"] for p in result["projects"]))
        self.assertTrue(all(d["error"] is None for d in result["datasets"]))
        self.assertTrue(self.dst.pose_labels.has_pose_labels("d1"))
        self.assertTrue(self.dst.behavior_labels.has_behavior_labels("vids"))
        self.assertTrue(self.dst.pose_labels.has_pose_labels("Odd [Name]"))
        self.assertFalse(result["cancelled"])

    def test_only_what_is_ticked(self):
        root = self.export()
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.import_projects = False
        sel.items["d1"].pose = False
        sel.items["d1"].behavior = False
        sel.items["vids"].dataset = False
        sel.items["vids"].pose = sel.items["vids"].behavior = False
        sel.items["Odd [Name]"].dataset = sel.items["Odd [Name]"].pose = False
        result, *_ = self.do_import(root, sel)
        self.assertEqual(self.dst.datasets.keys(), ["d1"])
        self.assertFalse(self.dst.pose_labels.has_pose_labels("d1"))
        self.assertEqual(list(self.dst.projects.keys()), [])
        self.assertEqual([d["name"] for d in result["datasets"]], ["d1"])

    def test_a_dataset_can_be_imported_under_a_new_name(self):
        root = self.export()
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.items["d1"].dataset_name = "renamed"
        self.do_import(root, sel)
        self.assertIn("renamed", self.dst.datasets.keys())
        meta = json.loads((self.dst.pose_labels.base_dir / "renamed" / "metadata.json").read_text())
        self.assertEqual(meta["dataset_name"], "renamed")

    def test_reimport_defaults_to_using_what_you_already_have(self):
        root = self.export()
        self.do_import(root)
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        self.assertEqual(sel.project_choices, {"proj": ("existing", "proj"), "proj2": ("existing", "proj2")})
        self.assertFalse(sel.adds_projects(contents))                    # nothing new to add
        for name, item in sel.items.items():
            self.assertFalse(item.dataset, name)
            self.assertEqual(item.target_dataset, name)

    def test_a_fresh_workspace_defaults_to_adding_every_project(self):
        contents = comp.inspect_compilation(self.export())
        sel = comp.default_compilation_selection(self.dst, contents)
        self.assertEqual(sel.project_choices, {"proj": ("new", "proj"), "proj2": ("new", "proj2")})
        self.assertTrue(sel.adds_projects(contents))

    def test_a_project_can_be_added_under_a_new_name(self):
        root = self.export()
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.project_choices["proj"] = ("new", "my_copy")
        result, *_ = self.do_import(root, sel)
        self.assertEqual(sorted(self.dst.projects.keys()), ["my_copy", "proj2"])
        self.assertEqual(self.dst.projects.get("my_copy").name, "my_copy")
        by = {p["name"]: p for p in result["projects"]}
        self.assertEqual((by["proj"]["installed"], by["proj"]["as"]), (True, "my_copy"))

    def test_a_project_can_be_mapped_onto_one_of_yours(self):
        root = self.export()
        self.add_project(self.dst, "mine", body_parts=["nose", "tail_base"], identities=["mouse1"], behaviors=["groom", "rear"])
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.project_choices["proj"] = ("existing", "mine")
        result, *_ = self.do_import(root, sel)
        self.assertEqual(sorted(self.dst.projects.keys()), ["mine", "proj2"])       # 'proj' itself was not added
        by = {p["name"]: p for p in result["projects"]}
        self.assertEqual((by["proj"]["installed"], by["proj"]["existing"]), (False, "mine"))
        self.assertTrue(by["proj2"]["installed"])

    def test_project_choices_are_validated(self):
        contents = comp.inspect_compilation(self.export())
        self.add_project(self.dst, "taken")
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.project_choices["proj"] = ("new", "taken")
        self.assertTrue(any("already have a project called 'taken'" in p
                            for p in comp.validate_compilation_selection(self.dst, contents, sel)))
        sel.project_choices["proj"] = ("new", "bad/name")
        self.assertTrue(any("new name can't" in p for p in comp.validate_compilation_selection(self.dst, contents, sel)))
        sel.project_choices["proj"] = ("existing", "ghost")
        self.assertTrue(any("choose which of your projects" in p
                            for p in comp.validate_compilation_selection(self.dst, contents, sel)))
        sel.project_choices["proj"] = ("new", "same")
        sel.project_choices["proj2"] = ("new", "same")
        self.assertTrue(any("would both be saved as 'same'" in p
                            for p in comp.validate_compilation_selection(self.dst, contents, sel)))

    def test_labels_merge_into_datasets_that_already_have_labels(self):
        root = self.export()
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "dm")
        self.add_pose_labels(self.dst, "d1", self.frames[2:5], x0=50.0)         # mine: frames 2-4 at x0=50
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        for n in ("vids", "Odd [Name]"):
            sel.items[n].dataset = sel.items[n].pose = sel.items[n].behavior = False
        sel.items["d1"].behavior = False
        sel.existing_labels_policy = LabelPolicy.MERGE_EXISTING
        result, *_ = self.do_import(root, sel)
        merged = PoseDataset.from_file(self.dst.pose_labels.base_dir / "d1" / "labels.h5")
        x = lambda f: float(merged.keypoint_data.xs((f, "mouse1", "nose"))["x"])
        self.assertEqual(sorted(merged.frame_indices), self.frames)             # 0,1 added + my 2,3,4
        self.assertEqual(x("frame_0002.png"), 50.0)                              # mine kept where both had one
        self.assertEqual(x("frame_0000.png"), 10.0)
        self.assertEqual(result["datasets"][0]["result"]["pose"]["mode"], "merged")

    def test_labels_that_match_nothing_are_skipped_with_a_note_and_the_rest_carry_on(self):
        root = self.export()
        self.add_dataset(self.dst, "d1", ["zzz_1.png", "zzz_2.png"], media_root=self.tmp / "zz")   # same name, different files
        sel = comp.default_compilation_selection(self.dst, comp.inspect_compilation(root))
        self.assertFalse(sel.items["d1"].dataset)
        result, *_ = self.do_import(root, sel)
        by = {d["name"]: d for d in result["datasets"]}
        self.assertIsNone(by["d1"]["error"])
        self.assertTrue(any("skipped: none matched" in n for n in by["d1"]["result"]["notes"]))
        self.assertFalse(self.dst.pose_labels.has_pose_labels("d1"))
        self.assertIn("vids", self.dst.datasets.keys())                          # others imported fully

    def test_labels_that_cannot_be_combined_leave_the_existing_ones_alone(self):
        root = self.export()
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "dm")
        self.add_pose_labels(self.dst, "d1", self.frames[3:5], body_parts=["nose", "ear"])
        before = (self.dst.pose_labels.base_dir / "d1" / "labels.h5").read_bytes()
        sel = comp.default_compilation_selection(self.dst, comp.inspect_compilation(root))
        for n in ("vids", "Odd [Name]"):
            sel.items[n].dataset = sel.items[n].pose = sel.items[n].behavior = False
        result, *_ = self.do_import(root, sel)
        note = " ".join(result["datasets"][0]["result"]["notes"])
        self.assertIn("not combined", note)
        self.assertIn("left unchanged", note)
        self.assertEqual((self.dst.pose_labels.base_dir / "d1" / "labels.h5").read_bytes(), before)

    def test_replace_policy_overrides_an_incompatible_merge(self):
        root = self.export()
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "dm")
        self.add_pose_labels(self.dst, "d1", self.frames[3:5], body_parts=["nose", "ear"])
        sel = comp.default_compilation_selection(self.dst, comp.inspect_compilation(root))
        for n in ("vids", "Odd [Name]"):
            sel.items[n].dataset = sel.items[n].pose = sel.items[n].behavior = False
        sel.items["d1"].behavior = False
        sel.existing_labels_policy = LabelPolicy.REPLACE
        self.do_import(root, sel)
        self.assertEqual(sorted(PoseDataset.from_file(self.dst.pose_labels.base_dir / "d1" / "labels.h5").body_parts),
                         ["nose", "tail_base"])

    def test_one_dataset_failing_does_not_stop_the_others(self):
        root = self.export()
        real = bi.import_from_bundle

        def flaky(ws, contents, sel, *a, **k):
            if contents.dataset_name == "vids":
                raise OSError("disk full")
            return real(ws, contents, sel, *a, **k)
        with mock.patch.object(bi, "import_from_bundle", flaky):
            result, *_ = self.do_import(root)
        by = {d["name"]: d for d in result["datasets"]}
        self.assertEqual(by["vids"]["error"], "disk full")
        self.assertIsNone(by["d1"]["error"])
        self.assertIsNone(by["Odd [Name]"]["error"])
        self.assertEqual(sorted(self.dst.datasets.keys()), ["Odd [Name]", "d1"])

    def test_cancelling_stops_after_the_current_dataset(self):
        root = self.export()
        seen = {"n": 0}

        def cancel():
            seen["n"] += 1
            return seen["n"] > 3
        result, *_ = self.do_import(root, cancel_cb=cancel)
        self.assertTrue(result["cancelled"])
        self.assertLess(len(self.dst.datasets.keys()), 3)

    def test_no_half_imported_dataset_is_left_when_cancelled_mid_copy(self):
        root = self.export()
        c = bi.inspect_bundle(root / "d1")
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 2
        with self.assertRaises(mi.ManualImportError):
            mi.install_dataset(self.dst, c.dataset_obj, "d1", c.media_dir, cancel_cb=cancel)
        self.assertNotIn("d1", [p.name for p in self.dst.datasets.base_dir.iterdir()])

    def test_projects_are_installed_once_and_existing_ones_kept(self):
        root = self.export()
        self.add_project(self.dst, "proj", behaviors=["something else"])       # mine differs from the export's
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.import_projects = True
        result, *_ = self.do_import(root, sel)
        by = {p["name"]: p["installed"] for p in result["projects"]}
        self.assertEqual(by, {"proj": False, "proj2": True})
        self.assertEqual(self.dst.projects.get("proj").behaviors, ["something else"])

    def test_the_inner_exports_still_import_on_their_own(self):
        root = self.export()
        r = bi.import_from_bundle(self.dst, bi.inspect_bundle(root / "d1"),
                                  ImportSelection(project=True, dataset=True, pose=True, behavior=True, dataset_name="d1"))
        self.assertEqual(r["num_media_copied"], 5)


class ValidationTests(CompCase):
    def setUp(self):
        super().setUp()
        self.root = self.export()
        self.contents = comp.inspect_compilation(self.root)

    def test_default_selection_is_valid(self):
        sel = comp.default_compilation_selection(self.dst, self.contents)
        self.assertEqual(comp.validate_compilation_selection(self.dst, self.contents, sel), [])
        self.assertTrue(sel.import_projects)

    def test_nothing_ticked(self):
        sel = comp.CompilationSelection(import_projects=False)
        self.assertEqual(comp.validate_compilation_selection(self.dst, self.contents, sel), ["Tick at least one thing to import."])

    def test_projects_alone_is_enough(self):
        sel = comp.CompilationSelection(import_projects=True)
        self.assertEqual(comp.validate_compilation_selection(self.dst, self.contents, sel), [])

    def test_two_datasets_cannot_take_the_same_name(self):
        sel = comp.default_compilation_selection(self.dst, self.contents)
        sel.items["vids"].dataset_name = "d1"
        problems = comp.validate_compilation_selection(self.dst, self.contents, sel)
        self.assertTrue(any("both be imported as 'd1'" in p for p in problems))

    def test_a_clash_with_an_existing_dataset_names_the_dataset(self):
        self.add_dataset(self.dst, "d1", self.frames, media_root=self.tmp / "dm")
        sel = comp.default_compilation_selection(self.dst, self.contents)
        sel.items["d1"].dataset = True
        self.assertEqual(sel.items["d1"].dataset_name, "d1_2")           # a free name is suggested
        self.assertEqual(comp.validate_compilation_selection(self.dst, self.contents, sel), [])
        sel.items["d1"].dataset_name = "d1"                                # ...but choosing the taken one is refused
        problems = comp.validate_compilation_selection(self.dst, self.contents, sel)
        self.assertTrue(any(p.startswith("'d1':") and "already have a dataset" in p for p in problems))

    def test_labels_for_a_dataset_you_dont_have_need_its_frames_too(self):
        sel = comp.default_compilation_selection(self.dst, self.contents)
        sel.items["d1"].dataset = False                       # labels ticked, no such dataset here, no target
        problems = comp.validate_compilation_selection(self.dst, self.contents, sel)
        self.assertTrue(any(p.startswith("'d1':") and "Choose which dataset" in p for p in problems))

    def test_a_broken_entry_cannot_be_ticked(self):
        shutil.rmtree(self.root / "vids")
        contents = comp.inspect_compilation(self.root)
        sel = comp.default_compilation_selection(self.dst, contents)
        sel.items["vids"].dataset = True
        problems = comp.validate_compilation_selection(self.dst, contents, sel)
        self.assertTrue(any(p.startswith("'vids':") and "missing" in p for p in problems))
        with self.assertRaises(bi.BundleImportError):
            comp.import_compilation(self.dst, contents, sel)


class AttachNotesTests(CompCase):
    def test_notes_describe_the_match(self):
        root = self.export()
        self.add_dataset(self.dst, "d1", self.frames[1:], media_root=self.tmp / "dm")     # lacks frame_0000
        self.add_pose_labels(self.dst, "d1", self.frames[2:4])
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        notes = comp.attach_notes(self.dst, contents.entry("d1"), sel.items["d1"])
        text = " | ".join(t for _l, t in notes)
        self.assertIn("Pose: 2/3 frames match (1 skipped)", text)
        self.assertIn("already has labels", text)
        self.assertIn("Behavior: none of 2 videos match", text)
        self.assertIn("bad", [lvl for lvl, _t in notes])

    def test_nothing_to_say_when_the_dataset_is_being_imported(self):
        root = self.export()
        contents = comp.inspect_compilation(root)
        sel = comp.default_compilation_selection(self.dst, contents)
        self.assertEqual(comp.attach_notes(self.dst, contents.entry("d1"), sel.items["d1"]), [])
