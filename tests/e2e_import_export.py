"""End-to-end check of import/export in the REAL main window (not run by `unittest discover`).

    python tests/e2e_import_export.py

Drives File > Import (full import, then a labels-only re-import that merges), File > Export, and a
multi-dataset compilation exported from one workspace and imported into another,
by clicking through the actual dialogs, with the real worker thread and refresh.

SAFETY: the app normally opens the workspace saved in your user settings (your real one), and
QSettings("org", "app") ignores setDefaultFormat. So QSettings is replaced, before any whisker
import, with a subclass that always writes to a temp .ini; the script then REFUSES to do anything
unless the window is provably on temp settings and a temp workspace, and at the end it confirms
your real workspace is unchanged. Exit code 0 = every check passed.
"""
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
REPO = Path(__file__).resolve().parents[1]
REAL_WORKSPACE = REPO      # where a developer's saved workspace usually lives
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.excepthook = lambda t, v, tb: print("!! UNHANDLED:", "".join(traceback.format_exception(t, v, tb)), flush=True)

tmp = Path(tempfile.mkdtemp(prefix="whisker_e2e_")).resolve()

import PyQt6.QtCore as QtCore
_RealQSettings = QtCore.QSettings


class IsolatedSettings(_RealQSettings):
    def __init__(self, org, app=None, *args, **kwargs):
        (tmp / "settings").mkdir(exist_ok=True)
        super().__init__(str(tmp / "settings" / f"{org}-{app}.ini"), _RealQSettings.Format.IniFormat)


QtCore.QSettings = IsolatedSettings          # every `from PyQt6.QtCore import QSettings` now gets this

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox, QRadioButton


def snapshot():
    out = {}
    for sub in ("datasets", "projects", "workflows"):
        base = REAL_WORKSPACE / sub
        out[sub] = sorted(str(p.relative_to(base)) for p in base.rglob("*")) if base.exists() else []
    return out


REAL_BEFORE = snapshot()

from fixtures import WorkspaceCase, frame_names, make_pose_dataset
from whisker.core import bundle_import as bi


class Build(WorkspaceCase):
    def runTest(self):
        self.tmp = tmp / "collab"
        self.tmp.mkdir()
        src = self.new_workspace("src")
        self.add_project(src)                      # the export carries project "proj"
        self.files = frame_names(6)
        self.add_dataset(src, "openfield", self.files)
        self.add_pose_labels(src, "openfield", self.files[:4])
        self.add_behavior_labels(src, "openfield", ["a.mp4", "b.mp4"])
        self.bundle = self.export_bundle(src, "openfield")


b = Build()
b.runTest()
bundle, files = b.bundle, b.files

# ---- the app's workspace: a temp folder that already has its own active project -------------
ws_dir = tmp / "my_workspace"
ws_dir.mkdir()
seed = Build()
seed.tmp = tmp / "seed"
seed.tmp.mkdir()
seed_ws = seed.new_workspace("x")
seed_ws.create_project("myproject", body_parts=["nose"], identities=["m1"], behaviors=["walk"])
import shutil
for f in (seed_ws.base_dir / "projects").glob("*.json"):
    (ws_dir / "projects").mkdir(exist_ok=True)
    shutil.copy2(f, ws_dir / "projects" / f.name)
seed._release_log_handlers()

seed_settings = IsolatedSettings("whisker", "main")
seed_settings.setValue("active_project", "myproject")
seed_settings.sync()

logging.disable(logging.NOTSET)
os.chdir(ws_dir)
from whisker.gui.application import Application

gui = Application(["whisker"])
mw = gui.window
app = QApplication.instance()

# ---- HARD GUARDS: refuse to act on anything real ----------------------------------------------
ws = mw._workspace
guards_ok = (
    tmp in Path(mw.settings.fileName()).resolve().parents
    and Path(ws.base_dir).resolve() == ws_dir.resolve()
    and Path(ws.base_dir).resolve() != REAL_WORKSPACE.resolve()
)
print("GUARD settings file :", mw.settings.fileName())
print("GUARD workspace     :", ws.base_dir)
if not guards_ok:
    print("!! ISOLATION FAILED - refusing to run. Nothing was touched.")
    os._exit(3)
print("GUARD ok - the app is on temp settings and a temp workspace\n")

messages = []
for kind in ("information", "warning", "critical"):
    setattr(QMessageBox, kind, staticmethod(lambda parent, title, text, *a, _k=kind, **k: messages.append((_k, title, text))))


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()


def wait_for(pred, timeout=30, what=""):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.02)
    print(f"!! TIMEOUT waiting for {what}")
    return False


modal = QApplication.activeModalWidget
results = []


def check(cond, label):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + label)


pump(0.5)
file_menu = next(a.menu() for a in mw.menuBar().actions() if a.text().replace("&", "") == "File")
texts = [a.text() for a in file_menu.actions() if a.text()]
check("Import..." in texts and "Import Labels from Other Software..." in texts, "File menu has the new entries")
check(mw._import_bundle_action.isEnabled(), "Import... enabled with a workspace open")
import_key = mw._import_bundle_action.shortcut().toString()
check(import_key == "Ctrl+Shift+O", "Import... shortcut is Ctrl+Shift+O")
# Ctrl+I is "Swap identities" on the pose labeling screen; a window-level shortcut would hijack it.
others = [a.shortcut().toString() for a in mw.findChildren(type(mw._import_bundle_action)) if a is not mw._import_bundle_action]
check(import_key not in others and import_key != "Ctrl+I", "Import shortcut collides with no other action and not with Ctrl+I")
check(mw._export_labels_action.text() == "Export Dataset / Labels...", "Export entry renamed")
check(not mw._export_labels_action.isEnabled(), "Export greyed out while the workspace has no datasets")
check(mw._active_project_name == "myproject", "starting active project is 'myproject'")


def active_saved():
    return mw.settings.value("active_project", "")


# ---- 1. FULL import (adds project 'proj') -----------------------------------------------------
def drive_full():
    d = modal()
    d.set_path(str(bundle))
    print("   ticks:", {k: c.isChecked() for k, (c, _n) in d._rows.items()})
    d.import_btn.click()


QTimer.singleShot(400, drive_full)
mw._import_bundle_action.trigger()
wait_for(lambda: any(m[0] == "information" for m in messages), what="import-complete message")
print("   message shown:\n      " + [m for m in messages if m[0] == "information"][-1][2].replace("\n", "\n      "))
pump(0.5)
check(ws.datasets.get("openfield") is not None and len(ws.datasets.get("openfield").files) == 6, "dataset imported (6 frames)")
check(ws.pose_labels.has_pose_labels("openfield") and ws.behavior_labels.has_behavior_labels("openfield"), "labels imported")
check(ws.projects.get("proj") is not None, "project 'proj' imported")
check(mw._export_labels_action.isEnabled(), "Export enabled once a dataset exists")
check(mw._active_project_name == "myproject", f"ACTIVE PROJECT preserved in the window (is {mw._active_project_name!r})")
check(active_saved() == "myproject", f"ACTIVE PROJECT preserved in saved settings (is {active_saved()!r})")
check(not [m for m in messages if m[0] in ("warning", "critical")], "no warnings or errors shown")

# ---- 2. LABELS-ONLY re-import: follow-up popup, merge ----------------------------------------
mine = make_pose_dataset(files[3:6], x0=50.0)
ws.pose_labels.write_poses_file("openfield", mine, ws.pose_labels.base_dir / "openfield" / "labels.h5")
ws.scan_labels()
messages.clear()


def attach_step():
    a = modal()
    print("   follow-up popup:", a.windowTitle())
    next(r for r in a.findChildren(QRadioButton) if "use the imported one" in r.text()).setChecked(True)
    a.ok_btn.click()


def drive_labels_only():
    d = modal()
    d.set_path(str(bundle))
    print("   defaults:", {k: c.isChecked() for k, (c, _n) in d._rows.items()}, "| button:", d.import_btn.text())
    QTimer.singleShot(300, attach_step)
    d.import_btn.click()


QTimer.singleShot(400, drive_labels_only)
mw._import_bundle_action.trigger()
wait_for(lambda: any(m[0] == "information" for m in messages), what="merge-complete message")
print("   message shown:\n      " + [m for m in messages if m[0] == "information"][-1][2].replace("\n", "\n      "))
pump(0.5)
from whisker.services.pose_estimation.public.data_structures import PoseDataset
merged = PoseDataset.from_file(ws.pose_labels.base_dir / "openfield" / "labels.h5")
x = lambda f: float(merged.keypoint_data.xs((f, "mouse1", "nose"))["x"])
check(len(merged.frame_indices) == 6, "merged labels cover all 6 frames")
check(x("frame_0003.png") == 10.0, "overlap frame took the IMPORTED value (as chosen)")
check(x("frame_0005.png") == 50.0, "frame only I had is untouched")
check(ws.pose_labels.get_pose_labeled_image_keys_from_summary("openfield") == set(files), "labeled-frame list is current")
check(mw._active_project_name == "myproject" and active_saved() == "myproject", "active project still preserved")
check(not [m for m in messages if m[0] in ("warning", "critical")], "no warnings or errors shown")

# ---- 3. EXPORT with nothing selected ---------------------------------------------------------
messages.clear()
out_parent = tmp / "outbox"
out_parent.mkdir()


def drive_export():
    e = modal()
    print("   dialog:", type(e).__name__, "-", e.windowTitle())
    e.include_media_checkbox.setChecked(False)
    e.dest_edit.setText(str(out_parent))
    print("   suggested name:", e.name_edit.text())
    e.button_box.button(e.button_box.StandardButton.Ok).click()


QTimer.singleShot(400, drive_export)
mw._export_annotations()
wait_for(lambda: any(m[0] == "information" for m in messages), what="export-complete message")
print("   message shown:\n      " + [m for m in messages if m[0] == "information"][-1][2].replace("\n", "\n      "))
out = out_parent / "openfield_labels"
check((out / "export_info.json").exists() and not (out / "frames").exists(), "labels-only export written (no frames folder)")
check(bi.inspect_bundle(out).pose.ok, "...and it re-imports cleanly")

# ---- 4. COMPILATION: export several datasets from this workspace ...
check(mw._export_compilation_action.isEnabled(), "Export Several Datasets... is enabled")
check(mw._export_compilation_action.text() == "Export Several Datasets...", "Export Several Datasets... is in the menu")
extra_files = frame_names(4)
b.add_dataset(ws, "extra", extra_files)
b.add_pose_labels(ws, "extra", extra_files[:3])
ws.scan_datasets(); ws.scan_labels()
messages.clear()
share_parent = tmp / "shared"
share_parent.mkdir()


def drive_comp_export():
    e = modal()
    print("   dialog:", type(e).__name__, "| datasets listed:", [r["name"] for r in e._rows])
    e._tick_all(True)
    e.dest_edit.setText(str(share_parent))
    e.name_edit.setText("labshare")
    e.ok_btn.click()


QTimer.singleShot(400, drive_comp_export)
mw._export_compilation_action.trigger()
wait_for(lambda: any(m[0] == "information" for m in messages), what="compilation-export message")
print("   message shown:\n      " + [m for m in messages if m[0] == "information"][-1][2].replace("\n", "\n      "))
shared = share_parent / "labshare"
from whisker.core import compilation as comp
check((shared / "compilation_info.json").exists() and (shared / "openfield" / "export_info.json").exists()
      and (shared / "extra" / "export_info.json").exists(), "compilation folder written with one export per dataset")
check(bi.inspect_bundle(shared / "extra").pose.ok, "each inner folder is a normal export that imports on its own")

# ---- 5. ... and import it into a second, empty workspace (real File > Import path)
ws2_dir = tmp / "second_workspace"
ws2_dir.mkdir()
mw.set_workspace(ws2_dir)
pump(0.3)
ws2 = mw._workspace
check(Path(ws2.base_dir).resolve() == ws2_dir.resolve() and tmp in Path(ws2.base_dir).resolve().parents,
      "app switched to the second temp workspace (guard)")
if not (Path(ws2.base_dir).resolve() == ws2_dir.resolve()):
    print("!! wrong workspace, aborting")
    os._exit(4)
messages.clear()


def choose_datasets_step():
    c = modal()
    print("   compilation dialog:", type(c).__name__, "| rows:", [r["entry"].name for r in c._rows],
          "| import enabled:", c.import_btn.isEnabled())
    c.import_btn.click()


def pick_compilation_step():
    d = modal()
    d.set_path(str(shared))
    print("   import dialog says:", d.location_label.text().split(" \u2014 ")[0], "| button:", d.import_btn.text())
    QTimer.singleShot(500, choose_datasets_step)
    d.import_btn.click()


QTimer.singleShot(400, pick_compilation_step)
mw._import_bundle_action.trigger()
wait_for(lambda: any(m[0] == "information" for m in messages), what="compilation-import message")
text = [m for m in messages if m[0] == "information"][-1][2]
print("   message shown:\n      " + text.replace("\n", "\n      "))
pump(0.5)
check(sorted(ws2.datasets.keys()) == ["extra", "openfield"], "both datasets arrived in the second workspace")
check(sorted(ws2.projects.keys()) == ["myproject", "proj"] or "proj" in ws2.projects.keys(), "their project(s) arrived")
check(ws2.pose_labels.has_pose_labels("extra") and ws2.pose_labels.has_pose_labels("openfield"), "pose labels arrived for both")
check("2 of 2 dataset(s) imported" in text, "message reports 2 of 2 datasets")
check(not [m for m in messages if m[0] in ("warning", "critical")], "no warnings or errors shown")

# ---- the real workspace must be exactly as it was ---------------------------------------------
check(snapshot() == REAL_BEFORE, "REAL workspace untouched (datasets/projects/workflows identical)")
print(f"\n{sum(results)}/{len(results)} checks passed")
os._exit(0 if all(results) else 1)
