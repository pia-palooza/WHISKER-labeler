# WHISKER Labeler

A **hand-annotation** build of the full WHISKER application. It reuses WHISKER's
interface, workspace/project structure, and on-disk data format, but with **all
model training and prediction removed** — so it runs anywhere with no GPU and no
deep-learning frameworks. Use it to:

- place **pose keypoints** on images / extracted frames (the **Pose Estimation** workflow)
- mark **behavior bouts** on videos (the **Behavior Classification** workflow)

Labels are saved in the **exact HDF5 layout the full WHISKER app uses**, so your
annotations move straight into WHISKER for model training.

> **Hand annotation only.** The labeler does not train models or generate
> predictions — those parts of WHISKER are intentionally trimmed out.

---

## What's in the app

It's the same shell as full WHISKER, so if you've used WHISKER it will feel
identical:

- A left **Navigation** panel where you pick a **Workflow** — *Pose Estimation*
  or *Behavior Classification* — and a **Task**.
- A **Data Explorer** listing every dataset and its files, with a ✓ on files
  you've already labeled.
- A menu bar (**File / Edit / Selection / View / Tools / Help**) and a
  collapsible console at the bottom.

**Tasks available:** **Welcome**, **Projects**, **Info**, **Jobs**, and
**Label**. The training, prediction, evaluation, and figure-maker tasks from full
WHISKER are removed in this build.

**Projects** define the labeling targets — body parts, identities (animals),
behaviors, and an optional skeleton. A **dataset** is a folder of media tied to a
project. Your source media is only ever read, never modified.

---

## Requirements

- **Miniconda or Anaconda** — <https://docs.conda.io/en/latest/miniconda.html>
- **Git** — to clone the repository (or download it as a ZIP from GitHub via
  **Code → Download ZIP** and unzip it instead)
- **Python 3.11** (installed automatically by the conda environment below)
- Windows, macOS, or Linux

---

## Installation (first time)

```bash
# 1. Get the code
git clone https://github.com/pia-palooza/WHISKER-labeler.git
cd WHISKER-labeler

# 2. Create and activate the environment
conda env create -f environment.yaml
conda activate whisker-labeler

# 3. Install the package (editable)
pip install -e .
```

Step 3 installs the `whisker` package and its dependencies so the app can be run
from any folder.

---

## Launching

**Any platform (recommended)** — from an Anaconda Prompt / terminal:

```bash
conda activate whisker-labeler
cd path/to/WHISKER-labeler
python -m whisker.main
```

Running it from the `WHISKER-labeler` folder makes that folder the default
workspace on first launch (you can always switch later).

**Windows shortcut:** double-click **`launch.bat`** in the `WHISKER-labeler`
folder — it runs the same command for you. Note that `launch.bat` hard-codes the
path to the environment's `python.exe`; if your conda isn't in the default
Miniconda location, edit the `ENV_PY` line near the top of `launch.bat` first (or
just use the Anaconda Prompt method above).

**Desktop icon (Windows / macOS)** — once the app is installed you can add a
WHISKER icon so it opens with a double-click, no terminal or `launch.bat` needed.
From inside the app choose **Tools → Install Desktop Shortcut…**, pick your
computer type, and choose where to put it — or run this from a terminal:

```bash
conda activate whisker-labeler
whisker-labeler --install-shortcut            # Desktop + Start Menu / Applications
whisker-labeler --install-shortcut --shortcut-location desktop   # or: menu
```

- **Windows:** creates a **WHISKER Labeler** shortcut on the Desktop and in the Start
  Menu. To pin it to the taskbar, right-click the Start Menu entry → *Pin to taskbar*.
- **macOS:** creates **WHISKER Labeler.app** in `~/Applications` (so it shows up in
  Launchpad and Spotlight) plus a link on the Desktop. Drag it to the Dock to keep it there.
- The icon opens the app with no console window. If it ever fails to start, the reason is
  written to `whisker-startup.log` (Windows: `%LOCALAPPDATA%\WHISKER\`; macOS:
  `~/Library/Application Support/WHISKER/`).
- The shortcut points at the Python environment you installed it from, so run
  *Install Desktop Shortcut* again if you move or rebuild that environment. It can only
  be created for the OS you are on.
- Until you choose a workspace, the icon starts in a `WHISKER Workspace` folder in your
  home directory rather than the Desktop; use **File → Open Workspace…** to switch (the
  app remembers your choice).

On first launch the app opens a WHISKER **workspace** — by default the current
folder, or the last workspace you used. Use **File → Open Workspace…** to point it
at a different workspace at any time; recently used workspaces are remembered
under **File → Recent Workspaces**.

---

## Workspaces & where your data lives

The labeler reads and writes a standard WHISKER **workspace** folder. The layout
is identical to full WHISKER, so you can point the labeler straight at a WHISKER
workspace (or hand one back):

```
<workspace>/
  projects/<name>.json                                          # label definitions (body parts, identities, behaviors, skeleton)
  datasets/<name>/manifest.json                                 # which media files belong to a dataset
  workflows/pose_estimation/labels/<dataset>/labels.h5          # pose labels
  workflows/behavior_classification/labels/<dataset>/labels.h5  # behavior labels
```

Workspace data folders (`workspace/`, `datasets/`, `projects/`, `workflows/`) are
excluded from version control, so your annotations stay local and private. Back
them up if you want to keep copies.

---

## Typical workflow

1. **Open or create a workspace** — File → Open Workspace…, or just use the
   default folder.
2. **Create a project** — File → New Project… — and define its **Body Parts**,
   **Identities**, **Behaviors**, and optional **Skeleton** (comma-separated,
   e.g. `nose, left_ear, right_ear, tail_base`). Or import an existing project
   (below).
3. **Add a dataset** — File → New Dataset… — pointing at a folder of **images /
   extracted frames** (→ Pose) or **videos** (→ Behavior).
4. In the **Navigation** panel pick the **Workflow** (Pose or Behavior) and the
   **Label** task.
5. Pick the active **project**, then click a file in the **Data Explorer** and
   annotate. Move between files with the **← / →** keys or the Data Explorer;
   labeled files show a ✓.
6. **Save** (`Ctrl+S`). Labels are written to the workspace's `labels.h5` files
   shown above.

### Pose Estimation (keypoints on images)

Select a body part for an identity in the side panel, then click on the image to
place it; drag to adjust.

| Key | Action |
| --- | --- |
| `Q` | Toggle drag / move mode |
| `W` / `S` | Previous / next body part |
| `A` / `D` | Previous / next identity |
| `Delete` / `X` | Clear the selected keypoint |
| `Ctrl+Delete` | Clear all keypoints on this image |
| `Ctrl+I` | Swap identities |
| `Ctrl+N` | Toggle body-part name labels |
| `Ctrl+S` | Save |

Move between images with **← / →** or the Data Explorer.

### Behavior Classification (bouts on videos)

Play the video, mark a behavior's start and end frames, choose the behavior, and
create the bout. Bouts appear in the table and on the timeline strip.

| Key | Action |
| --- | --- |
| `Space` | Play / pause |
| `←` / `→` | Step one frame back / forward |
| `Shift`+`←` / `→` | Skip back / forward |
| `T` | Set **start** frame to current |
| `E` | Set **end** frame to current |
| `C` | Create / update the bout |
| `Esc` | Clear the editor (start a new bout) |
| `Delete` | Remove the selected bout |
| `O` | Toggle the on-video behavior overlay |
| `Ctrl+S` | Save |

Move between videos with the Data Explorer.

---

## Moving data to / from full WHISKER

Because the labeler shares WHISKER's workspace layout, the simplest path is to
point it directly at a full WHISKER workspace — no conversion needed. To hand a
dataset to someone else (or move it between workspaces), use Export and Import:

### Export — File → Export → Export Dataset / Labels…

Choose the dataset (you don't need to select it first) and tick what to include: the
**project**, the **videos/frames**, the **pose labels**, the **behavior labels**. **Everything is
ticked by default, including the videos/frames**, so the package is complete and self-contained.
Untick the videos/frames only for a small labels-only export that's easy to email; whoever
imports it is then asked to find the files themselves. The result is one folder with a
`README.txt` describing it.

An export is only ever left behind if it is complete:

- **Before anything is copied**, every file is checked to exist and the destination is checked
  to have room. A missing file stops the export, naming some of them, instead of producing a
  package that can't be imported. (An existing package is not replaced if this fails.)
- Each copied file is checked to be the same size as the original.
- If anything fails or you cancel, the partly written folder is removed.
- **Afterwards the finished package is read back the way the importer will read it**, and the
  completion message tells you it was checked, or lists exactly what wouldn't import.

### Import — File → Import… (Ctrl+Shift+O)

Choose the export folder — or drag it onto the window. You can pick the folder itself, a
folder inside it, or a folder that contains it (unzipping adds a level); the app finds it and
explains exactly what it looked at if it can't. It then lists what the export contains, and
you tick what you want: **project**, **videos/frames**, **pose labels**, **behavior labels**.

- For the **project** and the **dataset** you choose **add as a new one** (under a name you
  can edit) **or use one you already have** (pick it from your list). If you already have a
  project or dataset with the same name, "use mine" is the starting choice, so a re-import
  defaults to just bringing in the labels; switch to "add as new" to get a copy under a free
  name (or tick *Replace*). Using an existing project installs nothing and tells you if it
  doesn't define body parts, identities or behaviors that the labels use. Using an existing
  dataset copies no videos/frames: the labels are added to it.
- If the videos/frames weren't included, you'll be asked where they are.
- **Labels without their dataset:** you're asked which of your datasets they belong to. The
  labels are checked against that dataset's files and any mismatches are reported. If the
  dataset already has labels you choose how to combine them: **combine, keeping yours where
  both have a label**, **combine, using the imported ones**, **replace yours**, or **skip**. Your
  existing labels are backed up while this runs and restored if it fails.
- Labels that can't be combined (different body parts or identities) are never merged.

### Several datasets in one package — File → Export → Export Several Datasets…

Tick any number of datasets in the table; for each, choose the project it was labeled under
(guessed from its labels) and whether to copy its **videos/frames**, **pose labels** and
**behavior labels**. The result is one folder, `compilation_info.json` and `README.txt` at the
top and a complete standard export per dataset inside — the same format a single-dataset export
produces, so each inner folder also imports on its own, and full WHISKER reads them the same way.
If the export fails or is cancelled, the partly written folder is removed.

To import one, pick the compilation folder in **File → Import…** and click **Choose datasets…**.
A table lists every dataset with a checkbox for its videos/frames, pose labels and behavior labels:

- Each dataset has a choice: **add as new** (under the name in "Add as new") or **use existing**
  (pick one of your datasets; nothing is copied and the labels are added to it). Datasets you
  already have start on "use existing" with the dataset of the same name selected. Each project
  gets the same choice: add it as new under a name, or use one of yours.
- Where labels would land on existing labels, one choice below the table decides:
  **combine keeping yours**, **combine using the imported ones**, or **replace yours**.
- The *Notes* column reports label mismatches as you tick (e.g. "7/9 frames match, 2 skipped"),
  and asks you to locate any videos/frames that weren't copied into the package (double-click).
- Each project is installed once even if several datasets share it, and one dataset failing
  never stops the others — the summary lists what was imported and what wasn't.

**Import from separate files…** (button at the bottom of the Import dialog) is for files that
didn't come from Export, for example a folder of frames and some label files. You shouldn't have to
dig through folders for things you already have in your workspace:

- **Project:** choose *one of your existing projects* from a list (your active project is offered
  first), or add one from a `.json` file. If the labels use body parts, identities or behaviors
  your chosen project doesn't define, you're told, but the import isn't blocked.
- **Dataset:** choose *one of your existing datasets* (the list is ordered by how well the label
  files fit each one), or add a new dataset from its info file and media folder.
- **Labels:** browse for the pose and/or behavior label files. Onto an existing dataset they get the
  same mismatch report and combine / replace choices as any other import.

With an existing project and dataset chosen, the label files are the only thing to browse for.

**File → Import Labels from Other Software…** brings in labels made with other tools
(MARS, DLC, …); pick one of your existing datasets from its list (or type a new name), and your
active project is preselected.

Or copy an exported `workflows/` (and `projects/`) folder into your full WHISKER workspace;
WHISKER discovers the labels on its next scan.

---

## Updating an existing install

```bash
cd WHISKER-labeler
git pull
conda activate whisker-labeler
```

Because it was installed in editable mode (`pip install -e .`), a `git pull` is
usually all you need. Only if the dependencies changed, also run:

```bash
conda env update -f environment.yaml --prune
pip install -e .
```

---

## Notes & tips

- **Behavior and body-part names must match** the WHISKER project you'll train
  with (same spelling/case) so the labels line up.
- A dataset's **filenames** are how labels are matched to media — keep them
  consistent with your full WHISKER datasets.
- Hand the tool to a collaborator by sharing this repo; they install it the same
  way, create or import projects, label, and send the workspace (or exported
  labels) back.
