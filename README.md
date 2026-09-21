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

The labeler and full WHISKER exchange work as **Whisker bundles**: an ordinary ZIP file
(ZIP64, so multi-gigabyte bundles are fine) holding one or more datasets with their labels,
and optionally prediction results and project definitions. Media and label files are stored
exactly as they exist in the workspace — uncompressed and unconverted — so nothing is
translated on the way between the two tools. A bundle made by either one opens in the other.

Because the labeler shares WHISKER's workspace layout, you can also point it directly at a
full WHISKER workspace — no bundle needed.

### Export — File → Export → Export Whisker Bundle…

Tick the datasets to export (their media and labels always go in), then optionally the
projects and prediction results to include, and choose where to save the `.zip`. The
suggested file name follows WHISKER's (`whisker_bundle_<dataset>_<date>.zip`).

- The bundle is written to a temporary file and only given its real name once complete, so a
  cancelled or failed export never leaves a partial bundle (and never replaces a good one).
- Afterwards the finished bundle is read back and every file is checked against its stored
  checksum; the completion message says so, or lists exactly what is wrong.
- A media file that can't be found on disk isn't in the bundle and is listed in it as
  *missing*, as WHISKER does; you're told which ones.
- Label edits still open in a labeling tab aren't on disk yet, so you're asked whether to
  save them first.

### Import — File → Import Whisker Bundle… (Ctrl+Shift+O)

Choose the `.zip`. Only its listing and manifest are read to show what it holds: datasets,
their labels, prediction results and projects, each marked **New** or **Already here**. New
things are ticked; anything you already have is left alone unless you tick it, and replacing
asks for confirmation first.

- **Frame subsets** are stored inside the workspace. **Videos and image collections** go in a
  folder you choose (each dataset gets its own folder inside it; media already outside the
  workspace are never overwritten) and the workspace links to them. Each dataset's stored media
  location is rewritten to match this machine.
- Labels, prediction results and projects are copied over unchanged.
- The bundle is treated as untrusted: unsafe paths are refused, and everything is extracted to a
  staging folder with each file's checksum verified. Only when the whole archive has been read
  cleanly are the pieces moved into place, so a corrupt, truncated or cancelled import leaves
  the workspace exactly as it was.
- A bundle from a newer format version is refused with an explanation; fields this version
  doesn't know are ignored.

Layout of a bundle:

```
whisker_bundle.json                                     manifest: what's inside (read first)
datasets/<dataset>/manifest.json                        the dataset; its `files` list is authoritative
datasets/<dataset>/media/<relative path>                frames / videos / images
workflows/<workflow>/labels/<dataset>/…                 pose labels.h5 + metadata.json, behavior labels.h5, …
workflows/<workflow>/predictions/<run>/<dataset>/…      prediction results
projects/<name>.json                                    project definitions
```

**File → Import Labels from Other Software…** brings in labels made with other tools
(MARS, DLC, …); pick one of your existing datasets from its list (or type a new name), and your
active project is preselected.

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
  way, create or import projects, label, and send a Whisker bundle (or the workspace) back.
