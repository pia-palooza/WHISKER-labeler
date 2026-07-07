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
point it directly at a full WHISKER workspace — no conversion needed. You can
also:

- **Import existing pose labels** — File → Import Pose Labels…
- **Export** — the File → Export submenu (behavior labels, bouts, charts).

Merge an exported `workflows/` (and `projects/`) folder into your full WHISKER
workspace; WHISKER discovers the labels on its next scan.

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
