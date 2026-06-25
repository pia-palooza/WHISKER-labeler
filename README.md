# WHISKER Labeler

A standalone desktop app for **hand-annotating** animal data:

- **Pose keypoints** on images / extracted frames (the *Pose Estimation* tab)
- **Behavior bouts** on videos (the *Behavior Classification* tab)

It is a lightweight spin-off of the main WHISKER analysis pipeline — no GPU and
no deep-learning libraries required — and it saves labels in the **exact HDF5
format the full WHISKER app uses**, so your annotations import straight back into
WHISKER for model training.

> This tool is for **hand annotation only**. It does not train models.

---

## How it works

- **One window, two tabs.** Switch freely between **Pose Estimation** (images) and
  **Behavior Classification** (videos) at the top.
- **A self-contained workspace.** All of your projects, datasets, and labels live
  inside the app's own `workspace/` folder (created next to the code). Nothing is
  ever written to your original videos/frames, and the app never touches a full
  WHISKER workspace. Everything you create reloads automatically next launch.
- **A Data Explorer** on the left lists every dataset and its files, with a ✓ on
  files you've already labeled.
- **Projects** define the labeling targets — body parts, identities (animals),
  behaviors, and an optional skeleton. A dataset is just a folder of media tied to
  a project.
- **Import / Export** move data between this labeler and a full WHISKER workspace.

---

## Requirements

- **Miniconda or Anaconda** (recommended) — <https://docs.conda.io/en/latest/miniconda.html>
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

# 3. Install the launcher command
pip install -e .
```

That's it. Step 3 installs a `whisker-labeler` command and lets the app run from
any folder.

---

## Updating an existing install

If you already installed it and just want the latest version:

```bash
cd WHISKER-labeler
git pull
conda activate whisker-labeler
```

Because it was installed in "editable" mode (`pip install -e .`), a `git pull` is
usually all you need — the new code is picked up automatically.

Only if the dependencies or launcher changed, also run:

```bash
conda env update -f environment.yaml --prune
pip install -e .
```

---

## Launching

```bash
conda activate whisker-labeler
whisker-labeler
```

**Windows:** you can also just **double-click `launch.bat`** in the
`WHISKER-labeler` folder.

The app opens straight into its built-in workspace — there is **no folder prompt**.
The first time you run it the workspace is empty; you fill it using **New Project /
Dataset…** or **Import…** (below).

---

## Loading data into the labeler

There are two ways to get datasets in. Both write into the app's own workspace and
**never modify your source media**.

### Option A — Create a new project + dataset

1. Click **➕ New Project / Dataset…** in the toolbar.
2. **Data Folder** — browse to a folder containing your media:
   - a folder of **videos** → becomes a Behavior dataset
   - a folder of **images / extracted frames** → becomes a Pose dataset
   - (the app auto-detects which, and finds files in sub-folders too)
3. **Dataset Name** — defaults to the folder name; change if you like.
4. **Project** — either:
   - **➕ Create new project…** and define **Body Parts**, **Identities**,
     **Behaviors**, and an optional **Skeleton** (all comma-separated, e.g.
     `nose, left_ear, right_ear, tail_base`), **or**
   - pick an **existing project** to reuse its definition for this new dataset.
5. Click **Create**. The dataset appears in the Data Explorer and is ready to label.

### Option B — Import existing projects / datasets / labels

Use this to pull work in from a **full WHISKER workspace** or a **previous export**.

1. Click **⬇ Import…**.
2. **Source Folder** — browse to a WHISKER workspace (a folder with
   `projects/` and/or `datasets/`) or a folder previously produced by Export.
3. Check the **projects** and **datasets** you want (datasets are tagged with the
   labels they contain — `pose` / `behavior`). **Select All** is available.
4. Click **Import**. The selected projects, datasets, and their existing labels are
   **copied into your labeler**. The source folder is only read, never changed.

> Projects appear in the toolbar **Project** dropdown; datasets appear in the Data
> Explorer.

---

## Labeling

Pick a project in the toolbar dropdown, then click a file in the Data Explorer.
Image datasets open in the **Pose Estimation** tab; video datasets in the
**Behavior Classification** tab. Press **`Ctrl+S`** to save; a ✓ marks labeled files.

### Pose Estimation (keypoints on images)

Select a body part for an identity in the side panel, then click on the image to
place it; drag to adjust.

| Key | Action |
| --- | --- |
| `Q` | Toggle drag/move mode |
| `W` / `S` | Previous / next body part |
| `A` / `D` | Previous / next identity |
| `Delete` / `X` | Clear selected keypoint |
| `Ctrl+Delete` | Clear all keypoints on this image |
| `M` / `N` | Previous / next image |
| `Ctrl+S` | Save |

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
| `M` / `N` | Previous / next video |
| `Ctrl+S` | Save |

---

## Exporting labels back to full WHISKER

1. Select the dataset you want in the Data Explorer.
2. Click **⬆ Export Labels…** and choose a destination folder.
3. The labeler writes the labels (plus the matching project and dataset manifest)
   into a **WHISKER-compatible layout**:
   ```
   <destination>/workflows/<pose_estimation|behavior_classification>/labels/<dataset>/labels.h5
   <destination>/projects/<project>.json
   <destination>/datasets/<dataset>/manifest.json
   ```
4. Merge that `workflows` (and `projects`) folder into your full WHISKER workspace.
   WHISKER discovers the labels on its next scan — no conversion needed.

---

## Where your data lives

Everything you create or import is stored under the app's own workspace:

```
WHISKER-labeler/workspace/
  projects/<name>.json                                   # label definitions
  datasets/<name>/manifest.json                          # which media files
  workflows/pose_estimation/labels/<dataset>/labels.h5   # pose labels
  workflows/behavior_classification/labels/<dataset>/labels.h5  # behavior labels
```

This `workspace/` folder is intentionally excluded from version control, so your
annotations stay local and private. Back it up if you want to keep copies.

---

## Notes & tips

- **Behavior names must match** the project you'll train with in full WHISKER
  (same spelling/case) for the labels to line up.
- A dataset's **video/frame filenames** are how labels are matched, so keep them
  consistent with your full WHISKER datasets.
- Hand the tool to a collaborator by sharing this repo; they install it the same
  way, create or import projects, label, and send you the **Export**ed folder.
