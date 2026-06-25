# WHISKER Labeler

A standalone GUI for hand-annotating animal **pose keypoints** (on images/frames)
and **behavior bouts** (on videos). It is extracted from the main WHISKER
analysis pipeline for fast, lightweight, local labeling — no GPU or deep-learning
libraries required — and it writes labels in the exact HDF5 format the full
WHISKER application uses, so annotations import straight back for training.

## Features

- **Two workflows, one window** — top tabs for *Pose Estimation* (images/frames)
  and *Behavior Classification* (videos), switchable at any time.
- **Self-contained workspace** — projects, datasets, and annotations are stored
  inside the app's own `workspace/` folder and reload automatically on launch.
- **Create projects/datasets in-app** — point at a folder of videos or frames and
  define body parts, identities, behaviors, and skeleton.
- **Import / Export** — pull existing projects and labels in from a WHISKER
  workspace, and export labels back out in a WHISKER-compatible layout.
- **Never touches your full WHISKER data** — source media is read-only and all
  output stays in the labeler's own workspace.
- **HDF5 labels** — pose and behavior labels are saved in the standard WHISKER
  format for seamless downstream training.

## Installation

### Prerequisites
- Python 3.11
- Anaconda or Miniconda (recommended)

### Setup
```bash
git clone https://github.com/pia-palooza/WHISKER-labeler.git
cd WHISKER-labeler

# Create and activate the environment
conda env create -f environment.yaml
conda activate whisker-labeler

# Install the launcher command (editable)
pip install -e .
```

## Launching

```bash
conda activate whisker-labeler
whisker-labeler
```

The app opens directly into its built-in workspace — no folder prompt. On
Windows you can also double-click `launch.bat`.

## Usage

1. Click **➕ New Project / Dataset…** to create a project (body parts /
   identities / behaviors) and add a dataset by pointing at a folder of videos
   or frames. You can also **reuse an existing project** for a new dataset.
2. Use the **Data Explorer** on the left to navigate datasets and files. Image
   datasets open in the Pose tab; video datasets in the Behavior tab.
3. Annotate. `Ctrl+S` saves; a ✓ marks labeled files.
4. **⬇ Import…** brings projects/datasets/labels in from a WHISKER workspace or a
   previous export. **⬆ Export Labels…** writes the current dataset's labels into
   a WHISKER-compatible folder to merge into your full WHISKER workspace.

## Notes

- This tool is for **hand annotation only** — it does not train models.
- The built-in `workspace/` folder holds your projects, datasets, and labels and
  is intentionally excluded from version control.
