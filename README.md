# WHISKER: Standalone Pose Labeling Tool

A standalone GUI for annotating animal pose keypoints in video frames and images. Extracted from the main WHISKER analysis pipeline, this tool is designed for rapid deployment, fast local installation (no GPU or deep learning libraries required), and efficient manual labeling directly to HDF5 files.

## Features

* **Fast & Lightweight**: Stripped of heavy ML dependencies for instant installation on standard laptops.
* **Intuitive UI**: PyQt6-based interface with interactive image panning, zooming, and drag-and-drop keypoint adjustment.
* **HDF5 Integration**: Saves labels directly to the standard HDF5 format used by downstream pose estimation models.
* **Project Aware**: Synchronizes labeling targets using your existing `project.json` configurations.

## Installation

### Prerequisites
* Python 3.11
* Anaconda or Miniconda (recommended)

### Setup

1.  Clone the repository:
    ```bash
    git clone [ https://github.com/pia-palooza/WHISKER-labeler.git](https://github.com/pia-palooza/WHISKER-labeler.git)
    cd WHISKER-labeler
    ```

2.  Create and activate the environment:
    ```bash
    conda env create -f environment.yaml
    conda activate whisker-labeler
    ```

    *Alternatively, install dependencies using pip:*
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Launch the application using:

```bash
python -m whisker.main