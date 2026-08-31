import enum
from PyQt6.QtGui import QColor

from whisker.core.constants import KEYPOINT_COLORS, WHISKER_BASE_SRC_DIR
from whisker.core.study.dataset import (
    IMAGE_FILE_EXTENSIONS,
    VIDEO_FILE_EXTENSIONS,
)

# --- Path Setup ---
ASSETS_DIR = WHISKER_BASE_SRC_DIR / "gui" / "assets"

# --- UI Constants ---
CHECKMARK_INDICATOR = "✔"

# --- Media File Extensions ---
# Kept in sync with the canonical lists in whisker.core.study.dataset so the
# "Create Dataset" dialog recognises exactly the same media the dataset scan
# will (previously the GUI list was a short subset, so e.g. a folder of .webp
# or .tif frames wasn't detected as an image dataset at all).
TEXT_EXTENSIONS = [".md", ".log", ".txt", ".yml", ".yaml", ".json"]
VIDEO_EXTENSIONS = list(VIDEO_FILE_EXTENSIONS)
IMAGE_EXTENSIONS = list(IMAGE_FILE_EXTENSIONS)
HDF5_EXTENSIONS = [".h5"]

KEYPOINT_QCOLORS = [
    QColor(c) for c in KEYPOINT_COLORS
]
