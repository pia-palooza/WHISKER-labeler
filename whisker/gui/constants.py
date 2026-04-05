import enum
from PyQt6.QtGui import QColor

from whisker.core.constants import KEYPOINT_COLORS, WHISKER_BASE_SRC_DIR

# --- Path Setup ---
ASSETS_DIR = WHISKER_BASE_SRC_DIR / "gui" / "assets"

# --- UI Constants ---
CHECKMARK_INDICATOR = "✔"

# --- Media File Extensions ---
TEXT_EXTENSIONS = [".md", ".log", ".txt", ".yml", ".yaml", ".json"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
HDF5_EXTENSIONS = [".h5"]

KEYPOINT_QCOLORS = [
    QColor(c) for c in KEYPOINT_COLORS
]
