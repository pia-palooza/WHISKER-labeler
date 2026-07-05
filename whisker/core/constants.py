from pathlib import Path
import enum
from typing import Callable, Optional


class BackendEnum(str, enum.Enum):
    WHISKER = "WHISKER"
    SLEAP = "SLEAP"
    DLC = "DLC"
    SIMBA = "SIMBA"
    MARS = "MARS"


WarnIfExistsFunctionType = Optional[Callable[[str], bool]]

WHISKER_BASE_SRC_DIR = Path(__file__).resolve().parent.parent

# DEV_NOTE: A palette of visually distinct colors for different identities.
# This list can be expanded if more than 10 simultaneous identities are needed.
KEYPOINT_COLORS = [
    "#FF00FF",  # Vibrant Magenta/Purple (High Luminance)
    "#00FF00",  # Neon Green (High Luminance)
    "#00FFFF",  # Neon Cyan (Extremely Bright)
    "#FF9100",  # Neon Orange (Bright)
    "#2979FF",  # Electric Blue (Medium Luminance)
    "#FFFF00",  # Neon Yellow (Highest Luminance)
    "#FF1744",  # Vibrant Red/Pink (Darker Saturated)
    "#1DE9B6",  # Neon Teal (Bright Green-Blue)
    "#9933FF",  # Vibrant Purple/Violet (Medium Luminance)
    "#FF6E40",  # Neon Coral (Bright Orange-Red)
]
