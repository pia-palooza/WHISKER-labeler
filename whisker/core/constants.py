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
    "#E6194B",  # Red
    "#3CB44B",  # Green
    "#4363D8",  # Blue
    "#FFE119",  # Yellow
    "#42D4F4",  # Cyan
    "#F032E6",  # Magenta
    "#FABEBE",  # Pink
    "#F58231",  # Orange
    "#911EB4",  # Purple
    "#000075",  # Navy
]
