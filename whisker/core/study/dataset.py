import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

VIDEO_FILE_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".m4v",
    ".ogv",
    ".vob",
    ".ts",
    ".mts",
    ".m2ts",
    ".seq",
)

IMAGE_FILE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heic",
    ".heif",
    ".raw",
    ".arw",
    ".cr2",
    ".nef",
    ".orf",
    ".sr2",
    ".dng",
    ".ico",
    ".svg",
)

class DatasetType(str, Enum):
    """Enumeration for the different types of datasets."""

    VIDEO_COLLECTION = "VIDEO_COLLECTION"
    IMAGE_COLLECTION = "IMAGE_COLLECTION"
    FRAME_SUBSET = "FRAME_SUBSET"

# Maps the expected file extensions for each DatasetType
GLOB_EXTENSIONS_PER_DATASET_TYPE = {
    DatasetType.VIDEO_COLLECTION: VIDEO_FILE_EXTENSIONS,
    DatasetType.IMAGE_COLLECTION: IMAGE_FILE_EXTENSIONS,
    DatasetType.FRAME_SUBSET: IMAGE_FILE_EXTENSIONS,
}

class MultiArenaConfig(BaseModel):
    """
    Optional multi-arena clipping configuration for a VIDEO_COLLECTION dataset.

    All arenas in a dataset share one box size (``box_width`` x ``box_height``);
    only each placement's top-left ``(x, y)`` varies. ``placements`` maps a
    video's dataset-relative path (forward slashes) to the list of arena box
    top-left positions placed on it. Everything is expressed in full-frame
    pixels — no cropping, so poses/labels stay in full-frame coordinates.
    """

    box_width: int
    box_height: int
    placements: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)

    def boxes_for(self, video_rel_path: str) -> List[Tuple[int, int, int, int]]:
        """Arena boxes for one video as ``(x, y, w, h)`` tuples in full-frame
        pixels (shared box size applied to each placement)."""
        return [
            (int(x), int(y), int(self.box_width), int(self.box_height))
            for (x, y) in self.placements.get(video_rel_path, [])
        ]

    @staticmethod
    def arena_stem(video_rel_path: str, arena_index: int) -> str:
        """The canonical per-arena "pseudo-video" stem, e.g. ``clip_arena0``.

        This is the single source of truth for the ``{video_stem}_arena{k}``
        naming used across masked frame extraction, pose prediction keys, and
        behavior label keys."""
        return f"{Path(video_rel_path).stem}_arena{arena_index}"

    def arena_units(self) -> List[Tuple[str, int, str, Tuple[int, int, int, int]]]:
        """Every placed arena as ``(video_rel_path, arena_index, arena_stem, box)``."""
        units = []
        for video_rel_path in self.placements:
            for k, box in enumerate(self.boxes_for(video_rel_path)):
                units.append((video_rel_path, k, self.arena_stem(video_rel_path, k), box))
        return units

    def resolve_arena_stem(
        self, stem: str
    ) -> Optional[Tuple[str, int, Tuple[int, int, int, int]]]:
        """Map an arena pseudo-video stem (e.g. ``clip_arena0``) back to its
        ``(video_rel_path, arena_index, box)``, or ``None`` if it isn't a valid
        arena of this config."""
        for video_rel_path, arena_index, arena_stem, box in self.arena_units():
            if arena_stem == stem:
                return (video_rel_path, arena_index, box)
        return None


class ArenaEditImpact(BaseModel):
    """Classification of how an edit to a multi-arena config changes existing
    per-arena "pseudo-videos" (keyed by the positional stem ``{stem}_arena{k}``).

    Because an arena's identity is its *index* in the per-video placement list,
    inserting/removing/reordering shifts the boxes underneath every later stem.
    ``moved_stems`` are stems that survive but now cover a different region (a
    genuine move, an index shift, or a shared box-size change) — their existing
    pose predictions were computed on the old box and are no longer valid.
    ``removed_stems`` no longer exist, so their artifacts are orphaned.
    ``added_stems`` are brand-new and carry no artifacts yet (always safe)."""

    box_size_changed: bool = False
    moved_stems: List[str] = Field(default_factory=list)
    removed_stems: List[str] = Field(default_factory=list)
    added_stems: List[str] = Field(default_factory=list)

    @property
    def invalidated_stems(self) -> List[str]:
        """Stems whose existing labels/pose/behavior artifacts no longer match
        the box they are keyed to (moved + removed)."""
        return list(self.moved_stems) + list(self.removed_stems)

    @property
    def has_risky_changes(self) -> bool:
        return bool(self.moved_stems or self.removed_stems)


def analyze_arena_edit(
    old_config: "MultiArenaConfig",
    new_box_width: int,
    new_box_height: int,
    new_placements: Dict[str, List[Tuple[int, int]]],
) -> ArenaEditImpact:
    """Compare an existing multi-arena config against a proposed edit and report
    which arena stems are added, moved (invalidated), or removed.

    Comparison is strictly index-by-index per video, because that is exactly how
    the ``{stem}_arena{k}`` naming binds artifacts to boxes. A change to the
    shared box size invalidates *every* surviving arena, since the crop region
    of each one changes."""
    box_size_changed = (
        int(old_config.box_width) != int(new_box_width)
        or int(old_config.box_height) != int(new_box_height)
    )

    impact = ArenaEditImpact(box_size_changed=box_size_changed)

    videos = set(old_config.placements) | set(new_placements)
    for video_rel in videos:
        old_list = [tuple(p) for p in old_config.placements.get(video_rel, [])]
        new_list = [tuple(p) for p in new_placements.get(video_rel, [])]
        for k in range(max(len(old_list), len(new_list))):
            stem = MultiArenaConfig.arena_stem(video_rel, k)
            in_old = k < len(old_list)
            in_new = k < len(new_list)
            if in_old and in_new:
                if box_size_changed or old_list[k] != new_list[k]:
                    impact.moved_stems.append(stem)
            elif in_old:
                impact.removed_stems.append(stem)
            else:
                impact.added_stems.append(stem)

    return impact


class Dataset(BaseModel):
    """
    Pydantic model for a single dataset's metadata.
    This provides validation and a clear structure.
    """

    name: str
    type: DatasetType
    base_data_path: str
    files: List[str] = Field(default_factory=list)
    # Present only for multi-arena datasets; None for ordinary single-arena ones.
    multi_arena: Optional[MultiArenaConfig] = None
    # For arena-sampled FRAME_SUBSET datasets only: maps each arena "pseudo-video"
    # subdir (e.g. "clip_arena0") to the fixed arena box (x, y, w, h) in full-frame
    # pixels that its masked frames were cut from. Lets top-down training with
    # crop_source="arena" recover the exact crop box per frame without re-deriving
    # it from keypoints. Absent (None) for ordinary/legacy datasets.
    arena_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None

    @property
    def is_multi_arena(self) -> bool:
        return self.multi_arena is not None

    @staticmethod
    def arena_subdir_of(rel_path: str) -> str:
        """The leading subdir of a dataset-relative frame path, which for arena
        frames is the arena stem (e.g. "clip_arena0/frame_000010.png" ->
        "clip_arena0"). Single source of truth for "arena stem == parts[0]"."""
        parts = Path(str(rel_path).replace("\\", "/")).parts
        return parts[0] if parts else ""

    def arena_box_for_file(
        self, rel_path: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """The arena box for a dataset-relative frame path (e.g.
        "clip_arena0/frame_000010.png"), keyed by its arena-stem subdir, or None
        if this dataset carries no arena boxes / the file isn't under an arena
        subdir."""
        if not self.arena_boxes:
            return None
        box = self.arena_boxes.get(self.arena_subdir_of(rel_path))
        return tuple(box) if box is not None else None

    def resolve_arena_stem(
        self, stem: str
    ) -> Optional[Tuple[str, int, Tuple[int, int, int, int]]]:
        """``(video_rel_path, arena_index, box)`` for an arena pseudo-video stem,
        or ``None`` if this dataset isn't multi-arena or the stem doesn't match."""
        if self.multi_arena is None:
            return None
        return self.multi_arena.resolve_arena_stem(stem)

    @classmethod
    def from_json(cls, json_str: str) -> "Dataset":
        """
        Create a Dataset object from a JSON string.
        """
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            base_data_path=data["base_data_path"],
            type=DatasetType(data["type"]),
            files=data["files"],
            # Additive/optional: absent in legacy manifests -> None.
            multi_arena=data.get("multi_arena"),
            arena_boxes=data.get("arena_boxes"),
        )

    def show(self, verbose: bool = False) -> None:
        """
        Show the dataset's metadata.
        """
        if verbose:
            logging.info(f"Dataset {self.name} ({self.type.value})")
            logging.info(self.model_dump_json(indent=2))
        else:
            logging.info(
                f"Dataset {self.name} ({self.type.value}) contains {len(self.files)} files"
            )

            num_files_to_show = min(10, len(self.files))
            logging.info(f"First {num_files_to_show} files:")
            for i in range(num_files_to_show):
                logging.info(f"  - {self.files[i]}")
