import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import shutil

from .dataset import Dataset, DatasetType, GLOB_EXTENSIONS_PER_DATASET_TYPE, MultiArenaConfig
from ..base_operation_helper import BaseOperationHelper

class DatasetOperations(BaseOperationHelper):
    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self._datasets: dict[str, Dataset] = {}
        
    def scan_datasets(self):
        """Scans the datasets directory and reloads metadata."""
        self._datasets.clear()
        if not self._base_dir.exists(): return

        for dataset_name in os.listdir(self._base_dir):
            if dataset_name.startswith("."): continue

            new_dataset = None            
            dataset_manifest_path = self._base_dir / dataset_name / "manifest.json"
            if dataset_manifest_path.exists():
                try:
                    with open(dataset_manifest_path, "rb") as f:
                        new_dataset = Dataset.from_json(f.read().decode())
                        self._datasets[dataset_name] = new_dataset
                except Exception as e:
                    logging.error(f"Failed to load dataset {dataset_name}: {e}")
    
        logging.info(f"[DatasetOperations] Completed scanning datasets. Found {len(self._datasets)} datasets.")
        for dataset in self._datasets.values():
            logging.debug(f"  - {dataset.name} ({dataset.type}, {len(dataset.files)} files)")

    def get_dataset(self, dataset_name: str) -> Optional[Dataset]:
        if dataset_name not in self._datasets:
            logging.warning(f"Dataset {dataset_name} does not exist!")
            return None
        return self._datasets[dataset_name]

    def refresh_dataset(self, dataset_name: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Scans the dataset's base path for changes, updates the file list,
        and manages symbolic links in the internal data directory.

        Args:
            dataset_name: The name of the dataset to refresh.
            dry_run: If True, returns changes without applying them.

        Returns:
            Dict containing 'added' (list), 'removed' (list), and 'current_count' (int).
        """
        logging.info(f"Performing refresh on dataset {dataset_name}...")
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        base_path = Path(dataset.base_data_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Source directory not found: {base_path}")

        # 1. Scan Source Directory
        extensions = GLOB_EXTENSIONS_PER_DATASET_TYPE.get(dataset.type, [])
        # Normalizing extensions to lowercase for case-insensitive comparison
        extensions = tuple(ext.lower() for ext in extensions)
        
        found_files = []
        
        # Recursive scan to handle hierarchy (e.g. FrameSubsets)
        logging.info(f"Scanning for dataset refresh under base path {base_path}...")
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(extensions):
                    full_path = Path(root) / file
                    logging.info(full_path)
                    try:
                        rel_path = full_path.relative_to(base_path)
                        # Normalize to forward slashes for internal consistency
                        found_files.append(str(rel_path).replace("\\", "/"))
                    except ValueError:
                        continue
        
        found_files_set = set(found_files)
        # Normalize current files to forward slashes for consistent comparison
        current_files_set = set(str(f).replace("\\", "/") for f in dataset.files)
        added_files = sorted(list(found_files_set - current_files_set))
        removed_files = sorted(list(current_files_set - found_files_set))

        if dry_run:
            return {
                "added": added_files,
                "removed": removed_files,
                "current_count": len(dataset.files)
            }

        # 2. Apply Changes
        dataset_dir = self._base_dir / dataset_name
        data_link_dir = dataset_dir / "data"
        os.makedirs(data_link_dir, exist_ok=True)

        # 2a. Add new symlinks
        for rel_path in added_files:
            src_path = base_path / rel_path
            dst_link = data_link_dir / rel_path
            
            # Ensure parent dir exists for the link (hierarchical datasets)
            if not dst_link.parent.exists():
                os.makedirs(dst_link.parent, exist_ok=True)

            if not dst_link.exists():
                try:
                    os.symlink(src_path, dst_link)
                except OSError as e:
                    logging.warning(f"Failed to create symlink for {src_path}: {e}")

        # 2b. Remove old symlinks
        for rel_path in removed_files:
            dst_link = data_link_dir / rel_path
            if dst_link.is_symlink() or dst_link.exists():
                try:
                    os.remove(dst_link)
                    # Optional: Cleanup empty directories could go here
                except OSError as e:
                    logging.warning(f"Failed to remove symlink {dst_link}: {e}")

        # 3. Update Manifest
        dataset.files = sorted(list(found_files_set))
        self.save_dataset(dataset_name)

        logging.info(f"Refreshed dataset '{dataset_name}': +{len(added_files)} / -{len(removed_files)}")

        return {
            "added": added_files,
            "removed": removed_files,
            "current_count": len(dataset.files)
        }

    def create_dataset(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        dataset_data_dir: Path,
        warn_if_exists: Optional[Callable[[str], bool]] = None,
    ):
        logging.info(f"Creating dataset: {dataset_name}")
        dataset_dir = self._base_dir / dataset_name
        
        if not BaseOperationHelper._delete_if_existing(dataset_dir, "dataset", warn_if_exists):
            return None

        extensions = GLOB_EXTENSIONS_PER_DATASET_TYPE[dataset_type]
        # Fixed list comprehension for correct extension checking
        files = [
            file for file in os.listdir(dataset_data_dir) if any(file.endswith(ext) for ext in extensions)
        ]
        files.sort()

        dataset_symbolic_link_dir = dataset_dir / "data"
        os.makedirs(dataset_symbolic_link_dir, exist_ok=True)
        for file in files:
            file_path = dataset_data_dir / file
            link_path = dataset_symbolic_link_dir / file
            os.symlink(file_path, link_path)

        new_dataset = Dataset(
            name=dataset_name,
            base_data_path=str(dataset_data_dir),
            type=dataset_type,
            files=files,
        )
        self._datasets[dataset_name] = new_dataset 
        self.add_dataset(dataset_name, new_dataset, overwrite_existing=False)

        logging.info(f"Created new dataset {dataset_name} with {len(files)} files")

    def create_multi_arena_dataset(
        self,
        dataset_name: str,
        dataset_data_dir: Path,
        box_width: int,
        box_height: int,
        placements: Dict[str, list],
        warn_if_exists: Optional[Callable[[str], bool]] = None,
    ):
        """
        Create a VIDEO_COLLECTION dataset whose entries are the parent videos
        that have at least one placed arena box, plus a ``multi_arena`` clipping
        config saved into the manifest. No masked video files are written; the
        per-arena masking is applied virtually on read by downstream consumers.
        """
        logging.info(f"Creating multi-arena dataset: {dataset_name}")
        dataset_dir = self._base_dir / dataset_name

        if not BaseOperationHelper._delete_if_existing(dataset_dir, "dataset", warn_if_exists):
            return None

        # Only videos with at least one arena become dataset entries.
        files = sorted(rel for rel, boxes in placements.items() if boxes)

        data_link_dir = dataset_dir / "data"
        os.makedirs(data_link_dir, exist_ok=True)
        for rel in files:
            src_path = Path(dataset_data_dir) / rel
            dst_link = data_link_dir / rel
            if not dst_link.parent.exists():
                os.makedirs(dst_link.parent, exist_ok=True)
            if not dst_link.exists():
                try:
                    os.symlink(src_path, dst_link)
                except OSError as e:
                    logging.warning(f"Failed to create symlink for {src_path}: {e}")

        new_dataset = Dataset(
            name=dataset_name,
            base_data_path=str(dataset_data_dir),
            type=DatasetType.VIDEO_COLLECTION,
            files=files,
            multi_arena=MultiArenaConfig(
                box_width=int(box_width),
                box_height=int(box_height),
                placements={rel: [(int(x), int(y)) for (x, y) in boxes] for rel, boxes in placements.items() if boxes},
            ),
        )
        self._datasets[dataset_name] = new_dataset
        self.add_dataset(dataset_name, new_dataset, overwrite_existing=False)

        logging.info(
            f"Created multi-arena dataset {dataset_name} with {len(files)} videos "
            f"and {sum(len(b) for b in placements.values())} arena placements"
        )
        return new_dataset

    def update_multi_arena_dataset(
        self,
        dataset_name: str,
        box_width: int,
        box_height: int,
        placements: Dict[str, list],
    ) -> Optional[Dataset]:
        """Revise an existing multi-arena dataset's box size and placements
        in place, preserving its name and ``base_data_path``.

        Unlike :meth:`create_multi_arena_dataset`, this does NOT delete the
        dataset directory, so downstream per-arena artifacts (behavior labels,
        pose/behavior predictions keyed by ``{stem}_arena{k}``) are left on disk.
        Callers are responsible for warning about invalidated arenas first — see
        :func:`whisker.core.study.dataset.analyze_arena_edit`. The ``files`` list
        is recomputed to the videos that still have at least one placed arena,
        and data symlinks are added for any newly-included videos.
        """
        existing = self._datasets.get(dataset_name)
        if existing is None or not existing.is_multi_arena:
            raise ValueError(
                f"Dataset '{dataset_name}' is not an existing multi-arena dataset."
            )

        files = sorted(rel for rel, boxes in placements.items() if boxes)

        # Ensure a data symlink exists for every included video (new ones may
        # have gained their first arena in this edit).
        data_link_dir = self._base_dir / dataset_name / "data"
        os.makedirs(data_link_dir, exist_ok=True)
        for rel in files:
            src_path = Path(existing.base_data_path) / rel
            dst_link = data_link_dir / rel
            if not dst_link.parent.exists():
                os.makedirs(dst_link.parent, exist_ok=True)
            if not dst_link.exists():
                try:
                    os.symlink(src_path, dst_link)
                except OSError as e:
                    logging.warning(f"Failed to create symlink for {src_path}: {e}")

        updated = existing.model_copy(update={
            "files": files,
            "multi_arena": MultiArenaConfig(
                box_width=int(box_width),
                box_height=int(box_height),
                placements={
                    rel: [(int(x), int(y)) for (x, y) in boxes]
                    for rel, boxes in placements.items() if boxes
                },
            ),
        })
        self._datasets[dataset_name] = updated
        self.save_dataset(dataset_name)

        logging.info(
            f"Updated multi-arena dataset {dataset_name}: {len(files)} videos, "
            f"{sum(len(b) for b in placements.values())} arena placements"
        )
        return updated

    def add_dataset(
        self,
        dataset_name: str,
        dataset: Dataset,
        warn_if_exists: Optional[Callable[[str], bool]] = None,
        overwrite_existing: bool = True,
    ):
        dataset_dir = self._base_dir / dataset_name

        if overwrite_existing and not BaseOperationHelper._delete_if_existing(
            dataset_dir, "dataset", warn_if_exists
        ):
            return None

        self._datasets[dataset_name] = dataset
        self.save_dataset(dataset_name)
        logging.info(f"Added dataset {dataset_name} to workspace")

    def remove(self, dataset_name: str) -> None:
        dataset = self._datasets.get(dataset_name)
        if not dataset:
            logging.warning(f"Called to remove Dataset {dataset_name} but no such dataset exists.")
            return
        
        del self._datasets[dataset_name]
        shutil.rmtree(self.base_dir / dataset_name)
        logging.info(f"Removed dataset {dataset_name} from the workspace")

    def save_dataset(self, dataset_name: str) -> None:
        dataset = self._datasets[dataset_name]
        dataset_dir = self._base_dir / dataset_name
        dataset_manifest_path = dataset_dir / "manifest.json"

        os.makedirs(dataset_dir, exist_ok=True)
        with open(dataset_manifest_path, "w", encoding="utf-8") as f:
            json_str = dataset.model_dump_json(indent=4)
            f.write(json_str)
        logging.info(
            f"Saved manifest for dataset {dataset_name} to {dataset_manifest_path}"
        )

    def show_dataset(self, dataset_name: Optional[str] = None, verbose: bool = False):
        if dataset_name is None:
            logging.info("Showing all datasets:")
            for name in self._datasets.keys():
                logging.info(f"  - {name}")
            for name in self._datasets.keys():
                self.show_dataset(name, verbose=verbose)
        else:
            dataset = self._datasets.get(dataset_name)
            if not dataset:
                raise ValueError(f"Dataset {dataset_name} does not exist.")
            dataset.show(verbose=verbose)

    def find_dataset_by_file_path(self, absolute_path: Path) -> Optional[Dataset]:
        if not absolute_path.is_absolute():
            logging.warning(f"Cannot find dataset for non-absolute path: {absolute_path}")
            return None

        for dataset in self._datasets.values():
            try:
                absolute_base = Path(dataset.base_data_path).resolve()
                absolute_file = absolute_path.resolve()
                relative_path = absolute_file.relative_to(absolute_base)
                relative_path_str = str(relative_path).replace("\\", "/")
                if relative_path_str in dataset.files:
                    return dataset
            except (ValueError, OSError):
                continue
        return None

    def get(self, name: str):
        return self._datasets.get(name)
    
    def prune(self) -> dict[Path, str]:
        """
        Scans for corrupt or incomplete dataset entries.
        Returns:
            dict[Path, str]: Mapping of folder path to reason for pruning.
        """
        findings = {}
        if not self._base_dir.exists():
            return findings

        for dataset_name in os.listdir(self._base_dir):
            if dataset_name.startswith("."): continue
            
            dataset_dir = self._base_dir / dataset_name
            if not dataset_dir.is_dir(): continue

            manifest_path = dataset_dir / "manifest.json"
            
            # 1. Missing Manifest
            if not manifest_path.exists():
                findings[dataset_dir] = "Missing manifest.json"
                continue

            # 2. Corrupt Manifest
            try:
                with open(manifest_path, "rb") as f:
                    Dataset.from_json(f.read().decode())
            except Exception as e:
                findings[dataset_dir] = f"Corrupt manifest.json: {e}"
                continue

            # 3. Missing or Empty data directory
            data_dir = dataset_dir / "data"
            if not data_dir.exists():
                findings[dataset_dir] = "Missing 'data' directory (symlinks)"
            elif not os.listdir(data_dir):
                findings[dataset_dir] = "Empty 'data' directory"

        return findings

    def keys(self) -> list[str]:
        return list(self._datasets.keys())
    
    def values(self) -> list[Dataset]:
        return list(self._datasets.values())

    def items(self) -> list[tuple[str, Dataset]]:
        return list(self._datasets.items())