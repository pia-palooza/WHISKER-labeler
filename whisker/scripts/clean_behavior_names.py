import argparse
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd
import numpy as np

# --- 1. Path Setup ---
# Add the project root to sys.path so we can import 'whisker'
# Assumes this script is in <root>/scripts/
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from whisker.core.workspace import Workspace
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset

# --- 2. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("clean_behavior_names")

def normalize(name: str) -> str:
    """The cleaning logic: lowercase and snake_case."""
    return name.strip().lower().replace(" ", "_")

def _has_changes(original_list: List[str]) -> bool:
    """Quick check if a list needs normalization."""
    for item in original_list:
        if item != normalize(item):
            return True
    return False

def _get_rename_map(original_list: List[str]) -> Dict[str, str]:
    """Generates a map of {OldName: NewName} for items that change."""
    mapping = {}
    for item in original_list:
        norm = normalize(item)
        if item != norm:
            mapping[item] = norm
    return mapping

# --- 3. Processors ---

def process_projects(ws: Workspace, apply: bool):
    logger.info("--- Scanning Projects ---")
    
    for project in ws.projects.values():
        if not _has_changes(project.behaviors):
            continue
            
        rename_map = _get_rename_map(project.behaviors)
        logger.info(f"Project '{project.name}': Found {len(rename_map)} behaviors to normalize.")
        for old, new in rename_map.items():
            logger.info(f"  - '{old}' -> '{new}'")
            
        if apply:
            # Set comprehension dedupes collisions (e.g. 'Sit' and 'sit' -> 'sit')
            new_behaviors = sorted(list({normalize(b) for b in project.behaviors}))
            project.behaviors = new_behaviors
            ws.save_project(project.name)
            logger.info(f"  [APPLIED] Saved project '{project.name}'.")

def process_labels(ws: Workspace, apply: bool):
    logger.info("\n--- Scanning Behavior Labels (Ground Truth) ---")
    
    for dataset_name in ws.datasets.keys():
        if not ws.behavior_labels.has_behavior_labels(dataset_name):
            continue
            
        labels = ws.get_behavior_labels(dataset_name)
        
        # Check Metadata List
        list_dirty = _has_changes(labels.behaviors)
        
        # Check DataFrame Values
        df_dirty = False
        rename_map = {}
        if not labels.bouts.empty:
            unique_behaviors = labels.bouts['behavior'].unique()
            rename_map = _get_rename_map(unique_behaviors)
            if rename_map:
                df_dirty = True

        if list_dirty or df_dirty:
            logger.info(f"Dataset '{dataset_name}': Needs normalization.")
            if df_dirty:
                logger.info(f"  - Renaming {len(rename_map)} behavior values in Bouts DataFrame.")
            
            if apply:
                # 1. Normalize list
                labels.behaviors = sorted(list({normalize(b) for b in labels.behaviors}))
                
                # 2. Normalize DataFrame
                if df_dirty:
                    labels.bouts['behavior'] = labels.bouts['behavior'].replace(rename_map)
                
                ws.save_behavior_labels(dataset_name)
                logger.info(f"  [APPLIED] Saved labels for '{dataset_name}'.")

def process_models(ws: Workspace, apply: bool):
    logger.info("\n--- Scanning Behavior Models (Config) ---")
    
    models_dir = ws.behavior_models.base_dir
    if not models_dir.exists():
        return

    for run_dir in models_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
            
        config_path = run_dir / "model_config.json"
        if not config_path.exists():
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            behaviors = config.get("behaviors", [])
            if _has_changes(behaviors):
                logger.info(f"Model '{run_dir.name}': Normalizing behavior list.")
                
                if apply:
                    # Backup
                    shutil.copy(config_path, str(config_path) + ".bak")
                    
                    # Normalize (dedupe if collision)
                    config["behaviors"] = sorted(list({normalize(b) for b in behaviors}))
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=4)
                    logger.info(f"  [APPLIED] Updated {config_path.name}")
                    
        except Exception as e:
            logger.warning(f"Failed to process model {run_dir.name}: {e}")

def process_predictions(ws: Workspace, apply: bool):
    logger.info("\n--- Scanning Behavior Predictions ---")
    
    # Force scan to populate internal dict
    ws.scan_predictions()
    
    # Iterate: Run -> Dataset -> Video
    preds_map = ws.behavior_predictions._behavior_predictions
    
    for run_name, datasets in preds_map.items():
        for dataset_name, videos in datasets.items():
            for video_stem in videos.keys():
                
                # Lazy load check: If None, it exists on disk but isn't loaded
                # We load it to inspect it.
                try:
                    ds = ws.get_behavior_predictions(run_name, dataset_name, video_stem)
                except Exception as e:
                    logger.error(f"Skipping broken prediction {run_name}/{dataset_name}/{video_stem}: {e}")
                    continue
                
                if not ds: continue

                # Check 1: Metadata
                meta_dirty = _has_changes(ds.behaviors)
                
                # Check 2: DataFrame Columns (Probabilities)
                cols_rename = _get_rename_map(ds.per_frame_probabilities.columns)
                
                # Check 3: Bouts
                bouts_rename = {}
                if not ds.bouts.empty:
                    bouts_rename = _get_rename_map(ds.bouts['behavior'].unique())

                if meta_dirty or cols_rename or bouts_rename:
                    logger.info(f"Prediction '{run_name}/{video_stem}':")
                    if cols_rename:
                        logger.info(f"  - Renaming {len(cols_rename)} probability columns")

                    if apply:
                        # 1. Update List
                        ds.behaviors = sorted(list({normalize(b) for b in ds.behaviors}))
                        
                        # 2. Update Probabilities (Handle Collisions via Max)
                        if cols_rename:
                            df = ds.per_frame_probabilities
                            # Rename columns
                            df = df.rename(columns=cols_rename)
                            # If we have duplicate columns now (e.g. 'sit' and 'sit'), group by level and take max
                            # (Max is safer than sum for probabilities usually)
                            df = df.groupby(level=0, axis=1).max()
                            ds.per_frame_probabilities = df
                        
                        # 3. Update Bouts
                        if bouts_rename:
                            ds.bouts['behavior'] = ds.bouts['behavior'].replace(bouts_rename)
                            
                        # Save
                        ws.add_video_behavior_predictions(run_name, dataset_name, video_stem, ds)
                        logger.info("  [APPLIED] Saved.")

def main():
    parser = argparse.ArgumentParser(
        description="Normalize behavior names to snake_case across the Workspace."
    )
    parser.add_argument("workspace_dir", type=Path, help="Path to the Whisker workspace directory")
    parser.add_argument("--apply", action="store_true", help="Apply changes to disk (default is Dry Run)")
    
    args = parser.parse_args()
    
    if not args.workspace_dir.exists():
        logger.error(f"Workspace dir does not exist: {args.workspace_dir}")
        return

    logger.info(f"Loading Workspace: {args.workspace_dir}")
    if args.apply:
        logger.warning("MODE: APPLY (Files will be modified)")
    else:
        logger.info("MODE: DRY RUN (No changes will be saved)")

    try:
        ws = Workspace(args.workspace_dir)
    except Exception as e:
        logger.error(f"Failed to load workspace: {e}")
        return

    process_projects(ws, args.apply)
    process_labels(ws, args.apply)
    process_models(ws, args.apply)
    process_predictions(ws, args.apply)
    
    logger.info("\nDone!")

if __name__ == "__main__":
    main()