# START_FILE: whisker/core/workflows/behavior_classification/ml/dataprep.py
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Generator

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from whisker.services.behavior_classification.public.data_structures import BehaviorDataset


def create_frame_wise_labels(
    behavior_dataset: BehaviorDataset,
    video_key: str,
    all_behaviors: List[str],
    num_frames: int,
) -> pd.DataFrame:
    """
    Converts bouts from a BehaviorDataset for a specific video into a
    frame-wise, multi-label binary DataFrame.
    """
    # Create the base DataFrame
    labels_df = pd.DataFrame(
        np.zeros((num_frames, len(all_behaviors)), dtype=np.float32),
        index=pd.RangeIndex(num_frames, name="frame_index"),
        columns=all_behaviors,
    )

    if behavior_dataset.bouts.empty:
        logging.info("DEBUG: dataprep: Bouts df is empty.")
        return labels_df

    # --- Robust Key Matching ---
    # 1. Try exact match first
    mask = behavior_dataset.bouts["video_key"] == video_key

    # 2. Fallback: Fuzzy match on filename stem
    if not mask.any():
        available_keys = behavior_dataset.bouts["video_key"].unique()
        target_stem = Path(video_key).stem

        candidate_key = next(
            (k for k in available_keys if Path(str(k)).stem == target_stem), 
            None
        )

        if candidate_key:
            mask = behavior_dataset.bouts["video_key"] == candidate_key

    # Apply mask
    video_bouts_df = behavior_dataset.bouts[mask]

    if video_bouts_df.empty:
        return labels_df

    # Fill in the 1s based on the bouts
    for row in video_bouts_df.itertuples():
        behavior_name = row.behavior
        if behavior_name not in all_behaviors:
            continue

        # Ensure bouts are within the video frame bounds
        start = max(0, int(row.start_frame))
        end = min(num_frames - 1, int(row.end_frame))
        if start <= end:
            # Use .loc to set the values for the frame range
            labels_df.loc[start : end, behavior_name] = 1.0

    return labels_df


def create_training_windows(
    features_df: pd.DataFrame, labels_df: pd.DataFrame, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy static window creation. Kept for backward compatibility and static validation.
    """
    if features_df.empty or labels_df.empty:
        return np.array([]), np.array([])

    features_aligned, labels_aligned = features_df.align(
        labels_df, join="inner", axis=0
    )

    if features_aligned.empty:
        return np.array([]), np.array([])

    features = features_aligned.values
    labels = labels_aligned.values
    num_frames, num_features = features.shape

    padding = np.zeros((window_size - 1, num_features), dtype=np.float32)
    features_padded = np.vstack([padding, features])

    shape = (num_frames, window_size, num_features)
    
    strides = (
        features_padded.strides[0],
        features_padded.strides[0],
        features_padded.strides[1],
    )
    
    X = np.lib.stride_tricks.as_strided(
        features_padded, shape=shape, strides=strides
    )
    y = labels

    return X, y


def build_behavior_anchors(sequences: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[int, int, int]]:
    """
    Scans sequences to find all contiguous behavior bouts.
    Returns a list of (video_index, start_frame, end_frame).
    """
    anchors = []
    for v_idx, (_, y_full) in enumerate(sequences):
        if y_full.size == 0: continue
        
        # Find any frame where at least one behavior is active
        # y_full shape: (T, num_behaviors)
        active_frames = np.any(y_full > 0.5, axis=1)
        
        # Simple boundary detection (1 = start, -1 = end)
        diffs = np.diff(active_frames.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        for start, end in zip(starts, ends):
            if end - start > 1:
                anchors.append((v_idx, start, end))
    return anchors


class AugmentedWindowGenerator:
    """
    A dynamic sequence generator for dense Action Segmentation.
    Implements Anchor Sampling and pre-padded slicing for high performance.
    """
    def __init__(
        self,
        sequences: List[Tuple[np.ndarray, np.ndarray]],
        window_size: int,
        batch_size: int,
        anchors: Optional[List[Tuple[int, int, int]]] = None,
        augment: bool = True,
        neg_to_pos_ratio: int = 1,
        proximity_masks: Optional[List[np.ndarray]] = None,
        proximity_neg_ratio: float = 0.5
    ):
        self.window_size = window_size
        self.batch_size = batch_size
        self.augment = augment
        self.neg_to_pos_ratio = max(1, neg_to_pos_ratio)
        self.proximity_neg_ratio = proximity_neg_ratio
        
        # 1. Store dimensions before discarding raw sequences
        self.num_features = sequences[0][0].shape[1]
        self.num_behaviors = sequences[0][1].shape[1]
        
        # 2. Build or use provided anchors
        if anchors is not None:
            self.positive_anchors = anchors
        else:
            self.positive_anchors = build_behavior_anchors(sequences)
            logging.info(f"Generator initialized with {len(self.positive_anchors)} positive behavior anchors.")

        # 3. Pre-pad all sequences to allow O(1) slicing without np.pad calls in flow()
        # We replace the original sequences to save memory.
        self.padded_sequences = []
        for x, y in sequences:
            # Mode 'constant' with value 0.0
            x_pad = np.pad(x, ((window_size, window_size), (0, 0)), mode='constant')
            y_pad = np.pad(y, ((window_size, window_size), (0, 0)), mode='constant')
            self.padded_sequences.append((x_pad, y_pad))

        # 4. Precompute valid negative window start indices (excluding positive bout periods)
        self.valid_neg_starts = []
        self.valid_prox_starts = []
        
        for v_idx, (x, y) in enumerate(sequences):
            T = x.shape[0]
            # Find frames where behavior is active
            active = np.any(y > 0.5, axis=1) # (T,)
            
            valid_starts = []
            prox_starts = []
            
            p_mask = proximity_masks[v_idx] if (proximity_masks is not None and len(proximity_masks) > v_idx) else None
            
            for t in range(0, T - window_size + 1):
                window_active = np.any(active[t : t + window_size])
                if not window_active:
                    valid_starts.append(t)
                    # Check center frame of the window for proximity
                    center_idx = t + window_size // 2
                    if p_mask is not None and center_idx < len(p_mask) and p_mask[center_idx]:
                        prox_starts.append(t)
            
            if not valid_starts:
                valid_starts = list(range(0, max(1, T - window_size + 1)))
                
            self.valid_neg_starts.append(np.array(valid_starts, dtype=np.int32))
            self.valid_prox_starts.append(np.array(prox_starts, dtype=np.int32))

    def flow(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Finite epoch generator. At the start of each epoch it builds a fixed plan:
          - One sample per positive anchor (with optional jitter)
          - neg_to_pos_ratio negatives per positive anchor (split into proximity/uniform)
        The plan is shuffled and yielded in complete batches. Incomplete trailing
        batches are discarded so the output_signature shape is always consistent.
        """
        num_vids = len(self.padded_sequences)

        while True:
            # ---- Build the epoch plan ----
            plan: List[Tuple[int, int]] = []  # (v_idx, chunk_start)

            # Positive anchors – one entry per anchor
            for v_idx, start, end in self.positive_anchors:
                bout_center = (start + end) // 2
                # Expanded jitter: attack onset can land anywhere valid within the chunk,
                # forcing shift-invariance rather than always centering attacks.
                half = self.window_size // 2
                jitter = (
                    np.random.randint(-half + 5, half - 5)
                    if self.augment else 0
                )
                center_padded = bout_center + self.window_size + jitter
                chunk_start = center_padded - (self.window_size // 2)
                # Clamp to valid padded range
                max_start = self.padded_sequences[v_idx][0].shape[0] - self.window_size
                chunk_start = int(np.clip(chunk_start, 0, max(0, max_start)))
                plan.append((v_idx, chunk_start))

            # Sample negatives split between Proximity and Uniform
            total_negs = len(self.positive_anchors) * self.neg_to_pos_ratio
            if len(self.positive_anchors) == 0:
                total_negs = self.batch_size

            num_prox_negs = int(total_negs * self.proximity_neg_ratio)
            num_uni_negs = total_negs - num_prox_negs

            # Sample proximity negatives (hard negatives)
            for _ in range(num_prox_negs):
                v_idx = np.random.randint(num_vids)
                prox_list = self.valid_prox_starts[v_idx]
                
                # Fallback to uniform if no proximity frames exist in this video
                if len(prox_list) == 0:
                    prox_list = self.valid_neg_starts[v_idx]
                
                start_orig = np.random.choice(prox_list)
                chunk_start = start_orig + self.window_size
                plan.append((v_idx, chunk_start))

            # Sample uniform negatives
            for _ in range(num_uni_negs):
                v_idx = np.random.randint(num_vids)
                neg_list = self.valid_neg_starts[v_idx]
                start_orig = np.random.choice(neg_list)
                chunk_start = start_orig + self.window_size
                plan.append((v_idx, chunk_start))

            random.shuffle(plan)

            # Ensure we have at least one complete batch if plan is small but non-empty
            if len(plan) > 0 and len(plan) < self.batch_size:
                repeats = (self.batch_size + len(plan) - 1) // len(plan)
                plan = (plan * repeats)[:self.batch_size]

            # ---- Yield complete batches ----
            for batch_start in range(0, len(plan) - self.batch_size + 1, self.batch_size):
                batch_items = plan[batch_start : batch_start + self.batch_size]

                X_batch = np.zeros((self.batch_size, self.window_size, self.num_features), dtype=np.float32)
                y_batch = np.zeros((self.batch_size, self.window_size, self.num_behaviors), dtype=np.float32)

                for i, (v_idx, chunk_start) in enumerate(batch_items):
                    X_batch[i] = self.padded_sequences[v_idx][0][chunk_start : chunk_start + self.window_size]
                    y_batch[i] = self.padded_sequences[v_idx][1][chunk_start : chunk_start + self.window_size]

                yield X_batch, y_batch

    def get_sample_timeline(self, v_idx: int) -> List[dict]:
        """
        Generates a sample plan for a single mock epoch and filters it for the given video.
        Returns a list of dicts with keys: 'start', 'end', 'type'.
        """
        samples = []
        
        # 1. Positive anchors for this video
        for pv_idx, start, end in self.positive_anchors:
            if pv_idx != v_idx:
                continue
            bout_center = (start + end) // 2
            half = self.window_size // 2
            center_padded = bout_center + self.window_size
            chunk_start = center_padded - half
            start_orig = chunk_start - self.window_size
            samples.append({
                "start": int(start_orig),
                "end": int(start_orig + self.window_size),
                "type": "positive"
            })
            
        # 2. Estimate negatives assigned to this video
        num_vids = len(self.padded_sequences)
        total_negs = len(self.positive_anchors) * self.neg_to_pos_ratio
        num_prox_negs = int(total_negs * self.proximity_neg_ratio)
        num_uni_negs = total_negs - num_prox_negs
        
        expected_prox = max(1, num_prox_negs // num_vids)
        expected_uni = max(1, num_uni_negs // num_vids)
        
        # Sample proximity negatives
        prox_list = self.valid_prox_starts[v_idx]
        if len(prox_list) > 0:
            # Avoid requesting more samples than available if replace=False
            replace = len(prox_list) < expected_prox
            sampled_prox = np.random.choice(prox_list, size=min(len(prox_list) if not replace else expected_prox, expected_prox), replace=replace)
            for start_orig in sampled_prox:
                samples.append({
                    "start": int(start_orig),
                    "end": int(start_orig + self.window_size),
                    "type": "proximity_negative"
                })
        else:
            # Fallback to uniform if no proximity frames
            prox_list = self.valid_neg_starts[v_idx]
            if len(prox_list) > 0:
                replace = len(prox_list) < expected_prox
                sampled_prox = np.random.choice(prox_list, size=min(len(prox_list) if not replace else expected_prox, expected_prox), replace=replace)
                for start_orig in sampled_prox:
                    samples.append({
                        "start": int(start_orig),
                        "end": int(start_orig + self.window_size),
                        "type": "proximity_negative"
                    })
                
        # Sample uniform negatives
        neg_list = self.valid_neg_starts[v_idx]
        if len(neg_list) > 0:
            replace = len(neg_list) < expected_uni
            sampled_uni = np.random.choice(neg_list, size=min(len(neg_list) if not replace else expected_uni, expected_uni), replace=replace)
            for start_orig in sampled_uni:
                samples.append({
                    "start": int(start_orig),
                    "end": int(start_orig + self.window_size),
                    "type": "uniform_negative"
                })
                
        return samples