import logging 
from typing import List, Dict, Any
import pandas as pd

from whisker.services.behavior_classification.public.data_structures import BoutExtractionParams

def extract_bouts(
    df: pd.DataFrame,
    classifier_names: List[str],
    fps: float,
    params: BoutExtractionParams
) -> pd.DataFrame:
    all_bouts_for_file: List[Dict[str, Any]] = []

    min_bout_frames = round(params.min_bout_duration_sec * fps)
    max_gap_frames = round(params.max_gap_fill_sec * fps)

    if min_bout_frames < 1 and params.min_bout_duration_sec > 0: 
        min_bout_frames = 1

    # --- THIS IS THE FIX ---
    # The incoming df has a string index like 'path/to/video.mp4/frame_000123'
    # We need a numeric index (123) for bout detection.
    df_numeric = df.copy()
    if not pd.api.types.is_numeric_dtype(df_numeric.index) and not df_numeric.index.empty:
        try:
            # 1. Split on '/' to get the last part: 'frame_000123'
            frame_part = df.index.str.split('/').str[-1]
            # 2. Split that part on '_' to get the number: '000123'
            frame_num_str = frame_part.str.split('_').str[-1]
            # 3. Convert to integer
            df_numeric.index = frame_num_str.astype(int)
            df_numeric.index.name = "frame_number"
        except (AttributeError, ValueError, TypeError) as e:
            logging.error(
                f"Could not parse numeric frame numbers from DataFrame index. "
                f"Index format may be unexpected. Error: {e}"
            )
            # Proceed with the original df; it will fail, but this logs the error.
            df_numeric = df
    # --- END FIX ---

    for classifier_name in classifier_names:
        prob_col_name = classifier_name.strip()
        
        # Pass the numerically-indexed df to detect_bouts
        initial_bouts = detect_bouts(
            df_numeric, # <-- Pass the modified one
            prob_col_name,
            params.probability_threshold,
            min_bout_frames,
            classifier_name.strip()
        )

        if not initial_bouts:
            logging.debug(f"    No initial bouts found for classifier '{classifier_name.strip()}'") 
            continue
        
        # Second pass: Merge bouts
        final_bouts_for_classifier = merge_bouts(initial_bouts, max_gap_frames)
        
        all_bouts_for_file.extend(final_bouts_for_classifier)

    if not all_bouts_for_file:
        logging.debug(f"    No bouts found for any specified classifier in") 
        return pd.DataFrame([])

    bouts_df = pd.DataFrame(all_bouts_for_file)
    bouts_df['duration_s'] = (bouts_df['end_frame'] - bouts_df['start_frame'] + 1) / fps

    # Sort for consistent output
    bouts_df = bouts_df.sort_values(by=['behavior', 'start_frame']).reset_index(drop=True)

    return bouts_df

def detect_bouts(
    df: pd.DataFrame,
    prob_col_name: str,
    probability_threshold: float,
    min_bout_frames: int,
    classifier_name: str
) -> List[Dict[str, Any]]:
    """
    First pass to detect bouts based on probability and minimum duration.
    A bout is defined by start_frame and end_frame (inclusive).
    Assumes df has a numeric index.
    """
    potential_bouts = []
    if prob_col_name not in df.columns:
        logging.warning(f"    Warning: Probability column '{prob_col_name}' not found. Skipping for this classifier.") 
        return potential_bouts

    is_above_threshold = df[prob_col_name] >= probability_threshold

    if not is_above_threshold.any():
        return potential_bouts

    true_block_groups = is_above_threshold.ne(is_above_threshold.shift()).cumsum()[is_above_threshold]

    for _, group_indices in df.loc[true_block_groups.index].groupby(true_block_groups):
        if not group_indices.empty:
            start_frame = group_indices.index.min() # This will now be an int
            end_frame = group_indices.index.max() # This will now be an int
            
            num_frames = end_frame - start_frame + 1 # This will now be int - int
            if num_frames >= min_bout_frames:
                potential_bouts.append({
                    "behavior": classifier_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })
    return potential_bouts

def merge_bouts(
    bouts: List[Dict[str, Any]],
    max_gap_frames: int
) -> List[Dict[str, Any]]:
    """
    Second pass to merge bouts that are close together.
    Assumes bouts are for the same behavior and sorted by start_frame.
    """
    if not bouts:
        return []

    sorted_bouts = sorted(bouts, key=lambda b: b['start_frame'])
    
    merged_bouts: List[Dict[str, Any]] = []
    if not sorted_bouts: 
        return []

    merged_bouts.append(dict(sorted_bouts[0]))

    for i in range(1, len(sorted_bouts)):
        last_merged_bout = merged_bouts[-1]
        current_bout = sorted_bouts[i]

        gap_frames = current_bout['start_frame'] - last_merged_bout['end_frame'] - 1

        if 0 <= gap_frames <= max_gap_frames:
            last_merged_bout['end_frame'] = current_bout['end_frame']
        else:
            merged_bouts.append(dict(current_bout))
            
    return merged_bouts
