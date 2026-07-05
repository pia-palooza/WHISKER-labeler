import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, Any, List, Optional
from whisker.core.workspace import Workspace

def _get_gt_bouts_for_stem(gt_ds, stem: str):
    """Match GT bouts for a video stem, with fallback to stem-only matching (handles different extensions/paths)."""
    mask = gt_ds.bouts['video_key'].isin([stem, stem + ".mp4"])
    if not mask.any():
        target_stem = Path(stem).stem
        available_keys = gt_ds.bouts['video_key'].unique()
        candidate_key = next(
            (k for k in available_keys if Path(str(k)).stem == target_stem),
            None
        )
        if candidate_key:
            mask = gt_ds.bouts['video_key'] == candidate_key
    return gt_ds.bouts[mask]

def gather_dataset_video_data(workspace: Workspace, model_run_name: str, dataset_name: str, video_stems: List[str]) -> List[Dict[str, Any]]:
    gathered = []
    for stem in sorted(video_stems):
        gt_ds = workspace.behavior_labels.get_behavior_labels(dataset_name)
        pred_ds = workspace.get_behavior_predictions(model_run_name, dataset_name, stem)
        v_len = 0
        if pred_ds and not pred_ds.per_frame_probabilities.empty: v_len = len(pred_ds.per_frame_probabilities)
        elif gt_ds and not gt_ds.bouts.empty:
            v_bouts = _get_gt_bouts_for_stem(gt_ds, stem)
            if not v_bouts.empty: v_len = int(v_bouts['end_frame'].max()) + 1
        if v_len == 0: continue

        video_entry = {"stem": stem, "len": v_len, "gt_bouts": [], "pred_bouts": [], "probs": {}}
        if gt_ds:
            video_entry["gt_bouts"] = _get_gt_bouts_for_stem(gt_ds, stem).to_dict('records')
        if pred_ds:
            video_entry["pred_bouts"] = pred_ds.bouts.to_dict('records')
            if not pred_ds.per_frame_probabilities.empty:
                for col in pred_ds.per_frame_probabilities.columns:
                    video_entry["probs"][col] = pred_ds.per_frame_probabilities[col].values
        gathered.append(video_entry)
    return gathered

def render_dataset_behavior_chart(
    options: Dict[str, Any], 
    model_name: str,
    video_data: List[Dict[str, Any]],
    fig: Optional[Figure] = None
) -> Figure:
    behavior = options["behavior"]
    normalize = options["normalize"]
    hide_gt, hide_preds, hide_probs = options["hide_gt"], options["hide_preds"], options["hide_probs"]
    gt_color = options.get("gt_color", 'tab:orange')
    pred_color = options.get("pred_color", 'tab:green')
    prob_cmap = options.get("prob_cmap", 'viridis')

    tracks = []
    if not hide_gt: tracks.append(("gt", mcolors.ListedColormap([(0,0,0,0), gt_color]), "GT"))
    if not hide_preds: tracks.append(("pred", mcolors.ListedColormap([(0,0,0,0), pred_color]), "Pred"))
    if not hide_probs: tracks.append(("prob", prob_cmap, "Prob"))
    if not tracks: raise ValueError("At least one data track must be visible.")

    selected_v_stems = options.get("selected_videos", [])
    v_data_to_plot = video_data
    if selected_v_stems:
        v_data_to_plot = [v for v in video_data if v["stem"] in selected_v_stems]
    
    if not v_data_to_plot:
         if fig is not None: fig.clear()
         return fig

    n_videos = len(v_data_to_plot)
    n_tracks = len(tracks)
    track_h, v_gap = 0.1, 0.1 
    fig_h = 1.5 + (n_tracks * track_h + v_gap) * n_videos
    
    if fig is None: fig = Figure(figsize=(14, fig_h))
    else: fig.clear(); fig.set_figheight(fig_h)
    
    ax = fig.add_subplot(111)
    y_ticks, y_labels = [], []

    for v_idx, v_data in enumerate(v_data_to_plot):
        v_len = v_data["len"]
        v_base_y = v_idx * (n_tracks * track_h + v_gap)
        x = np.linspace(0, 1, v_len+1) if normalize else np.arange(v_len+1)
        
        rect_w = 1.0 if normalize else v_len
        ax.add_patch(patches.Rectangle((0, v_base_y), rect_w, n_tracks * track_h, color='lightgray', alpha=0.3, lw=0, zorder=0))

        for t_idx, (key, cmap, _) in enumerate(tracks):
            y_s, y_e = v_base_y + t_idx * track_h, v_base_y + (t_idx + 1) * track_h
            vals = np.zeros(v_len)
            if key == "gt":
                for b in v_data["gt_bouts"]:
                    if b["behavior"] == behavior: vals[int(b["start_frame"]):int(b["end_frame"])] = 1
            elif key == "pred":
                if v_data["pred_bouts"]:
                    for b in v_data["pred_bouts"]:
                        if b["behavior"] == behavior: vals[int(b["start_frame"]):int(b["end_frame"])] = 1
                else:
                    vals = v_data["probs"].get(f"{behavior}_binary", np.zeros(v_len))
            elif key == "prob":
                vals = v_data["probs"].get(behavior, np.zeros(v_len))

            ax.pcolormesh(x, [y_s, y_e], vals.reshape(1, -1), cmap=cmap, vmin=0, vmax=1, shading='flat', zorder=1)

        y_ticks.append(v_base_y + (n_tracks * track_h) / 2)
        y_labels.append(v_data["stem"][:25])

    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Normalized Time (0.0 - 1.0)" if normalize else "Frames")
    ax.set_title(f"Behavior: {behavior} | Model: {model_name}", pad=15)
    return fig
