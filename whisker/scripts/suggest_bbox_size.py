import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from whisker.core.workspace import Workspace
from whisker.services.pose_estimation.public.data_structures import PoseDataset
from whisker.services.animal_detection.internal.core.calculate_bbox import calculate_instance_bbox

def get_file_path(workspace: str, dataset: str) -> PoseDataset:
    ws = Workspace(pathlib.Path(workspace))
    ds = ws.get_pose_dataset(dataset)
    if ds is None:
        raise ValueError(f"Dataset '{dataset}' not found in workspace '{workspace}'.")
    return ds

def main():
    parser = argparse.ArgumentParser(description="Suggest optimal square crop size.")
    parser.add_argument("workspace", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--conf", type=float, default=0.1)
    args = parser.parse_args()

    ds = get_file_path(args.workspace, args.dataset)
    
    if ds.keypoint_data.empty:
        print("Dataset is empty. No boxes to calculate.")
        return

    # Extract all (frame, individual) pairs present in the MultiIndex
    instances = ds.keypoint_data.index.droplevel('body_part').unique()
    
    max_dims = []
    for frame_idx, ind_id in instances:
        bbox = calculate_instance_bbox(
            ds.keypoint_data, 
            frame_idx, 
            ind_id, 
            margin=args.margin, 
            min_conf=args.conf
        )
        if bbox:
            # For a square crop to contain the bbox, side length must be >= max(w, h)
            max_dims.append(max(bbox.width, bbox.height))

    if not max_dims:
        print("No valid bounding boxes found with current constraints.")
        return

    max_dims = np.sort(max_dims)
    n = len(max_dims)
    y_percentile = np.arange(1, n + 1) / n * 100

    # Stats
    stats = {
        "Mean": np.mean(max_dims),
        "Median": np.median(max_dims),
        "90th %": np.percentile(max_dims, 90),
        "95th %": np.percentile(max_dims, 95),
        "99th %": np.percentile(max_dims, 99),
        "Max": max_dims[-1]
    }

    print(f"\n--- Distribution Stats for {args.dataset} ---")
    for k, v in stats.items():
        print(f"{k:8}: {v:.2f} px")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(max_dims, y_percentile, linestyle='-', marker='None', color='b')
    plt.axhline(95, color='r', linestyle='--', label='95% Coverage')
    plt.title(f"Cumulative Distribution of Instance Sizes ({args.dataset})")
    plt.xlabel("Square Crop Size (px)")
    plt.ylabel("% of Instances Fully Contained")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_plot = Path(f"{args.dataset}_bbox_dist.png")
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()