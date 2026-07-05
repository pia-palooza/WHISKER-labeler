import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path to allow imports when run standalone
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Subtract background from a video using median subtraction.")
    parser.add_argument("input_filename", type=str, help="Path to the input video file")
    parser.add_argument("output_filename", type=str, help="Path to the output video file")
    parser.add_argument("--threshold", type=int, default=25, help="Difference threshold (default: 25)")

    args = parser.parse_args()
    input_path = Path(args.input_filename)
    output_path = Path(args.output_filename)
    diff_threshold = args.threshold

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Computing median background for '{input_path.name}'...")
    bg_gray = compute_median_background(str(input_path))
    if bg_gray is None:
        print("Error: Failed to compute median background. The video may be corrupt or empty.")
        sys.exit(1)

    # Save the background image for inspection
    bg_gray = ensure_uint8_gray(bg_gray)
    bg_img_path = output_path.parent / f"{output_path.stem}_background.png"
    cv2.imwrite(str(bg_img_path), bg_gray)
    print(f"Saved computed background image to '{bg_img_path}'")

    print("Background computed successfully. Subtracting background and writing output video...")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("Error: Could not open input video.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not open output video writer.")
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Grayscale and data type conversion safety
        gray = ensure_uint8_gray(frame)
        
        # Subtraction with Gaussian smoothing and morphological opening
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(bg_gray, gray_blurred)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Convert back to BGR for video writer compatibility
        color_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        out.write(color_thresh)

        frame_idx += 1
        if frame_idx % max(1, total_frames // 10) == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {frame_idx}/{total_frames} frames processed ({pct:.1f}%)")

    cap.release()
    out.release()
    print(f"Success! Saved background subtracted video to '{output_path}'")

if __name__ == "__main__":
    main()
