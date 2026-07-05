import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

# Maximum image dimension for cached video thumbnail images
# Limit image sizes for storage size considerations.
MAX_VIDEO_THUMBNAIL_CACHE_IMAGE_DIM = 256


def _get_frame_variance(frame: np.ndarray) -> float:
    """Calculates the variance of the Laplacian of a frame."""
    # Convert to grayscale and compute the Laplacian
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # The variance of the Laplacian is a good measure of focus/detail
    return laplacian.var()


def generate_video_thumbnail(video_path: Path, cache_dir: Path) -> Path | None:
    """
    Core function to generate a thumbnail for a video by finding the most "interesting" frame.
    Returns the path to the generated or cached PNG thumbnail, or None on failure.
    """
    if not video_path.exists():
        return None

    # Create a unique, filesystem-safe filename for the cache
    path_str = str(video_path.resolve())
    hash_id = hashlib.md5(path_str.encode()).hexdigest()
    cache_file = cache_dir / f"{hash_id}.png"

    # Try loading from cache first
    if cache_file.exists():
        logging.debug(f"Loading thumbnail for {video_path.name} from cache.")
        return cache_file

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path.name}")
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            sample_indices = [0]
        else:
            sample_indices = [
                int(total_frames * 0.25),
                int(total_frames * 0.50),
                int(total_frames * 0.75),
            ]

        best_frame = None
        max_variance = -1.0

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                variance = _get_frame_variance(frame)
                if variance > max_variance:
                    max_variance = variance
                    best_frame = frame

        if best_frame is not None:
            rgb_image = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            max_dim = MAX_VIDEO_THUMBNAIL_CACHE_IMAGE_DIM
            scale = min(max_dim / w, max_dim / h, 1.0)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_image = cv2.resize(
                    rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            cache_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_file), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            logging.debug(f"Generated and cached thumbnail for {video_path.name}")
            return cache_file

    except Exception as e:
        logging.error(f"Error generating thumbnail for {video_path.name}: {e}")
    finally:
        cap.release()

    return None
