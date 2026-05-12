"""
Frame extraction script for MultiCue-DF.
Run from multicue-df/ root: python src/extract_frames.py
"""

import sys
import os
from pathlib import Path

# Resolve project root as parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import random
import json
import numpy as np
from tqdm import tqdm
from config import CONFIG

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def get_evenly_spaced_indices(total_frames: int, n: int) -> list[int]:
    return [int(total_frames * i / n) for i in range(n)]


def extract_frames_from_video(video_path: Path, output_dir: Path, video_stem: str,
                               n_frames: int) -> tuple[int, str | None]:
    """
    Extract n_frames evenly spaced frames from video_path into output_dir.
    Returns (frames_saved, error_message_or_None).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, f"Cannot open: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0, f"Zero frames: {video_path}"

    indices = get_evenly_spaced_indices(total_frames, n_frames)
    saved = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = output_dir / f"{video_stem}_frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved, None


def process_class(video_dir: Path, output_dir: Path, label: str,
                  n_videos: int, n_frames: int) -> tuple[int, list[dict]]:
    """
    Sample n_videos from video_dir, extract n_frames each into output_dir.
    Returns (total_frames_saved, skipped_list).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(video_dir.glob("*.mp4"))
    if len(all_videos) < n_videos:
        print(f"  [WARNING] {label}: only {len(all_videos)} videos available, using all.")
        sampled = all_videos
    else:
        sampled = random.sample(all_videos, n_videos)

    total_saved = 0
    skipped = []

    for video_path in tqdm(sampled, desc=f"  {label}", unit="vid", ncols=90):
        saved, err = extract_frames_from_video(
            video_path, output_dir, video_path.stem, n_frames
        )
        if err:
            skipped.append({"video": str(video_path), "reason": err})
        else:
            total_saved += saved

    return total_saved, skipped


def main():
    dataset_root = PROJECT_ROOT / CONFIG["dataset_root"]
    frames_root  = PROJECT_ROOT / CONFIG["frames_dir"]
    logs_dir     = PROJECT_ROOT / CONFIG["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    n_videos = CONFIG["videos_per_class"]
    n_frames = CONFIG["frames_per_video"]

    summary = {}
    all_skipped = []

    print("\n=== MultiCue-DF: Frame Extraction ===\n")

    # --- Real class ---
    real_src = dataset_root / CONFIG["real_folder"]
    real_dst = frames_root / "real"
    saved, skipped = process_class(real_src, real_dst, "original (real)", n_videos, n_frames)
    summary["original"] = saved
    all_skipped.extend(skipped)

    # --- Fake classes ---
    for folder in CONFIG["fake_folders"]:
        fake_src = dataset_root / folder
        fake_dst = frames_root / "fake" / folder
        saved, skipped = process_class(fake_src, fake_dst, f"{folder} (fake)", n_videos, n_frames)
        summary[folder] = saved
        all_skipped.extend(skipped)

    # Save skipped log
    skipped_path = logs_dir / "skipped_videos.json"
    with open(skipped_path, "w") as f:
        json.dump(all_skipped, f, indent=2)

    # Print summary table
    print("\n" + "=" * 52)
    print(f"{'Class':<25} {'Frames Extracted':>16}")
    print("=" * 52)
    total = 0
    for cls, count in summary.items():
        print(f"  {cls:<23} {count:>16,}")
        total += count
    print("-" * 52)
    print(f"  {'TOTAL':<23} {total:>16,}")
    print(f"  {'Skipped videos':<23} {len(all_skipped):>16,}")
    print("=" * 52)
    print(f"\nSkipped log saved to: {skipped_path}\n")


if __name__ == "__main__":
    main()
