"""
MultiCue-DF Preprocessing Pipeline — 3 streams per frame.
Run from multicue-df/ root: python src/preprocess.py

Stream 1 — Full face crop via MTCNN (224×224)
Stream 2 — Eye + mouth region crops via MTCNN 5-point landmarks (224×224)
Stream 3 — FFT magnitude spectrum (224×224)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import json
import csv
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN

from config import CONFIG

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if (CONFIG["device"] == "auto" and torch.cuda.is_available()) else (
    CONFIG["device"] if CONFIG["device"] != "auto" else "cpu"
)

IMG_SIZE = CONFIG["image_size"]
PAD = 20  # pixels of padding around landmark bounding boxes

# ── MTCNN — used for both stream 1 and stream 2 ───────────────────────────────
# keep_all=False  → return best face only
# post_process=False → raw pixel values [0,255] as float tensor
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=False,
    device=DEVICE,
    post_process=False,
)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _detect(pil_img):
    """
    Run MTCNN detection once, return (face_tensor_or_None, landmarks_or_None).
    landmarks shape: (5, 2) — [left_eye, right_eye, nose, mouth_left, mouth_right]
    """
    try:
        boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)
    except Exception:
        return None, None

    if boxes is None or len(boxes) == 0:
        return None, None

    # Get the face crop tensor for stream 1
    try:
        face_tensor = mtcnn(pil_img)  # (3, H, W) float [0,255]
    except Exception:
        face_tensor = None

    lms = points[0]  # (5, 2) — best face landmarks
    return face_tensor, lms


# ── Stream 1 ─────────────────────────────────────────────────────────────────
def process_stream1(bgr_frame, face_tensor):
    """Full aligned face crop. Returns (224×224 BGR ndarray, fallback_used)."""
    if face_tensor is not None:
        face_np = face_tensor.cpu().permute(1, 2, 0).clamp(0, 255).byte().numpy()
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        return cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE)), False
    return cv2.resize(bgr_frame, (IMG_SIZE, IMG_SIZE)), True


# ── Stream 2 ─────────────────────────────────────────────────────────────────
def process_stream2(bgr_frame, lms):
    """
    Eye region + mouth region stacked vertically → 224×224.
    lms: (5,2) array — [left_eye, right_eye, nose, mouth_left, mouth_right]
    Falls back to black image if landmarks are absent.
    """
    if lms is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), True

    h, w = bgr_frame.shape[:2]

    def crop_region(pts):
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1 = _clamp(min(xs) - PAD, 0, w - 1)
        y1 = _clamp(min(ys) - PAD, 0, h - 1)
        x2 = _clamp(max(xs) + PAD, 0, w - 1)
        y2 = _clamp(max(ys) + PAD, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((20, 20, 3), dtype=np.uint8)
        return bgr_frame[y1:y2, x1:x2]

    # MTCNN 5-point order: left_eye(0), right_eye(1), nose(2), mouth_left(3), mouth_right(4)
    eye_crop   = crop_region([lms[0], lms[1]])
    mouth_crop = crop_region([lms[3], lms[4]])

    strip_w = IMG_SIZE
    eye_h   = max(1, int(eye_crop.shape[0]   * strip_w / max(1, eye_crop.shape[1])))
    mouth_h = max(1, int(mouth_crop.shape[0] * strip_w / max(1, mouth_crop.shape[1])))

    eye_r   = cv2.resize(eye_crop,   (strip_w, eye_h))
    mouth_r = cv2.resize(mouth_crop, (strip_w, mouth_h))

    stacked = np.vstack([eye_r, mouth_r])
    return cv2.resize(stacked, (IMG_SIZE, IMG_SIZE)), False


# ── Stream 3 ─────────────────────────────────────────────────────────────────
def process_stream3(bgr_frame):
    """FFT magnitude spectrum → 3-channel 224×224. Always succeeds."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft  = np.fft.fftshift(np.fft.fft2(gray))
    mag  = np.log1p(np.abs(fft))
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    mag8 = mag.astype(np.uint8)
    out  = cv2.merge([mag8, mag8, mag8])
    return cv2.resize(out, (IMG_SIZE, IMG_SIZE))


# ── File list ─────────────────────────────────────────────────────────────────
def build_file_list():
    frames_root    = PROJECT_ROOT / CONFIG["frames_dir"]
    processed_root = PROJECT_ROOT / "processed"
    entries = []

    real_src = frames_root / "real"
    for p in sorted(real_src.glob("*.jpg")):
        entries.append((
            p, 0, "original",
            processed_root / "stream1" / "real"         / p.name,
            processed_root / "stream2" / "real"         / p.name,
            processed_root / "stream3" / "real"         / p.name,
        ))

    for folder in CONFIG["fake_folders"]:
        fake_src = frames_root / "fake" / folder
        for p in sorted(fake_src.glob("*.jpg")):
            entries.append((
                p, 1, folder,
                processed_root / "stream1" / "fake" / folder / p.name,
                processed_root / "stream2" / "fake" / folder / p.name,
                processed_root / "stream3" / "fake" / folder / p.name,
            ))

    return entries


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n=== MultiCue-DF: Preprocessing Pipeline (device={DEVICE}) ===\n")

    entries = build_file_list()
    print(f"Total frames to process: {len(entries):,}\n")

    logs_dir      = PROJECT_ROOT / CONFIG["logs_dir"]
    manifest_path = PROJECT_ROOT / "processed" / "manifest.csv"
    logs_dir.mkdir(exist_ok=True)

    manifest_rows = []
    stats = {
        "total_processed":      0,
        "stream1_fallback":     0,
        "stream2_fallback":     0,
        "stream2_fallback_files": [],
        "per_class":            {},
    }

    for src_path, label, manip, s1_dst, s2_dst, s3_dst in tqdm(
        entries, desc="Preprocessing", unit="frame", ncols=100
    ):
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            continue

        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # Single MTCNN call — used by both stream 1 and stream 2
        face_tensor, lms = _detect(pil_img)

        s1_img, s1_fb = process_stream1(bgr, face_tensor)
        s2_img, s2_fb = process_stream2(bgr, lms)
        s3_img        = process_stream3(bgr)

        cv2.imwrite(str(s1_dst), s1_img)
        cv2.imwrite(str(s2_dst), s2_img)
        cv2.imwrite(str(s3_dst), s3_img)

        stats["total_processed"] += 1
        if s1_fb:
            stats["stream1_fallback"] += 1
        if s2_fb:
            stats["stream2_fallback"] += 1
            if len(stats["stream2_fallback_files"]) < 200:
                stats["stream2_fallback_files"].append(src_path.name)

        stats["per_class"][manip] = stats["per_class"].get(manip, 0) + 1

        manifest_rows.append({
            "filename":          src_path.name,
            "label":             label,
            "manipulation_type": manip,
            "stream1_path":      str(s1_dst.relative_to(PROJECT_ROOT)),
            "stream2_path":      str(s2_dst.relative_to(PROJECT_ROOT)),
            "stream3_path":      str(s3_dst.relative_to(PROJECT_ROOT)),
        })

    # Write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "label", "manipulation_type",
            "stream1_path", "stream2_path", "stream3_path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write stats
    stats_path = logs_dir / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary table
    print("\n" + "=" * 62)
    print(f"  {'Metric':<38} {'Value':>10}")
    print("=" * 62)
    print(f"  {'Total frames processed':<38} {stats['total_processed']:>10,}")
    print(f"  {'Stream1 fallbacks (no face detected)':<38} {stats['stream1_fallback']:>10,}")
    print(f"  {'Stream2 fallbacks (no landmarks)':<38} {stats['stream2_fallback']:>10,}")
    print("-" * 62)
    for cls, cnt in sorted(stats["per_class"].items()):
        print(f"  {cls:<38} {cnt:>10,}")
    print("=" * 62)
    print(f"\nManifest -> {manifest_path}")
    print(f"Stats    -> {stats_path}\n")


if __name__ == "__main__":
    main()
