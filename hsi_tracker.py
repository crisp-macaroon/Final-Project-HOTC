"""
HSI Object Tracker  --  Frame-Difference Blob Tracker with Velocity Prediction
===============================================================================
Designed for fast-moving objects in hyperspectral/NIR image sequences where:
  - Object moves too fast for optical flow or windowed NCC (80-110 px/frame)
  - Object appearance is not globally unique (NCC/mean-shift fail)
  - But the object creates strong pixel differences between consecutive frames
    (ball diff mean ~143 vs background ~7.75 in this HSI dataset)

Algorithm
---------
For each frame t:
  1. Compute |frame[t] - frame[t-1]| in the raw 16-bit space (maximum precision).
  2. Threshold the difference image to get a binary motion mask.
     Threshold = max(fixed_thresh, percentile(diff, thresh_pct)).
  3. Morphological close+open to merge fragmented blobs.
  4. Find all connected components (blobs) in the motion mask.
  5. Filter blobs by area (within [0.15, 5.0] x initial ball area).
  6. Among qualifying blobs, select the one closest to the VELOCITY-PREDICTED
     position (constant-velocity Kalman-like prediction).
  7. The selected blob's centroid is corrected for the "midpoint bias" caused
     by the difference image straddling old and new ball positions.
  8. Update velocity via exponential smoothing.
  9. Fallback: if no qualifying blob, advance by the current velocity estimate.

Data loading mirrors overlay_gt_boxes.py exactly:
  - Frames     : folder of 16-bit grayscale PNGs, sorted naturally
  - Init box   : init_rect.txt  (x y w h, tab/comma/space separated)
  - GT file    : groundtruth file with one x y w h per line (optional)
  - Scale      : --scale_x / --scale_y (default 5.0)

Usage:
    python hsi_tracker.py <frames_dir> <init_rect.txt> <output.mp4>
                          [--gt gt.txt] [--fps 25]
                          [--scale_x 5] [--scale_y 5]
                          [--threshold 60] [--thresh_pct 97]
                          [--vel_alpha 0.6] [--max_search_dist 600]

Requirements:
    pip install opencv-python numpy pillow --break-system-packages
"""

import sys
import os
import re
import glob
import argparse
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# I/O helpers  (identical logic to overlay_gt_boxes.py)
# ---------------------------------------------------------------------------

def natural_sort_key(path):
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', os.path.basename(path))]


def load_frame(path):
    return np.array(Image.open(path))


def compute_global_range(frame_paths, sample_every=5, low_pct=0.5, high_pct=99.5):
    """Percentile-based contrast-stretch range over a subset of frames."""
    samples = []
    for p in frame_paths[::sample_every]:
        arr = load_frame(p)
        samples.append(arr.astype(np.float32).ravel())
    all_vals = np.concatenate(samples)
    return np.percentile(all_vals, low_pct), np.percentile(all_vals, high_pct)


def stretch_to_uint8(arr, lo, hi):
    arr = arr.astype(np.float32)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def parse_box_line(line):
    """
    Parse one line into (x, y, w, h) floats.
    Handles comma, tab, or space-separated values.
    Returns None for empty / invalid / sentinel lines.
    """
    line = line.strip()
    if not line:
        return None
    parts = re.split(r'[,\s]+', line)
    try:
        vals = [float(v) for v in parts if v]
    except ValueError:
        return None
    if len(vals) >= 4:
        x, y, w, h = vals[:4]
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h
    return None


def load_init_box(init_path, scale_x=1.0, scale_y=1.0):
    """Read the first valid line of init_rect.txt and return pixel coords."""
    with open(init_path, 'r') as f:
        for line in f:
            box = parse_box_line(line)
            if box is not None:
                x, y, w, h = box
                return (int(round(x * scale_x)),
                        int(round(y * scale_y)),
                        int(round(w * scale_x)),
                        int(round(h * scale_y)))
    raise ValueError(f"No valid bounding box found in {init_path}")


def load_groundtruth(gt_path, scale_x=1.0, scale_y=1.0):
    """Load all GT boxes (optional, for evaluation)."""
    boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            box = parse_box_line(line)
            if box is not None:
                x, y, w, h = box
                boxes.append((int(round(x * scale_x)),
                               int(round(y * scale_y)),
                               int(round(w * scale_x)),
                               int(round(h * scale_y))))
            else:
                boxes.append(None)
    return boxes


# ---------------------------------------------------------------------------
# Frame-difference blob tracker
# ---------------------------------------------------------------------------

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def compute_diff_mask(prev_raw, curr_raw, threshold, thresh_pct):
    """
    Build a binary motion mask from the absolute difference of two 16-bit frames.
    threshold : hard minimum threshold value
    thresh_pct: adaptive component – threshold = max(threshold,
                                                     percentile(diff, thresh_pct))
    """
    diff = np.abs(curr_raw.astype(np.int32) - prev_raw.astype(np.int32)).astype(np.float32)
    adaptive_thresh = float(np.percentile(diff, thresh_pct))
    final_thresh = max(threshold, adaptive_thresh)
    binary = (diff > final_thresh).astype(np.uint8)

    # Morphological close (fills holes) then open (removes speckle)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  MORPH_KERNEL)
    return binary, diff, final_thresh


def find_best_blob(binary, diff, predicted_cx, predicted_cy,
                   min_area, max_area, max_search_dist):
    """
    Find the blob in `binary` that:
      1. Has area within [min_area, max_area]
      2. Is within max_search_dist of (predicted_cx, predicted_cy)
      3. Among those, is closest to the predicted position
         (tie-broken by blob's average diff magnitude -- brighter = more ball-like)

    Returns (cx, cy, confidence) of the best blob, or None if none qualify.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    if num_labels <= 1:
        return None

    best = None
    best_dist = float('inf')
    best_conf = 0.0

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        cx_b, cy_b = float(centroids[lbl][0]), float(centroids[lbl][1])
        dist = np.sqrt((cx_b - predicted_cx)**2 + (cy_b - predicted_cy)**2)

        if dist > max_search_dist:
            continue

        if dist < best_dist:
            best_dist = dist
            blob_pixels = diff[labels == lbl]
            best_conf = float(blob_pixels.mean())
            best = (cx_b, cy_b)

    return (best[0], best[1], best_conf) if best is not None else None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def centre_error(box_pred, box_gt):
    px = box_pred[0] + box_pred[2] / 2.0
    py = box_pred[1] + box_pred[3] / 2.0
    gx = box_gt[0] + box_gt[2] / 2.0
    gy = box_gt[1] + box_gt[3] / 2.0
    return float(np.sqrt((px - gx)**2 + (py - gy)**2))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(frames_dir, init_path, output_path,
         gt_path=None,
         fps=25,
         scale_x=5.0, scale_y=5.0,
         threshold=60,
         thresh_pct=97.0,
         vel_alpha=0.6,
         max_search_dist=600,
         pred_color=(0, 255, 0),   # green – predicted
         gt_color=(0, 0, 255),     # red   – ground-truth
         thickness=2):

    # ------------------------------------------------------------------
    # 1. Load frame list
    # ------------------------------------------------------------------
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")),
                         key=natural_sort_key)
    if not frame_paths:
        print(f"ERROR: No PNG frames found in {frames_dir}")
        sys.exit(1)
    print(f"Found {len(frame_paths)} frames.")

    # ------------------------------------------------------------------
    # 2. Load initial bounding box
    # ------------------------------------------------------------------
    init_box = load_init_box(init_path, scale_x=scale_x, scale_y=scale_y)
    ix, iy, iw, ih = init_box
    print(f"Initial box (pixel coords): x={ix}, y={iy}, w={iw}, h={ih}")

    ball_area = iw * ih
    min_area  = int(ball_area * 0.15)
    max_area  = int(ball_area * 5.0)   # diff blob = ~2x ball area (old + new position)
    print(f"Blob area filter: [{min_area}, {max_area}] px²  (ball area = {ball_area})")

    # ------------------------------------------------------------------
    # 3. Optionally load ground-truth
    # ------------------------------------------------------------------
    gt_boxes = None
    if gt_path:
        gt_boxes = load_groundtruth(gt_path, scale_x=scale_x, scale_y=scale_y)
        print(f"Loaded {len(gt_boxes)} ground-truth boxes from {gt_path}")

    # ------------------------------------------------------------------
    # 4. Compute contrast-stretch range for display (same as overlay_gt_boxes.py)
    # ------------------------------------------------------------------
    print("Computing contrast-stretch range for display...")
    lo, hi = compute_global_range(frame_paths)
    print(f"  lo={lo:.1f}, hi={hi:.1f}")

    # ------------------------------------------------------------------
    # 5. Initialise tracker state
    # ------------------------------------------------------------------
    cx = float(ix + iw / 2.0)
    cy = float(iy + ih / 2.0)
    bw, bh = iw, ih
    vx, vy = 0.0, 0.0   # velocity estimate (pixels per frame)

    first_raw = load_frame(frame_paths[0])
    fh, fw    = first_raw.shape[:2]

    # ------------------------------------------------------------------
    # 6. Set up video writer (using 8-bit stretched frames for display)
    # ------------------------------------------------------------------
    first_u8  = stretch_to_uint8(first_raw, lo, hi)
    vid_h, vid_w = (first_u8.shape[:2] if first_u8.ndim == 3 else first_u8.shape)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h), isColor=True)

    # ------------------------------------------------------------------
    # 7. Tracking loop
    # ------------------------------------------------------------------
    ious_list, cerrors, predicted_boxes = [], [], []
    prev_raw = first_raw.copy()
    fallback_count = 0

    for i, fpath in enumerate(frame_paths):
        curr_raw = load_frame(fpath)
        curr_u8  = stretch_to_uint8(curr_raw, lo, hi)

        if curr_u8.ndim == 3:
            frame_bgr = cv2.cvtColor(curr_u8, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = cv2.cvtColor(curr_u8, cv2.COLOR_GRAY2BGR)

        conf = 0.0
        used_fallback = False

        if i == 0:
            # First frame: use initial box exactly
            new_cx, new_cy = cx, cy
        else:
            # ---- Velocity-predicted position ----
            pred_cx = cx + vx
            pred_cy = cy + vy

            # Clamp prediction to frame
            pred_cx = float(np.clip(pred_cx, bw / 2, fw - bw / 2))
            pred_cy = float(np.clip(pred_cy, bh / 2, fh - bh / 2))

            # ---- Frame difference + blob detection ----
            binary, diff, used_thresh = compute_diff_mask(
                prev_raw, curr_raw, threshold, thresh_pct)

            result = find_best_blob(
                binary, diff, pred_cx, pred_cy,
                min_area, max_area, max_search_dist)

            if result is not None:
                blob_cx, blob_cy, conf = result

                # The diff-blob centroid is biased toward the midpoint between
                # the old and new ball positions. Correct by adding half the
                # velocity (velocity prediction is in the direction of motion).
                correction_x = vx * 0.5
                correction_y = vy * 0.5
                new_cx = blob_cx + correction_x
                new_cy = blob_cy + correction_y

                # Clamp to frame
                new_cx = float(np.clip(new_cx, bw / 2, fw - bw / 2))
                new_cy = float(np.clip(new_cy, bh / 2, fh - bh / 2))

                # Update velocity with exponential smoothing
                measured_vx = new_cx - cx
                measured_vy = new_cy - cy
                vx = vel_alpha * measured_vx + (1 - vel_alpha) * vx
                vy = vel_alpha * measured_vy + (1 - vel_alpha) * vy
            else:
                # Fallback: coast on current velocity
                new_cx = pred_cx
                new_cy = pred_cy
                used_fallback = True
                fallback_count += 1
                # Slightly decay velocity on fallback (uncertainty grows)
                vx *= 0.85
                vy *= 0.85

        cx, cy = new_cx, new_cy
        prev_raw = curr_raw.copy()

        # Predicted box
        px_box = int(round(cx - bw / 2))
        py_box = int(round(cy - bh / 2))
        pred_box = (px_box, py_box, bw, bh)
        predicted_boxes.append(pred_box)

        # ---- Draw predicted box ----
        color = (0, 165, 255) if used_fallback else pred_color   # orange if fallback
        cv2.rectangle(frame_bgr,
                      (px_box, py_box), (px_box + bw, py_box + bh),
                      color, thickness)
        status = "fallback" if used_fallback else f"conf={conf:.0f}"
        cv2.putText(frame_bgr,
                    f"f{i:04d}  vx={vx:.1f} vy={vy:.1f}  {status}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

        # ---- Draw GT box & compute metrics ----
        if gt_boxes is not None and i < len(gt_boxes) and gt_boxes[i] is not None:
            gx, gy, gw, gh = gt_boxes[i]
            cv2.rectangle(frame_bgr,
                          (gx, gy), (gx + gw, gy + gh),
                          gt_color, thickness)
            cv2.putText(frame_bgr, "GT",
                        (gx, max(gy - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, gt_color, 1, cv2.LINE_AA)

            iou_val = iou(pred_box, gt_boxes[i])
            ce_val  = centre_error(pred_box, gt_boxes[i])
            ious_list.append(iou_val)
            cerrors.append(ce_val)

            cv2.putText(frame_bgr,
                        f"IoU={iou_val:.2f}  CE={ce_val:.1f}px",
                        (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 0), 1, cv2.LINE_AA)

        writer.write(frame_bgr)

        if i % 50 == 0:
            iou_str = f"  IoU={ious_list[-1]:.3f}" if ious_list else ""
            print(f"  frame {i:4d}/{len(frame_paths)}  "
                  f"cx={cx:.1f} cy={cy:.1f}  vx={vx:.1f} vy={vy:.1f}  "
                  f"conf={conf:.0f}  fb={used_fallback}{iou_str}")

    writer.release()

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print(f"\nDone. Output video: {output_path}")
    print(f"  Fallback frames (velocity coasting): {fallback_count}/{len(frame_paths)}")
    if ious_list:
        arr_iou = np.array(ious_list)
        arr_ce  = np.array(cerrors)
        print(f"\n=== Tracking Evaluation ===")
        print(f"  Frames evaluated  : {len(ious_list)}")
        print(f"  Mean IoU          : {np.mean(arr_iou):.4f}")
        print(f"  Median IoU        : {np.median(arr_iou):.4f}")
        print(f"  IoU > 0.50        : {np.mean(arr_iou > 0.50)*100:.1f}%")
        print(f"  IoU > 0.25        : {np.mean(arr_iou > 0.25)*100:.1f}%")
        print(f"  Mean Centre Err   : {np.mean(arr_ce):.2f} px")
        print(f"  Median Centre Err : {np.median(arr_ce):.2f} px")

    pred_txt = os.path.splitext(output_path)[0] + "_predicted_boxes.txt"
    with open(pred_txt, 'w') as f:
        for b in predicted_boxes:
            f.write(f"{b[0]}\t{b[1]}\t{b[2]}\t{b[3]}\n")
    print(f"  Predicted boxes   : {pred_txt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HSI Object Tracker - Frame-difference blob tracker with velocity prediction")

    parser.add_argument("frames_dir",
                        help="Folder containing PNG frames (natural-sorted)")
    parser.add_argument("init_rect",
                        help="init_rect.txt - initial bounding box (x y w h)")
    parser.add_argument("output",
                        help="Output video path, e.g. tracked.mp4")
    parser.add_argument("--gt", default=None,
                        help="(Optional) groundtruth.txt for evaluation")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--scale_x", type=float, default=5.0,
                        help="Scale: init-box/GT x-coord -> pixel (default 5.0)")
    parser.add_argument("--scale_y", type=float, default=5.0,
                        help="Scale: init-box/GT y-coord -> pixel (default 5.0)")
    parser.add_argument("--threshold", type=float, default=60,
                        help="Hard minimum pixel difference threshold (default 60). "
                             "Background diff mean ~7.75, ball diff mean ~143.")
    parser.add_argument("--thresh_pct", type=float, default=97.0,
                        help="Adaptive threshold percentile of the diff image (default 97.0). "
                             "Final threshold = max(--threshold, percentile(diff, thresh_pct)).")
    parser.add_argument("--vel_alpha", type=float, default=0.6,
                        help="Velocity update smoothing factor in [0,1]. "
                             "Higher = faster adaptation (default 0.6).")
    parser.add_argument("--max_search_dist", type=float, default=600,
                        help="Max distance (px) from predicted position to accept a blob "
                             "(default 600, to handle very fast motion).")

    args = parser.parse_args()

    main(
        frames_dir=args.frames_dir,
        init_path=args.init_rect,
        output_path=args.output,
        gt_path=args.gt,
        fps=args.fps,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        threshold=args.threshold,
        thresh_pct=args.thresh_pct,
        vel_alpha=args.vel_alpha,
        max_search_dist=args.max_search_dist,
    )
