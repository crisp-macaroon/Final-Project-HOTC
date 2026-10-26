"""
Build a video from a folder of HSI-NIR PNG frames with the ground-truth
bounding box drawn on each frame.

Expects a groundtruth file (common HOT/VOT format) with one line per frame:
    x,y,w,h
or
    x,y,w,h    (whitespace/tab separated)
where (x, y) is the top-left corner of the box and (w, h) is width/height,
in pixel coordinates. Adjust `parse_gt_line` below if your file uses a
different convention (e.g. 4-point polygon x1,y1,x2,y2,x3,y3,x4,y4).

Usage:
    python overlay_gt_boxes.py /path/to/ball_frames_folder groundtruth.txt output.mp4

Requirements:
    pip install opencv-python numpy pillow --break-system-packages
"""

import sys
import glob
import os
import re
import numpy as np
import cv2
from PIL import Image


def natural_sort_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(path))]


def load_frame(path):
    img = Image.open(path)
    return np.array(img)


def compute_global_range(frame_paths, sample_every=5, low_pct=0.5, high_pct=99.5):
    samples = []
    for p in frame_paths[::sample_every]:
        arr = load_frame(p)
        samples.append(arr.astype(np.float32).ravel())
    all_vals = np.concatenate(samples)
    lo = np.percentile(all_vals, low_pct)
    hi = np.percentile(all_vals, high_pct)
    return lo, hi


def stretch_to_uint8(arr, lo, hi):
    arr = arr.astype(np.float32)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def parse_gt_line(line):
    """
    Parse one line of the groundtruth file into (x, y, w, h) as ints.
    Handles comma, tab, or space separated values.
    Returns None for empty/invalid lines (e.g. occluded/out-of-frame markers).
    """

    line = line.strip()
    if not line:
        return None
    parts = re.split(r'[,\s]+', line)
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        return None
    if len(vals) == 4:
        cx, cy, w, h = vals
        if w <= 0 or h <= 0:
            return None
        return cx, cy, w, h   # keep as float until scaling is applied
    return None

    # line = line.strip()
    # if not line:
    #     return None
    # parts = re.split(r'[,\s]+', line)
    # try:
    #     vals = [float(v) for v in parts]
    # except ValueError:
    #     return None

    # if len(vals) == 4:
    #     x, y, w, h = vals
    #     if x < 0 or y < 0 or w <= 0 or h <= 0:
    #         # sentinel for occluded / out-of-view target (e.g. -1,-1,-1,-1)
    #         return None
    #     return int(round(x)), int(round(y)), int(round(w)), int(round(h))
    # elif len(vals) == 8:
    #     # 4-point polygon x1,y1,x2,y2,x3,y3,x4,y4 -> convert to axis-aligned box
    #     xs = vals[0::2]
    #     ys = vals[1::2]
    #     x, y = min(xs), min(ys)
    #     w, h = max(xs) - x, max(ys) - y
    #     return int(round(x)), int(round(y)), int(round(w)), int(round(h))
    # else:
    #     return None


def load_groundtruth(gt_path):
    with open(gt_path, "r") as f:
        lines = f.readlines()
    return [parse_gt_line(l) for l in lines]


def infer_gt_native_size(boxes):
    """
    Estimate the resolution the groundtruth was annotated on, by taking the
    max (x+w) and max (y+h) across all valid boxes. Not exact (annotations
    might not touch the frame edge) but gives a solid ballpark to compare
    against the actual frame size.
    """
    max_x2 = 0
    max_y2 = 0
    for b in boxes:
        if b is None:
            continue
        x, y, w, h = b
        max_x2 = max(max_x2, x + w)
        max_y2 = max(max_y2, y + h)
    return max_x2, max_y2


def main(frames_dir, gt_path, output_path, fps=25, box_color=(0, 0, 255), thickness=2,
         scale_x=None, scale_y=None):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")), key=natural_sort_key)
    if not frame_paths:
        print(f"No PNG files found in {frames_dir}")
        return

    boxes = load_groundtruth(gt_path)

    # valid = [b for b in boxes if b is not None]

    # print("Max x :", max(x for x, y, w, h in valid))
    # print("Max y :", max(y for x, y, w, h in valid))
    # print("Max x+w :", max(x + w for x, y, w, h in valid))
    # print("Max y+h :", max(y + h for x, y, w, h in valid))
    # print("Min x :", min(x for x, y, w, h in valid))
    # print("Min y :", min(y for x, y, w, h in valid))

    print(f"Found {len(frame_paths)} frames and {len(boxes)} groundtruth lines.")

    if len(boxes) != len(frame_paths):
        print("WARNING: number of groundtruth lines does not match number of frames. "
              "Boxes will be matched by index up to the shorter length.")

    # Alignment sanity check: print filename <-> gt-line pairing for the first
    # and last few frames so you can confirm the sort order actually matches
    # the groundtruth file's row order BEFORE rendering the full video.
    print("\n--- alignment check (filename -> groundtruth line) ---")
    check_idxs = list(range(min(5, len(frame_paths)))) + \
                 list(range(max(0, len(frame_paths) - 5), len(frame_paths)))
    for i in sorted(set(check_idxs)):
        fname = os.path.basename(frame_paths[i])
        box = boxes[i] if i < len(boxes) else None
        print(f"  idx {i:4d}: {fname:20s} -> {box}")
    print("--- if these look misaligned (e.g. filenames not in numeric order), "
          "fix natural_sort_key or re-check your file listing before proceeding ---\n")

    first = load_frame(frame_paths[0])
    #print("Shape:", first.shape)
    print(f"First frame: shape={first.shape}, dtype={first.dtype}, "
          f"min={first.min()}, max={first.max()}, mean={first.mean():.1f}")

    print("Scanning frames to compute a stable contrast-stretch range...")
    lo, hi = compute_global_range(frame_paths)
    print(f"Using stretch range: lo={lo:.1f}, hi={hi:.1f}")

    h, w = first.shape[:2]

    # # --- resolution mismatch check / auto-scale ---
    # if scale_x is None or scale_y is None:
    #     gt_w, gt_h = infer_gt_native_size(boxes)
    #     print(f"Actual frame size:      {w} x {h}")
    #     print(f"Inferred GT canvas size (max box extent): {gt_w} x {gt_h}")
    #     if gt_w > 0 and gt_h > 0 and (gt_w > w * 1.15 or gt_h > h * 1.15 or
    #                                    gt_w < w * 0.85 or gt_h < h * 0.85):
    #         auto_sx = w / gt_w
    #         auto_sy = h / gt_h
    #         print(f"WARNING: groundtruth appears annotated on a different resolution "
    #               f"than your frames. Auto-scaling boxes by ({auto_sx:.4f}, {auto_sy:.4f}). "
    #               f"If this looks wrong, pass --scale_x/--scale_y manually.")
    #         scale_x, scale_y = auto_sx, auto_sy
    #     else:
    #         scale_x, scale_y = 1.0, 1.0
    # #         print("GT canvas size roughly matches frame size, no scaling applied.")

    scale_x, scale_y = 5.0, 5.0
    print(f"Using scale_x={scale_x:.4f}, scale_y={scale_y:.4f}\n")
    # -----------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    n = min(len(frame_paths), len(boxes))
    for i in range(n):
        arr = load_frame(frame_paths[i])
        stretched = stretch_to_uint8(arr, lo, hi)

        if stretched.ndim == 2:
            frame_bgr = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = cv2.cvtColor(stretched, cv2.COLOR_RGB2BGR)

        box = boxes[i]
        if box is not None:
            x, y, bw, bh = box
            x *= scale_x
            y *= scale_y
            bw *= scale_x
            bh *= scale_y

            x = int(round(x))
            y = int(round(y))
            bw = int(round(bw))
            bh = int(round(bh))
            cv2.rectangle(frame_bgr, (x, y), (x + int(round(bw)), y + int(round(bh))), box_color, thickness)
            cv2.putText(frame_bgr, f"frame {i}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(frame_bgr,f"({x},{y})",(x + 10, y),cv2.FONT_HERSHEY_SIMPLEX,
            #              0.4,(0,255,0),1)
        else:
            # No valid box for this frame (e.g. target out of view/occluded)
            cv2.putText(frame_bgr, f"frame {i} (no gt)", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        writer.write(frame_bgr)

        if i % 100 == 0:
            print(f"  processed {i}/{n} frames")

    writer.release()
    print(f"Done. Video written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python overlay_gt_boxes.py <frames_dir> <groundtruth.txt> <output_video.mp4> "
              "[fps] [scale_x] [scale_y]")
        print("  scale_x/scale_y: optional manual override if auto-detection gets it wrong, "
              "e.g. if GT was annotated on a 512x512 canvas and your frames are 128x128, "
              "pass scale_x=0.25 scale_y=0.25")
        sys.exit(1)

    frames_dir = sys.argv[1]
    gt_path = sys.argv[2]
    output_path = sys.argv[3]
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    scale_x = float(sys.argv[5]) if len(sys.argv) > 5 else None
    scale_y = float(sys.argv[6]) if len(sys.argv) > 6 else None

    main(frames_dir, gt_path, output_path, fps, scale_x=scale_x, scale_y=scale_y)