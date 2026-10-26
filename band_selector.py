"""
band_selector.py  --  HSI Band Selection Algorithm for Object Tracking
=======================================================================

CONTEXT
-------
hsi_tracker.py computes |frame[t] - frame[t-1]| on raw 16-bit data and
thresholds the result to get a motion mask.  The ball's diff signal is ~18x
stronger than background in the raw band, but 10.7 % of frames are missed
because the contrast collapses in some background regions.

In a real HSI camera each wavelength band has a different quantum-efficiency
curve (non-linear response), a different effective blur (atmospheric PSF), and
different per-material reflectance -- all affecting object/background contrast.

We simulate this spectral diversity by generating 12 VIRTUAL BANDS from each
16-bit frame via spatial and photometric transforms.  The most informative
bands are scored, selected, and fused before thresholding, yielding a stronger
and more consistent motion mask.

VIRTUAL BANDS  (IDs 0-11)
--------------------------
 ID  Name          Transform
  0  raw           Identity  (baseline -- current tracker)
  1  gauss_s1      Gaussian sigma=1  (light denoising)
  2  gauss_s3      Gaussian sigma=3  (medium smoothing)
  3  gauss_s5      Gaussian sigma=5  (heavy smoothing)
  4  dog_fine      |G(s=1) - G(s=3)|    fine-detail bandpass
  5  dog_med       |G(s=2) - G(s=5)|    medium-detail bandpass
  6  dog_coarse    |G(s=3) - G(s=9)|    coarse-detail bandpass
  7  log_s1        |Laplacian(G(s=1))|  edge-enhanced
  8  log_s2        |Laplacian(G(s=2))|  edge-enhanced, coarser
  9  morph_grad    dilate(f,5x5) - erode(f,5x5)  local contrast
 10  sqrt_comp     sqrt(pixel)   mimics high-QE detector
 11  log_comp      log1p(pixel)  emphasises dim-region texture

SCORING CRITERIA  (5 total)
----------------------------
  MER    Motion Energy Ratio = mean(diff_ball) / mean(diff_bg)
  SNR    Signal-to-Noise     = mean(diff_ball) / std(diff_bg)
  Fisher (mu_ball - mu_bg)^2 / (var_ball + var_bg)
  Bhat   Bhattacharyya Distance  -ln(sum(sqrt(p_ball * p_bg)))
  TC     Temporal Consistency  1 / (1 + std_t(MER))

FUSION METHODS
--------------
  mean      equal-weight average of K band diffs
  weighted  score-proportional weighted average
  max       pixel-wise maximum across K band diffs

USAGE
-----
  # Score and auto-select top-3 by MER, fuse with mean:
  python band_selector.py Datasets/ball Datasets/ball/init_rect.txt out.mp4 \
         --gt Datasets/groundtruth_ball.txt \
         --n_score_frames 50 --n_bands 3 --criterion mer --fusion mean \
         --show_scores --scale_x 5 --scale_y 5

  # Manually specify bands (skip scoring):
  python band_selector.py Datasets/ball Datasets/ball/init_rect.txt out.mp4 \
         --gt Datasets/groundtruth_ball.txt --band_ids 0,2,10 --fusion mean
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
# I/O helpers  (identical to hsi_tracker.py for compatibility)
# ---------------------------------------------------------------------------

def natural_sort_key(path):
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', os.path.basename(path))]


def load_frame(path):
    return np.array(Image.open(path))


def compute_global_range(frame_paths, sample_every=5, low_pct=0.5, high_pct=99.5):
    """Percentile contrast-stretch range over a sampled subset of frames."""
    samples = []
    for p in frame_paths[::sample_every]:
        samples.append(load_frame(p).astype(np.float32).ravel())
    all_vals = np.concatenate(samples)
    return np.percentile(all_vals, low_pct), np.percentile(all_vals, high_pct)


def stretch_to_uint8(arr, lo, hi):
    arr = arr.astype(np.float32)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def parse_box_line(line):
    """Parse one x y w h line; returns None for empty/sentinel entries."""
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
    with open(init_path) as f:
        for line in f:
            box = parse_box_line(line)
            if box is not None:
                x, y, w, h = box
                return (int(round(x * scale_x)), int(round(y * scale_y)),
                        int(round(w * scale_x)), int(round(h * scale_y)))
    raise ValueError(f"No valid bounding box in {init_path}")


def load_groundtruth(gt_path, scale_x=1.0, scale_y=1.0):
    """Load all GT boxes in pixel space; None for occluded/invalid frames."""
    boxes = []
    with open(gt_path) as f:
        for line in f:
            box = parse_box_line(line)
            if box is not None:
                x, y, w, h = box
                boxes.append((int(round(x * scale_x)), int(round(y * scale_y)),
                               int(round(w * scale_x)), int(round(h * scale_y))))
            else:
                boxes.append(None)
    return boxes


# ---------------------------------------------------------------------------
# Virtual Band Registry
# ---------------------------------------------------------------------------

BAND_IDS = list(range(12))

BAND_NAMES = [
    'raw',        'gauss_s1',   'gauss_s3',   'gauss_s5',
    'dog_fine',   'dog_med',    'dog_coarse',
    'log_s1',     'log_s2',     'morph_grad',
    'sqrt_comp',  'log_comp',
]

BAND_DESCRIPTIONS = [
    'Raw 16-bit (baseline -- current tracker)',
    'Gaussian s=1 (light denoising)',
    'Gaussian s=3 (medium smoothing)',
    'Gaussian s=5 (heavy smoothing)',
    'DoG (s=1,3): fine-detail bandpass',
    'DoG (s=2,5): medium-detail bandpass',
    'DoG (s=3,9): coarse-detail bandpass',
    'LoG s=1: edge-enhanced |Lap(Gauss)|',
    'LoG s=2: edge-enhanced, coarser scale',
    'Morphological gradient (5x5 ellipse)',
    'Sqrt compression -- mimics high-QE detector',
    'Log compression -- emphasises dim-region texture',
]

N_BANDS = len(BAND_IDS)   # 12

CRITERIA = ['mer', 'snr', 'fisher', 'bhat', 'tc']
CRITERIA_LABELS = {
    'mer':    'Motion Energy Ratio',
    'snr':    'Signal-to-Noise Ratio',
    'fisher': "Fisher's Discriminant",
    'bhat':   'Bhattacharyya Distance',
    'tc':     'Temporal Consistency',
}

_MORPH_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_MORPH_K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


# ---------------------------------------------------------------------------
# Band Transform Engine
# ---------------------------------------------------------------------------

def apply_band(frame_f32, band_id):
    """
    Apply a virtual spectral-band transform to a float32 frame.

    All inputs are float32 (uint16 cast to float32 before calling).
    All outputs are float32, same spatial shape, always >= 0.

    Parameters
    ----------
    frame_f32 : np.ndarray  float32  shape (H, W)
    band_id   : int  0--11  (see BAND_NAMES)
    """
    f = frame_f32
    if band_id == 0:    return f
    elif band_id == 1:  return cv2.GaussianBlur(f, (0, 0), 1.0)
    elif band_id == 2:  return cv2.GaussianBlur(f, (0, 0), 3.0)
    elif band_id == 3:  return cv2.GaussianBlur(f, (0, 0), 5.0)
    elif band_id == 4:
        return np.abs(cv2.GaussianBlur(f,(0,0),1.0) - cv2.GaussianBlur(f,(0,0),3.0))
    elif band_id == 5:
        return np.abs(cv2.GaussianBlur(f,(0,0),2.0) - cv2.GaussianBlur(f,(0,0),5.0))
    elif band_id == 6:
        return np.abs(cv2.GaussianBlur(f,(0,0),3.0) - cv2.GaussianBlur(f,(0,0),9.0))
    elif band_id == 7:
        return np.abs(cv2.Laplacian(cv2.GaussianBlur(f,(0,0),1.0), cv2.CV_32F))
    elif band_id == 8:
        return np.abs(cv2.Laplacian(cv2.GaussianBlur(f,(0,0),2.0), cv2.CV_32F))
    elif band_id == 9:
        return cv2.dilate(f, _MORPH_K5) - cv2.erode(f, _MORPH_K5)
    elif band_id == 10: return np.sqrt(np.maximum(f, 0.0))
    elif band_id == 11: return np.log1p(np.maximum(f, 0.0))
    else:               return f


def compute_band_diff(prev_f32, curr_f32, band_id):
    """
    Absolute temporal diff of virtual band `band_id` between two frames.
    Result = |apply_band(curr) - apply_band(prev)|  >= 0 always.
    """
    return np.abs(apply_band(curr_f32, band_id) - apply_band(prev_f32, band_id))


# ---------------------------------------------------------------------------
# Band Scoring
# ---------------------------------------------------------------------------

def _score_diff_image(diff, gt_box_px, frame_h, frame_w, pad=15):
    """
    Score one band diff image against one GT bounding box.

    Object  = pixels inside gt_box_px  (top-left x,y,w,h format)
    Background = pixels outside the box padded by `pad` pixels

    Returns dict {mer, snr, fisher, bhat} or None if box is degenerate.
    """
    x, y, bw, bh = [int(round(v)) for v in gt_box_px]

    ox1 = max(0, x);            oy1 = max(0, y)
    ox2 = min(frame_w, x + bw); oy2 = min(frame_h, y + bh)
    if ox2 <= ox1 or oy2 <= oy1:
        return None
    obj = diff[oy1:oy2, ox1:ox2].ravel()

    ex1 = max(0, x - pad);             ey1 = max(0, y - pad)
    ex2 = min(frame_w, x + bw + pad);  ey2 = min(frame_h, y + bh + pad)
    bg_mask = np.ones((frame_h, frame_w), dtype=bool)
    bg_mask[ey1:ey2, ex1:ex2] = False
    bg = diff[bg_mask]

    if len(obj) < 4 or len(bg) < 50:
        return None

    mu_o = float(obj.mean());  mu_b = float(bg.mean())
    sd_o = float(obj.std()) + 1e-8
    sd_b = float(bg.std())  + 1e-8

    mer    = mu_o / (mu_b + 1e-8)
    snr    = mu_o / sd_b
    fisher = (mu_o - mu_b) ** 2 / (sd_o ** 2 + sd_b ** 2 + 1e-8)

    rng    = max(float(max(obj.max(), bg.max())), 1.0)
    h_o, _ = np.histogram(obj, bins=32, range=(0, rng), density=True)
    h_b, _ = np.histogram(bg,  bins=32, range=(0, rng), density=True)
    bc     = float(np.sum(np.sqrt((h_o + 1e-10) * (h_b + 1e-10)))) * (rng / 32)
    bhat   = float(-np.log(bc + 1e-10))

    return {'mer': mer, 'snr': snr, 'fisher': fisher, 'bhat': bhat}


def score_all_bands(frame_paths, gt_boxes, n_sample=50, verbose=True):
    """
    Score all 12 virtual bands by sampling n_sample consecutive frame pairs.

    Parameters
    ----------
    frame_paths : sorted list of PNG file paths
    gt_boxes    : pixel-space (x,y,w,h) list from load_groundtruth()
                  (None for occluded/invalid frames)
    n_sample    : number of frame pairs to sample uniformly
    verbose     : print progress

    Returns
    -------
    scores : dict  {band_id: {criterion: float}}
        Criteria: 'mer', 'snr', 'fisher', 'bhat', 'tc'
        Also '<crit>_std' for temporal variability.
    """
    valid_idx = [
        i for i in range(1, len(frame_paths))
        if (i < len(gt_boxes)
            and gt_boxes[i] is not None
            and gt_boxes[i - 1] is not None)
    ]
    if not valid_idx:
        raise ValueError("No valid frame pairs with GT boxes for scoring.")

    if len(valid_idx) > n_sample:
        sel    = np.linspace(0, len(valid_idx) - 1, n_sample, dtype=int)
        sample = [valid_idx[i] for i in sel]
    else:
        sample = valid_idx

    if verbose:
        print(f"\n[Band Scoring]  {N_BANDS} bands x {len(sample)} frame pairs "
              f"(from {len(valid_idx)} valid)")

    raw = {bid: {c: [] for c in ['mer', 'snr', 'fisher', 'bhat']}
           for bid in BAND_IDS}

    for k, fi in enumerate(sample):
        prev_f32 = load_frame(frame_paths[fi - 1]).astype(np.float32)
        curr_f32 = load_frame(frame_paths[fi    ]).astype(np.float32)
        fh, fw   = curr_f32.shape[:2]
        gt_px    = gt_boxes[fi]

        for bid in BAND_IDS:
            diff = compute_band_diff(prev_f32, curr_f32, bid)
            sc   = _score_diff_image(diff, gt_px, fh, fw)
            if sc:
                for c in ['mer', 'snr', 'fisher', 'bhat']:
                    raw[bid][c].append(sc[c])

        if verbose:
            print(f"  Scored {k + 1:3d}/{len(sample)} pairs", end='\r')

    if verbose:
        print()

    scores = {}
    for bid in BAND_IDS:
        scores[bid] = {}
        for c in ['mer', 'snr', 'fisher', 'bhat']:
            vals = np.array(raw[bid][c])
            scores[bid][c]          = float(vals.mean()) if len(vals) else 0.0
            scores[bid][f'{c}_std'] = float(vals.std())  if len(vals) else 0.0
        mer_vals          = np.array(raw[bid]['mer'])
        scores[bid]['tc'] = (1.0 / (1.0 + float(mer_vals.std()))
                             if len(mer_vals) > 1 else 0.0)
    return scores


def select_top_bands(scores, criterion='mer', n_bands=3):
    """
    Rank all bands by `criterion` and return top n_bands as
    list of (band_id, score) sorted best-first.
    """
    n_bands = min(n_bands, N_BANDS)
    ranked  = sorted(BAND_IDS, key=lambda b: scores[b].get(criterion, 0.0),
                     reverse=True)
    return [(bid, scores[bid][criterion]) for bid in ranked[:n_bands]]


def print_band_scores(scores, highlight='mer'):
    """Print the full band scoring table with best-per-column markers (*)."""
    crits = ['mer', 'snr', 'fisher', 'bhat', 'tc']
    best  = {c: max(BAND_IDS, key=lambda b: scores[b].get(c, 0)) for c in crits}
    sep   = '-' * 95
    hdr   = (f"{'ID':>3}  {'Name':<13}  " +
             "  ".join(f"{c.upper():>8}" for c in crits) +
             "  Description")
    print(f"\n{sep}\n{hdr}\n{sep}")
    for bid in BAND_IDS:
        sc   = scores[bid]
        vals = "  ".join(
            f"{'*'+f'{sc[c]:.3f}':>8}" if best[c] == bid else f"{sc[c]:>8.3f}"
            for c in crits
        )
        print(f"{bid:>3}  {BAND_NAMES[bid]:<13}  {vals}  {BAND_DESCRIPTIONS[bid]}")
    print(f"{sep}\n  * = best in column\n")


# ---------------------------------------------------------------------------
# Multi-Band Fused Diff Mask
# ---------------------------------------------------------------------------

def compute_fused_diff_mask(prev_raw, curr_raw, band_ids,
                             fusion='mean', band_weights=None,
                             threshold=60.0, thresh_pct=97.0):
    """
    Compute a binary motion mask fused across multiple virtual bands.

    Drop-in replacement for compute_diff_mask() in hsi_tracker.py.

    Algorithm
    ---------
    1. For each bid in band_ids:
         diff_k = |apply_band(curr,bid) - apply_band(prev,bid)|
    2. Fuse diffs:
         mean     -> mean(diff_k, axis=0)
         weighted -> weighted mean using band_weights
         max      -> pixel-wise max(diff_k, axis=0)
    3. threshold = max(threshold, percentile(fused, thresh_pct))
    4. binary = (fused > threshold)
    5. morphological close + open  (9x9 ellipse, same as hsi_tracker.py)

    Returns
    -------
    binary, fused_diff, final_thresh  -- same interface as compute_diff_mask
    """
    prev_f32 = prev_raw.astype(np.float32)
    curr_f32 = curr_raw.astype(np.float32)

    diffs = []
    for bid in band_ids:
        d = compute_band_diff(prev_f32, curr_f32, bid)
        # Robust dynamic range normalization: scale each band so its background
        # percentile matches raw baseline (~45.0). Subsampled 4x4 for fast compute.
        p_sub = float(np.percentile(d[::4, ::4], thresh_pct))
        if p_sub > 1e-4:
            d = d * (45.0 / p_sub)
        diffs.append(d)

    if len(diffs) == 1:
        fused = diffs[0]
    else:
        stack = np.stack(diffs, axis=0)
        if fusion == 'max':
            fused = stack.max(axis=0)
        elif fusion == 'weighted' and band_weights is not None:
            w = np.array(band_weights, dtype=np.float32)
            w /= (w.sum() + 1e-8)
            fused = (stack * w[:, None, None]).sum(axis=0)
        else:
            fused = stack.mean(axis=0)

    adaptive     = float(np.percentile(fused[::4, ::4], thresh_pct))
    final_thresh = max(threshold, adaptive)
    binary       = (fused > final_thresh).astype(np.uint8)
    binary       = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _MORPH_K9)
    binary       = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _MORPH_K9)
    return binary, fused, final_thresh



# ---------------------------------------------------------------------------
# Evaluation helpers  (identical to hsi_tracker.py)
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a;  bx1, by1, bw, bh = box_b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax1+aw,bx1+bw); iy2=min(ay1+ah,by1+bh)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def centre_error(box_pred, box_gt):
    px=box_pred[0]+box_pred[2]/2;  py=box_pred[1]+box_pred[3]/2
    gx=box_gt[0]  +box_gt[2]  /2;  gy=box_gt[1]  +box_gt[3]  /2
    return float(np.sqrt((px-gx)**2+(py-gy)**2))


def find_best_blob(binary, diff, pred_cx, pred_cy,
                   min_area, max_area, max_dist):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if n <= 1: return None
    best=None; best_d=float('inf'); best_c=0.0
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area: continue
        cx_b,cy_b = float(centroids[lbl][0]),float(centroids[lbl][1])
        d = float(np.hypot(cx_b-pred_cx, cy_b-pred_cy))
        if d > max_dist: continue
        if d < best_d:
            best_d=d; best_c=float(diff[labels==lbl].mean()); best=(cx_b,cy_b)
    return (best[0], best[1], best_c) if best is not None else None


# ---------------------------------------------------------------------------
# Band-Aware Tracker  (self-contained pipeline)
# ---------------------------------------------------------------------------

def run_band_tracker(frames_dir, init_path, output_path,
                     band_ids, fusion='mean', band_weights=None,
                     gt_path=None, fps=25,
                     scale_x=5.0, scale_y=5.0,
                     threshold=60.0, thresh_pct=97.0,
                     vel_alpha=0.6, max_search_dist=600.0,
                     write_video=True, verbose=False,
                     preloaded_frames=None):
    """
    Full band-aware frame-difference blob tracker.

    Identical algorithm to hsi_tracker.py, but compute_diff_mask() is
    replaced by compute_fused_diff_mask() with the given band configuration.

    Returns
    -------
    dict:
      n_frames, n_eval,
      mean_iou, median_iou,
      iou_gt_0_25 (%), iou_gt_0_50 (%),
      mean_ce, median_ce,
      fallback_rate (%),
      ious, cerrors, pred_boxes
    """
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')),
                         key=natural_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames in '{frames_dir}'")

    ix, iy, iw, ih = load_init_box(init_path, scale_x, scale_y)
    ball_area = iw * ih
    min_area  = int(ball_area * 0.15)
    max_area  = int(ball_area * 5.0)

    gt_boxes = None
    if gt_path:
        gt_boxes = load_groundtruth(gt_path, scale_x, scale_y)

    lo, hi    = compute_global_range(frame_paths)
    first_raw = preloaded_frames[0] if preloaded_frames is not None else load_frame(frame_paths[0])
    fh, fw    = first_raw.shape[:2]

    writer = None
    if write_video:
        first_u8 = stretch_to_uint8(first_raw, lo, hi)
        vh, vw   = first_u8.shape[:2]
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(output_path, fourcc, fps, (vw, vh), True)

    cx,cy = float(ix+iw/2), float(iy+ih/2)
    bw,bh = iw, ih
    vx,vy = 0.0, 0.0

    ious_list, cerrors, pred_boxes = [], [], []
    prev_raw       = first_raw.copy()
    fallback_count = 0

    for i, fpath in enumerate(frame_paths):
        curr_raw = preloaded_frames[i] if preloaded_frames is not None else load_frame(fpath)
        conf=0.0; used_fb=False

        if i == 0:
            new_cx, new_cy = cx, cy
        else:
            pred_cx = float(np.clip(cx+vx, bw/2, fw-bw/2))
            pred_cy = float(np.clip(cy+vy, bh/2, fh-bh/2))

            binary, diff, _ = compute_fused_diff_mask(
                prev_raw, curr_raw, band_ids, fusion, band_weights,
                threshold, thresh_pct)

            result = find_best_blob(binary, diff, pred_cx, pred_cy,
                                    min_area, max_area, max_search_dist)
            if result is not None:
                blob_cx, blob_cy, conf = result
                # Midpoint-bias correction: diff blob centroid is the midpoint
                # of old and new ball positions -> shift forward by 0.5*velocity
                new_cx = float(np.clip(blob_cx+vx*0.5, bw/2, fw-bw/2))
                new_cy = float(np.clip(blob_cy+vy*0.5, bh/2, fh-bh/2))
                vx = vel_alpha*(new_cx-cx) + (1-vel_alpha)*vx
                vy = vel_alpha*(new_cy-cy) + (1-vel_alpha)*vy
            else:
                new_cx,new_cy = pred_cx, pred_cy
                used_fb=True; fallback_count+=1
                vx*=0.85; vy*=0.85

        cx,cy    = new_cx, new_cy
        prev_raw = curr_raw.copy()

        px_box   = int(round(cx-bw/2))
        py_box   = int(round(cy-bh/2))
        pred_box = (px_box, py_box, bw, bh)
        pred_boxes.append(pred_box)

        if gt_boxes and i < len(gt_boxes) and gt_boxes[i] is not None:
            ious_list.append(iou(pred_box, gt_boxes[i]))
            cerrors.append(centre_error(pred_box, gt_boxes[i]))

        if write_video and writer is not None:
            curr_u8   = stretch_to_uint8(curr_raw, lo, hi)
            frame_bgr = (cv2.cvtColor(curr_u8, cv2.COLOR_GRAY2BGR)
                         if curr_u8.ndim == 2
                         else cv2.cvtColor(curr_u8, cv2.COLOR_RGB2BGR))
            color = (0,165,255) if used_fb else (0,255,0)
            cv2.rectangle(frame_bgr,(px_box,py_box),(px_box+bw,py_box+bh),color,2)
            status = 'fallback' if used_fb else f'conf={conf:.0f}'
            cv2.putText(frame_bgr,
                        f"f{i:04d}  vx={vx:.1f} vy={vy:.1f}  {status}",
                        (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
            if gt_boxes and i < len(gt_boxes) and gt_boxes[i] is not None:
                gx,gy,gw,gh = gt_boxes[i]
                cv2.rectangle(frame_bgr,(gx,gy),(gx+gw,gy+gh),(0,0,255),2)
                cv2.putText(frame_bgr,'GT',(gx,max(gy-5,10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,0,255),1,cv2.LINE_AA)
                if ious_list:
                    cv2.putText(frame_bgr,
                                f"IoU={ious_list[-1]:.2f}  CE={cerrors[-1]:.1f}px",
                                (5,42),cv2.FONT_HERSHEY_SIMPLEX,0.45,
                                (255,255,0),1,cv2.LINE_AA)
            writer.write(frame_bgr)

        if verbose and i % 50 == 0:
            istr = f"  IoU={ious_list[-1]:.3f}" if ious_list else ''
            print(f"  frame {i:4d}/{len(frame_paths)}  "
                  f"cx={cx:.1f} cy={cy:.1f}  fb={used_fb}{istr}")

    if writer is not None:
        writer.release()

    arr_iou = np.array(ious_list)
    arr_ce  = np.array(cerrors)
    n       = len(frame_paths)
    return {
        'n_frames'     : n,
        'n_eval'       : len(ious_list),
        'mean_iou'     : float(arr_iou.mean())                 if len(arr_iou) else 0.0,
        'median_iou'   : float(np.median(arr_iou))             if len(arr_iou) else 0.0,
        'iou_gt_0_25'  : float((arr_iou>0.25).mean()*100)     if len(arr_iou) else 0.0,
        'iou_gt_0_50'  : float((arr_iou>0.50).mean()*100)     if len(arr_iou) else 0.0,
        'mean_ce'      : float(arr_ce.mean())                  if len(arr_ce)  else 0.0,
        'median_ce'    : float(np.median(arr_ce))              if len(arr_ce)  else 0.0,
        'fallback_rate': fallback_count / n * 100,
        'ious'         : ious_list,
        'cerrors'      : cerrors,
        'pred_boxes'   : pred_boxes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="HSI Band Selection Tracker")
    p.add_argument('frames_dir')
    p.add_argument('init_rect')
    p.add_argument('output')
    p.add_argument('--gt',               default=None)
    p.add_argument('--fps',              type=int,   default=25)
    p.add_argument('--scale_x',          type=float, default=5.0)
    p.add_argument('--scale_y',          type=float, default=5.0)
    p.add_argument('--n_score_frames',   type=int,   default=50)
    p.add_argument('--n_bands',          type=int,   default=3)
    p.add_argument('--criterion',        default='mer', choices=CRITERIA)
    p.add_argument('--fusion',           default='mean',
                   choices=['mean','weighted','max'])
    p.add_argument('--show_scores',      action='store_true')
    p.add_argument('--band_ids',         default=None,
                   help="e.g. --band_ids 0,4,10  (skips scoring)")
    p.add_argument('--threshold',        type=float, default=60.0)
    p.add_argument('--thresh_pct',       type=float, default=97.0)
    p.add_argument('--vel_alpha',        type=float, default=0.6)
    p.add_argument('--max_search_dist',  type=float, default=600.0)
    args = p.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir,'*.png')),
                         key=natural_sort_key)
    if not frame_paths:
        print(f"ERROR: No PNG frames in '{args.frames_dir}'"); sys.exit(1)
    print(f"Found {len(frame_paths)} frames.")

    if args.band_ids:
        selected_ids = [int(b.strip()) for b in args.band_ids.split(',')]
        band_weights = None
        print(f"\nManually specified bands: {selected_ids}")
        for bid in selected_ids:
            print(f"  B{bid}: {BAND_NAMES[bid]}  --  {BAND_DESCRIPTIONS[bid]}")
    else:
        if not args.gt:
            print("\nWARNING: --gt not provided; defaulting to raw band (B0).")
            selected_ids = [0];  band_weights = None
        else:
            gt_px  = load_groundtruth(args.gt, args.scale_x, args.scale_y)
            scores = score_all_bands(frame_paths, gt_px,
                                     n_sample=args.n_score_frames, verbose=True)
            if args.show_scores:
                print_band_scores(scores, args.criterion)

            top          = select_top_bands(scores, args.criterion, args.n_bands)
            selected_ids = [bid for bid,_ in top]
            raw_scores   = [sc  for _,sc  in top]
            band_weights = raw_scores if args.fusion == 'weighted' else None

            crit_lbl = CRITERIA_LABELS[args.criterion]
            print(f"\nSelected top-{args.n_bands} bands by {crit_lbl}:")
            for bid, sc in top:
                print(f"  B{bid} {BAND_NAMES[bid]:<13}  score={sc:.4f}"
                      f"  [{BAND_DESCRIPTIONS[bid]}]")

    print(f"\nFusion method: {args.fusion}")
    print(f"Running tracker on {len(frame_paths)} frames ...")

    res = run_band_tracker(
        frames_dir=args.frames_dir,   init_path=args.init_rect,
        output_path=args.output,      band_ids=selected_ids,
        fusion=args.fusion,           band_weights=band_weights,
        gt_path=args.gt,              fps=args.fps,
        scale_x=args.scale_x,        scale_y=args.scale_y,
        threshold=args.threshold,     thresh_pct=args.thresh_pct,
        vel_alpha=args.vel_alpha,     max_search_dist=args.max_search_dist,
        write_video=True,             verbose=True,
    )

    print(f"\n{'='*55}")
    print(f"  BAND-AWARE TRACKER RESULTS")
    print(f"{'='*55}")
    print(f"  Bands used   : {[BAND_NAMES[b] for b in selected_ids]}")
    print(f"  Fusion       : {args.fusion}")
    print(f"  Frames eval  : {res['n_eval']} / {res['n_frames']}")
    print(f"  Mean IoU     : {res['mean_iou']:.4f}")
    print(f"  Median IoU   : {res['median_iou']:.4f}")
    print(f"  IoU > 0.50   : {res['iou_gt_0_50']:.1f}%")
    print(f"  IoU > 0.25   : {res['iou_gt_0_25']:.1f}%")
    print(f"  Mean CE      : {res['mean_ce']:.2f} px")
    print(f"  Fallback     : {res['fallback_rate']:.1f}%")
    print(f"{'='*55}")
    print(f"  Video        : {args.output}")

    txt = os.path.splitext(args.output)[0] + '_predicted_boxes.txt'
    with open(txt, 'w') as f:
        for b in res['pred_boxes']:
            f.write(f"{b[0]}\t{b[1]}\t{b[2]}\t{b[3]}\n")
    print(f"  Pred boxes   : {txt}")


if __name__ == '__main__':
    main()
