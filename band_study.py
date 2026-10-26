"""
band_study.py  --  Comparative Band Selection Study for HSI Object Tracking
============================================================================

Runs the band-aware tracker across multiple experimental configurations and
produces a full comparative analysis: printed tables, CSV exports, and
matplotlib plots.

EXPERIMENT GROUPS
-----------------
  Group A  Single Bands (12 experiments)
           Run tracker on each of the 12 virtual bands individually.
           Establishes per-band baseline performance.

  Group B  Top-K Fusion by Criterion (15 experiments)
           For each of 3 criteria (MER, SNR, Fisher) x 5 K values (1,2,3,5,12):
           select top-K bands and fuse with mean fusion.
           Reveals: (a) which criterion predicts tracking performance best, and
                    (b) whether more bands always helps.

  Group C  Fusion Method Comparison (2 extra experiments)
           Use best criterion (MER), K=3:
           compare weighted-mean and max fusion against the mean-fusion
           result already computed in Group B.

OUTPUTS  (all written to --output_dir)
-----------------------------------------
  results.csv               -- all experiment metrics
  band_scores.csv           -- per-band per-criterion scores
  iou_comparison.png        -- bar chart: Mean IoU per config
  ce_comparison.png         -- bar chart: Mean Centre Error per config
  band_score_heatmap.png    -- heatmap: bands x criteria (normalised)
  topk_effect.png           -- line plot: IoU and CE vs K per criterion
  fusion_comparison.png     -- bar chart: fusion method comparison
  mer_vs_iou.png            -- scatter: band MER score vs tracking IoU

USAGE
-----
  # Full study, no video output (fast):
  python band_study.py Datasets/ball Datasets/ball/init_rect.txt \
         --gt Datasets/groundtruth_ball.txt \
         --scale_x 5 --scale_y 5 \
         --n_score_frames 50 \
         --output_dir band_study_results \
         --no_video

  # With video output for best config only:
  python band_study.py Datasets/ball Datasets/ball/init_rect.txt \
         --gt Datasets/groundtruth_ball.txt \
         --output_dir band_study_results
"""

import sys
import os
import glob
import csv
import time
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found -- plots will not be generated.")

from band_selector import (
    BAND_IDS, BAND_NAMES, BAND_DESCRIPTIONS, N_BANDS,
    CRITERIA, CRITERIA_LABELS,
    natural_sort_key, load_groundtruth, load_frame,
    score_all_bands, select_top_bands, print_band_scores,
    run_band_tracker,
)
import glob as _glob


# ---------------------------------------------------------------------------
# Experiment Builder
# ---------------------------------------------------------------------------

def build_experiments(scores, k_values=(1, 2, 3, 5, 12),
                      criteria=('mer', 'snr', 'fisher')):
    """
    Build the full list of experimental configurations.

    Each config is a dict:
      label, group, band_ids, fusion, band_weights, description
    """
    exps = []

    # -- Group A: single bands -----------------------------------------------
    for bid in BAND_IDS:
        exps.append({
            'label':       f"A_B{bid:02d}_{BAND_NAMES[bid]}",
            'group':       'A_single_band',
            'band_ids':    [bid],
            'fusion':      'mean',
            'band_weights': None,
            'description': f"Single band B{bid}: {BAND_DESCRIPTIONS[bid]}",
        })

    # -- Group B: top-K fusion by criterion ----------------------------------
    for crit in criteria:
        for k in k_values:
            top   = select_top_bands(scores, crit, min(k, N_BANDS))
            bids  = [b for b, _ in top]
            bscrs = [s for _, s in top]
            exps.append({
                'label':       f"B_Top{k}_{crit}_mean",
                'group':       f"B_topK_{crit}",
                'band_ids':    bids,
                'fusion':      'mean',
                'band_weights': None,
                'description': (f"Top-{k} by {CRITERIA_LABELS[crit]}, "
                                f"mean fusion, bands={bids}"),
            })

    # -- Group C: fusion methods on best config (MER, K=3) ------------------
    top3  = select_top_bands(scores, 'mer', 3)
    bids3 = [b for b, _ in top3]
    ws3   = [s for _, s in top3]
    # -- Group C: fusion methods on top-3 by MER -----------------------------
    for fusion in ('weighted', 'max'):
        exps.append({
            'label':        f"C_Top3_mer_{fusion}",
            'group':        'C_fusion_method',
            'band_ids':     bids3,
            'fusion':       fusion,
            'band_weights': ws3 if fusion == 'weighted' else None,
            'description':  (f"Top-3 by MER, {fusion} fusion, bands={bids3}"),
        })

    # -- Group D: Synergistic Multi-Scale & Edge Combos ---------------------
    synergies = [
        ([0, 1],    'mean', 'D_Raw_Gauss1_mean',   'Raw + Gauss s=1 mean fusion'),
        ([0, 7],    'max',  'D_Raw_LoG1_max',      'Raw + LoG s=1 edge max fusion'),
        ([1, 3, 7], 'max',  'D_Gauss1_3_LoG1_max', 'Multi-scale Gauss + LoG max fusion'),
    ]
    for bids, fus, lbl, desc in synergies:
        exps.append({
            'label':        lbl,
            'group':        'D_synergy',
            'band_ids':     bids,
            'fusion':       fus,
            'band_weights': None,
            'description':  desc,
        })

    return exps



# ---------------------------------------------------------------------------
# Study Runner
# ---------------------------------------------------------------------------

def run_study(frames_dir, init_path, gt_path, output_dir,
              scores, experiments,
              scale_x=5.0, scale_y=5.0,
              threshold=60.0, thresh_pct=97.0,
              vel_alpha=0.6, max_search_dist=600.0,
              fps=25, write_video=False):
    """Run all experiments sequentially; return list of result dicts."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    total   = len(experiments)

    print("\nPre-loading frames into RAM for fast multi-experiment execution...")
    frame_paths = sorted(
        _glob.glob(os.path.join(frames_dir, '*.png')),
        key=natural_sort_key)
    preloaded = [load_frame(p) for p in frame_paths]
    print(f"Loaded {len(preloaded)} frames into memory.")

    for ei, exp in enumerate(experiments):
        label    = exp['label']
        band_ids = exp['band_ids']
        fusion   = exp['fusion']
        bw       = exp['band_weights']

        bnames = [BAND_NAMES[b] for b in band_ids]
        print(f"\n[{ei+1:2d}/{total}] {label}")
        print(f"         bands  : {band_ids} ({bnames})")
        print(f"         fusion : {fusion}")

        vid_path = os.path.join(output_dir, f"{label}.mp4") if write_video else ""

        t0 = time.time()
        res = run_band_tracker(
            frames_dir=frames_dir,  init_path=init_path,
            output_path=vid_path,   band_ids=band_ids,
            fusion=fusion,          band_weights=bw,
            gt_path=gt_path,        fps=fps,
            scale_x=scale_x,       scale_y=scale_y,
            threshold=threshold,    thresh_pct=thresh_pct,
            vel_alpha=vel_alpha,    max_search_dist=max_search_dist,
            write_video=write_video, verbose=False,
            preloaded_frames=preloaded,
        )
        elapsed = time.time() - t0

        res.update({
            'label':       label,
            'group':       exp['group'],
            'band_ids':    band_ids,
            'band_names':  bnames,
            'fusion':      fusion,
            'description': exp['description'],
            'runtime_s':   elapsed,
        })
        results.append(res)

        print(f"         MeanIoU={res['mean_iou']:.4f}  "
              f"MeanCE={res['mean_ce']:.1f}px  "
              f"FB={res['fallback_rate']:.1f}%  "
              f"({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results_csv(results, output_dir):
    path = os.path.join(output_dir, 'results.csv')
    cols = ['label', 'group', 'band_ids', 'band_names', 'fusion',
            'mean_iou', 'median_iou', 'iou_gt_0_25', 'iou_gt_0_50',
            'mean_ce', 'median_ce', 'fallback_rate', 'runtime_s', 'description']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = dict(r)
            row['band_ids']   = '+'.join(str(b) for b in r['band_ids'])
            row['band_names'] = '+'.join(r['band_names'])
            w.writerow(row)
    print(f"\nSaved: {path}")
    return path


def save_band_scores_csv(scores, output_dir):
    path = os.path.join(output_dir, 'band_scores.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band_id','name','description','mer','snr','fisher','bhat','tc',
                    'mer_std','snr_std','fisher_std','bhat_std'])
        for bid in BAND_IDS:
            sc = scores[bid]
            w.writerow([bid, BAND_NAMES[bid], BAND_DESCRIPTIONS[bid],
                        f"{sc['mer']:.4f}", f"{sc['snr']:.4f}",
                        f"{sc['fisher']:.4f}", f"{sc['bhat']:.4f}",
                        f"{sc['tc']:.4f}",
                        f"{sc.get('mer_std',0):.4f}", f"{sc.get('snr_std',0):.4f}",
                        f"{sc.get('fisher_std',0):.4f}", f"{sc.get('bhat_std',0):.4f}"])
    print(f"Saved: {path}")


def print_results_table(results):
    """Print a formatted comparison table sorted by mean IoU."""
    sorted_res = sorted(results, key=lambda r: r['mean_iou'], reverse=True)
    sep = '=' * 112
    hdr = (f"{'Rank':>4}  {'Label':<32}  {'Bands':>6}  "
           f"{'MeanIoU':>8}  {'MedIoU':>7}  {'IoU>0.5':>7}  {'IoU>0.25':>8}  "
           f"{'MeanCE':>7}  {'FB%':>5}")
    print(f"\n{sep}\n{hdr}\n{sep}")
    for rank, r in enumerate(sorted_res, 1):
        bids = '+'.join(str(b) for b in r['band_ids'])
        print(f"{rank:>4}  {r['label']:<32}  {bids:>6}  "
              f"{r['mean_iou']:>8.4f}  {r['median_iou']:>7.4f}  "
              f"{r['iou_gt_0_50']:>6.1f}%  {r['iou_gt_0_25']:>7.1f}%  "
              f"{r['mean_ce']:>7.1f}  {r['fallback_rate']:>4.1f}%")
    print(sep)

    best_iou = max(results, key=lambda r: r['mean_iou'])
    best_ce  = min(results, key=lambda r: r['mean_ce'])
    best_fb  = min(results, key=lambda r: r['fallback_rate'])
    bl       = next((r for r in results if r['band_ids'] == [0]), None)

    print(f"\n  Best Mean IoU   : {best_iou['label']}  ({best_iou['mean_iou']:.4f})")
    print(f"  Best Mean CE    : {best_ce['label']}  ({best_ce['mean_ce']:.1f} px)")
    print(f"  Best Fallback % : {best_fb['label']}  ({best_fb['fallback_rate']:.1f} %)")
    if bl:
        imp = (best_iou['mean_iou'] - bl['mean_iou']) / (bl['mean_iou'] + 1e-8) * 100
        print(f"  Baseline (raw)  : IoU={bl['mean_iou']:.4f}  "
              f"CE={bl['mean_ce']:.1f}px")
        print(f"  Best vs baseline: +{imp:.1f}% IoU")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    'A_single_band'  : '#4C8BE8',
    'B_topK_mer'     : '#E84C4C',
    'B_topK_snr'     : '#27AE60',
    'B_topK_fisher'  : '#E8B44C',
    'C_fusion_method': '#9B59B6',
}

def _gcolor(group):
    for k, v in GROUP_COLORS.items():
        if k in group:
            return v
    return '#888888'


def plot_iou_comparison(results, output_dir):
    if not HAS_MPL:
        return
    labels = [r['label'] for r in results]
    ious   = [r['mean_iou'] for r in results]
    colors = [_gcolor(r['group']) for r in results]

    fig, ax = plt.subplots(figsize=(max(14, len(labels)*0.52), 6))
    ax.bar(range(len(labels)), ious, color=colors, edgecolor='white', linewidth=0.4)

    bl = next((r for r in results if r['band_ids'] == [0]), None)
    if bl:
        ax.axhline(bl['mean_iou'], color='black', linewidth=1.5, linestyle='--',
                   label=f"Baseline raw B0: {bl['mean_iou']:.3f}")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('Mean IoU', fontsize=12)
    ax.set_title('Comparative Study: Mean IoU by Band Configuration',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(ious)*1.18)
    ax.grid(axis='y', alpha=0.3)

    patches = [mpatches.Patch(color=v, label=k.replace('_',' '))
               for k, v in GROUP_COLORS.items()]
    ax.legend(handles=patches + (ax.get_lines()[:1] if bl else []),
              fontsize=8, loc='upper right')
    plt.tight_layout()
    p = os.path.join(output_dir, 'iou_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")


def plot_ce_comparison(results, output_dir):
    if not HAS_MPL:
        return
    labels = [r['label'] for r in results]
    ces    = [r['mean_ce'] for r in results]
    colors = [_gcolor(r['group']) for r in results]

    fig, ax = plt.subplots(figsize=(max(14, len(labels)*0.52), 6))
    ax.bar(range(len(labels)), ces, color=colors, edgecolor='white', linewidth=0.4)

    bl = next((r for r in results if r['band_ids'] == [0]), None)
    if bl:
        ax.axhline(bl['mean_ce'], color='black', linewidth=1.5, linestyle='--',
                   label=f"Baseline raw B0: {bl['mean_ce']:.1f} px")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('Mean Centre Error (px)', fontsize=12)
    ax.set_title('Comparative Study: Mean Centre Error by Band Configuration',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    patches = [mpatches.Patch(color=v, label=k.replace('_',' '))
               for k, v in GROUP_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc='upper right')
    plt.tight_layout()
    p = os.path.join(output_dir, 'ce_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")


def plot_band_score_heatmap(scores, output_dir):
    if not HAS_MPL:
        return
    crits = ['mer', 'snr', 'fisher', 'bhat', 'tc']
    data  = np.array([[scores[b][c] for c in crits] for b in BAND_IDS])

    # Normalise each column to [0,1] for visual comparison across criteria
    col_min  = data.min(axis=0)
    col_max  = data.max(axis=0)
    data_n   = (data - col_min) / (col_max - col_min + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data_n, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(len(crits)))
    ax.set_xticklabels([c.upper() for c in crits], fontsize=11)
    ax.set_yticks(range(N_BANDS))
    ax.set_yticklabels([f"B{b}: {BAND_NAMES[b]}" for b in BAND_IDS], fontsize=9)

    for i in range(N_BANDS):
        for j in range(len(crits)):
            txt_col = 'white' if data_n[i, j] > 0.65 else 'black'
            ax.text(j, i, f"{data[i,j]:.2f}", ha='center', va='center',
                    fontsize=7.5, color=txt_col, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Normalised Score')
    ax.set_title('Band Scoring Heatmap (normalised per criterion)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Scoring Criterion', fontsize=11)
    ax.set_ylabel('Virtual Band', fontsize=11)
    plt.tight_layout()
    p = os.path.join(output_dir, 'band_score_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")


def plot_topk_effect(results, k_values, output_dir):
    """Line plot: Mean IoU and CE vs K for each scoring criterion."""
    if not HAS_MPL:
        return
    crits      = ['mer', 'snr', 'fisher']
    crit_cols  = {'mer': '#E84C4C', 'snr': '#27AE60', 'fisher': '#E8B44C'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for crit in crits:
        iou_k, ce_k = [], []
        for k in k_values:
            lbl = f"B_Top{k}_{crit}_mean"
            row = next((r for r in results if r['label'] == lbl), None)
            iou_k.append(row['mean_iou'] if row else float('nan'))
            ce_k.append(row['mean_ce']   if row else float('nan'))
        c = crit_cols[crit]
        ax1.plot(k_values, iou_k, 'o-', color=c, label=crit.upper(),
                 linewidth=2, markersize=7)
        ax2.plot(k_values, ce_k,  'o-', color=c, label=crit.upper(),
                 linewidth=2, markersize=7)

    bl = next((r for r in results if r['band_ids'] == [0]), None)
    if bl:
        ax1.axhline(bl['mean_iou'], color='black', linestyle='--',
                    linewidth=1.5, label='Baseline (raw B0)')
        ax2.axhline(bl['mean_ce'],  color='black', linestyle='--',
                    linewidth=1.5, label='Baseline (raw B0)')

    for ax, ylabel, title in [
        (ax1, 'Mean IoU',              'Mean IoU vs. Number of Bands (K)'),
        (ax2, 'Mean Centre Error (px)', 'Centre Error vs. Number of Bands (K)'),
    ]:
        ax.set_xlabel('K (number of bands fused)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(k_values)

    plt.tight_layout()
    p = os.path.join(output_dir, 'topk_effect.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")


def plot_fusion_comparison(results, output_dir):
    """Bar chart comparing mean/weighted/max fusion for Top-3 MER bands."""
    if not HAS_MPL:
        return
    configs = [
        ('B_Top3_mer_mean',    'Top-3 MER\n(mean)'),
        ('C_Top3_mer_weighted','Top-3 MER\n(weighted)'),
        ('C_Top3_mer_max',     'Top-3 MER\n(max)'),
    ]
    rows = []
    for lbl, disp in configs:
        r = next((x for x in results if x['label'] == lbl), None)
        if r:
            rows.append((disp, r['mean_iou'], r['mean_ce'], r['fallback_rate']))
    if not rows:
        return

    disps, ious, ces, fbs = zip(*rows)
    x   = np.arange(len(disps))
    pal = ['#4C8BE8', '#E84C4C', '#27AE60']

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, vals, lbl_y, title_s in [
        (axes[0], ious, 'Mean IoU',            'IoU by Fusion Method'),
        (axes[1], ces,  'Mean CE (px)',         'Centre Error by Fusion Method'),
        (axes[2], fbs,  'Fallback Rate (%)',    'Fallback Rate by Fusion Method'),
    ]:
        bars = ax.bar(x, vals, color=pal[:len(vals)])
        ax.set_xticks(x)
        ax.set_xticklabels(disps, fontsize=9)
        ax.set_ylabel(lbl_y)
        ax.set_title(title_s, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.3f}" if 'IoU' in lbl_y else f"{v:.1f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Effect of Fusion Method (Top-3 MER Bands)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(output_dir, 'fusion_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")


def plot_mer_vs_iou(results, scores, output_dir):
    """Scatter: single-band MER score vs. tracking Mean IoU."""
    if not HAS_MPL:
        return
    single = [r for r in results if r['group'] == 'A_single_band']
    if not single:
        return

    mer_vals = [scores[r['band_ids'][0]]['mer'] for r in single]
    iou_vals = [r['mean_iou'] for r in single]
    names    = [BAND_NAMES[r['band_ids'][0]] for r in single]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mer_vals, iou_vals, s=90, zorder=5, color='#4C8BE8',
               edgecolors='white', linewidths=0.8)
    for mx, iy, nm in zip(mer_vals, iou_vals, names):
        ax.annotate(nm, (mx, iy), textcoords='offset points',
                    xytext=(5, 3), fontsize=8)

    if len(mer_vals) >= 2:
        m, b = np.polyfit(mer_vals, iou_vals, 1)
        xs   = np.linspace(min(mer_vals), max(mer_vals), 50)
        corr = float(np.corrcoef(mer_vals, iou_vals)[0, 1])
        ax.plot(xs, m*xs+b, 'r--', alpha=0.6,
                label=f"Trend  r={corr:.2f}  slope={m:.4f}")
        ax.legend(fontsize=9)

    ax.set_xlabel('Motion Energy Ratio (MER) Score', fontsize=11)
    ax.set_ylabel('Tracking Mean IoU', fontsize=11)
    ax.set_title('Band MER Score vs. Achieved Tracking IoU',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'mer_vs_iou.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="HSI Band Selection Comparative Study")
    p.add_argument('frames_dir')
    p.add_argument('init_rect')
    p.add_argument('--gt',              required=True)
    p.add_argument('--scale_x',         type=float, default=5.0)
    p.add_argument('--scale_y',         type=float, default=5.0)
    p.add_argument('--n_score_frames',  type=int,   default=50)
    p.add_argument('--output_dir',      default='band_study_results')
    p.add_argument('--no_video',        action='store_true',
                   help="Skip video writing for all experiments (much faster)")
    p.add_argument('--threshold',       type=float, default=60.0)
    p.add_argument('--thresh_pct',      type=float, default=97.0)
    p.add_argument('--vel_alpha',       type=float, default=0.6)
    p.add_argument('--max_search_dist', type=float, default=600.0)
    p.add_argument('--fps',             type=int,   default=25)
    p.add_argument('--k_values',        default='1,2,3,5,12',
                   help="Comma-separated K values for top-K study (default 1,2,3,5,12)")
    args = p.parse_args()

    k_values    = [int(k) for k in args.k_values.split(',')]
    write_video = not args.no_video
    os.makedirs(args.output_dir, exist_ok=True)

    banner = '=' * 60
    print(f"\n{banner}")
    print(f"  HSI BAND SELECTION COMPARATIVE STUDY")
    print(f"{banner}")
    print(f"  Frames dir   : {args.frames_dir}")
    print(f"  GT file      : {args.gt}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Score frames : {args.n_score_frames}")
    print(f"  K values     : {k_values}")
    print(f"  Write video  : {write_video}")

    # 1. Frame list
    import glob as _glob, re
    frame_paths = sorted(
        _glob.glob(os.path.join(args.frames_dir, '*.png')),
        key=natural_sort_key)
    if not frame_paths:
        print(f"ERROR: No PNG frames in '{args.frames_dir}'"); sys.exit(1)
    print(f"\nFound {len(frame_paths)} frames.")

    # 2. GT in pixel space
    gt_boxes = load_groundtruth(args.gt, args.scale_x, args.scale_y)
    print(f"Loaded {len(gt_boxes)} GT boxes.")

    # --- Phase 1: Band Scoring ---
    print(f"\n{'-'*40}")
    print("  Phase 1: Band Scoring")
    print(f"{'-'*40}")
    scores = score_all_bands(frame_paths, gt_boxes,
                             n_sample=args.n_score_frames, verbose=True)
    print_band_scores(scores)
    save_band_scores_csv(scores, args.output_dir)

    # --- Phase 2: Build Experiments ---
    print(f"\n{'-'*40}")
    print("  Phase 2: Experiment Configurations")
    print(f"{'-'*40}")
    experiments = build_experiments(scores, k_values=k_values)
    print(f"Total: {len(experiments)} experiments")
    for i, e in enumerate(experiments):
        print(f"  {i+1:2d}. {e['label']}")

    # --- Phase 3: Run Experiments ---
    print(f"\n{'-'*40}")
    print("  Phase 3: Running All Configurations")
    print(f"{'-'*40}")
    all_results = run_study(
        frames_dir=args.frames_dir,  init_path=args.init_rect,
        gt_path=args.gt,             output_dir=args.output_dir,
        scores=scores,               experiments=experiments,
        scale_x=args.scale_x,       scale_y=args.scale_y,
        threshold=args.threshold,    thresh_pct=args.thresh_pct,
        vel_alpha=args.vel_alpha,    max_search_dist=args.max_search_dist,
        fps=args.fps,                write_video=write_video,
    )

    # --- Phase 4: Results ---
    print(f"\n{'-'*40}")
    print("  Phase 4: Results Summary")
    print(f"{'-'*40}")
    print_results_table(all_results)
    save_results_csv(all_results, args.output_dir)

    # --- Phase 5: Plots ---
    print(f"\n{'-'*40}")
    print("  Phase 5: Generating Plots")
    print(f"{'-'*40}")
    plot_band_score_heatmap(scores, args.output_dir)
    plot_iou_comparison(all_results, args.output_dir)
    plot_ce_comparison(all_results, args.output_dir)
    plot_topk_effect(all_results, k_values, args.output_dir)
    plot_fusion_comparison(all_results, args.output_dir)
    plot_mer_vs_iou(all_results, scores, args.output_dir)

    # --- Summary ---
    best     = max(all_results, key=lambda r: r['mean_iou'])
    baseline = next((r for r in all_results if r['band_ids']==[0]), None)
    print(f"\n{banner}")
    print(f"  STUDY COMPLETE")
    print(f"  Best config  : {best['label']}")
    print(f"  Bands        : {best['band_names']}")
    print(f"  Fusion       : {best['fusion']}")
    print(f"  Mean IoU     : {best['mean_iou']:.4f}")
    print(f"  Mean CE      : {best['mean_ce']:.1f} px")
    if baseline:
        imp = (best['mean_iou'] - baseline['mean_iou']) / (baseline['mean_iou']+1e-8)*100
        print(f"  Baseline IoU : {baseline['mean_iou']:.4f}  (+{imp:.1f}% improvement)")
    print(f"  All outputs  : {args.output_dir}/")
    print(f"{banner}")


if __name__ == '__main__':
    main()
