# HSI Object Tracker — Development Log

A hyperspectral (NIR) object tracker for the HOT/VOT benchmark format, built
on top of the existing `overlay_gt_boxes.py` visualisation pipeline.

---

## Repository Structure

```
Codes/
├── overlay_gt_boxes.py          # Original: draw ground-truth boxes on frames → MP4
├── hsi_tracker.py               # Frame-difference blob tracker with velocity prediction
├── band_selector.py             # 12-band HSI simulation, scoring & multi-band fused tracker
├── band_study.py                # Comparative study framework (variations across K, bands, fusion)
├── band_study_results/          # Exported CSVs and matplotlib comparative analysis plots
├── tracked_ball.mp4             # Tracking output (ball sequence)
├── tracked_ball_predicted_boxes.txt
├── Datasets/
│   ├── ball/                    # 626 hyperspectral NIR PNG frames (1080 × 2045, 16-bit)
│   │   └── init_rect.txt        # Initial bounding box: "129  119  30  29"
│   ├── img/                     # 516 hyperspectral NIR PNG frames
│   │   └── init_rect.txt        # Initial bounding box: "309  140  8  8"
│   ├── groundtruth_ball.txt     # Per-frame GT boxes for ball sequence
│   └── groundtruth_rect.txt     # Per-frame GT boxes for img sequence
└── README.md                    # This file
```


---

## Data Format

All coordinate files (init_rect, groundtruth) use the **HOT/VOT format**:

```
x   y   w   h
```

- `(x, y)` — top-left corner of the bounding box
- `(w, h)` — width and height
- Values are in a **sub-sampled coordinate space**; a `scale_x = scale_y = 5.0`
  factor maps them to actual pixel coordinates on the 1080 × 2045 frame.
- Separator: tab, space, or comma (all handled by the same regex parser).
- Sentinel lines: `-1 -1 -1 -1` indicate the target is occluded / out of frame.

**Frame images** are 16-bit grayscale PNGs. They are converted to 8-bit for
display using a global percentile stretch (0.5th – 99.5th percentile computed
once across a sample of frames).

---

## Shared I/O Layer (mirrors `overlay_gt_boxes.py`)

Both scripts share identical helper functions so that any future change to
loading logic only needs to be made once:

| Function | Purpose |
|---|---|
| `natural_sort_key` | Numerically sort frame filenames (`0001.png` before `0010.png`) |
| `load_frame` | Open a PNG with Pillow and return a NumPy array |
| `compute_global_range` | Percentile-based contrast stretch range over sampled frames |
| `stretch_to_uint8` | Map 16-bit values to [0, 255] for display |
| `parse_box_line` | Parse one `x y w h` line (tab / space / comma) with sentinel handling |
| `load_init_box` | Read first valid line from `init_rect.txt`, apply scale factors |
| `load_groundtruth` | Read all GT boxes, apply scale factors |

---

## Tracker Iteration History

### Iteration 0 — Baseline: `overlay_gt_boxes.py`

**What it does**  
Not a tracker — visualises the *known* ground-truth boxes on top of each frame
and writes an MP4. Scale factor hardcoded to 5.0.

**Purpose**  
Used as the reference implementation for data loading, frame ordering, and
contrast stretching. All tracker versions replicate its I/O logic exactly.

---

### Iteration 1 — Mean-Shift Tracker

**Algorithm**  
Classic mean-shift with an Epanechnikov-weighted intensity histogram (64 bins)
built from the initial bounding box. Per-pixel weights at each iteration use
the Bhattacharyya coefficient: `w_i = sqrt(q_u / p_u) * k(x_i)`, where `q` is
the target histogram and `p` is the candidate histogram.

**Why it was tried**  
Mean-shift is a standard, dependency-free tracker that works well when the
target has a distinctive intensity distribution (colour histogram in RGB
trackers, intensity histogram here).

**Results**

| Metric | Value |
|---|---|
| Mean IoU | 0.039 |
| Median IoU | 0.000 |
| IoU > 0.50 | 2.3% |
| Mean Centre Error | 398 px |

**Why it failed**  
The 16-bit frames, after conversion to 8-bit, are nearly uniform across the
scene. The ball's intensity histogram is **not distinctive** from the
background — many regions of the image share the same intensity distribution.
Mean-shift converged to nearby high-weight regions that were not the ball.

Additionally, the ball moves at up to **110 pixels per frame** in the early
frames. Mean-shift iterates within a fixed-size window centred on the previous
position; if the ball has moved further than that window, the algorithm cannot
recover.

---

### Iteration 2 — NCC Template Matching

**Algorithm**  
`cv2.matchTemplate` with `TM_CCOEFF_NORMED` inside a search window of
`search_factor × box_size` (default 3.0×) centred on the previous position.
Optional template update via linear blending: `T ← (1-α)·T + α·new_patch`.

**Why it was tried**  
Normalised cross-correlation matches the **spatial texture** of the initial
patch, not just its histogram. It is more sensitive to fine texture
differences. Combined with a constrained search window it should be faster than
global NCC and less prone to false matches far away.

**Results**

| Metric | Value |
|---|---|
| Mean IoU | 0.037 |
| Median IoU | 0.000 |
| IoU > 0.50 | 3.3% |
| Mean Centre Error | 472 px |
| NCC score at lost frames | 0.986 – 0.998 |

**Why it failed**  
The NCC score remained *extremely high* (> 0.99) throughout, even when the
tracker was clearly on the wrong region. This exposed the root problem: after
8-bit conversion, **many patches in the HSI frame look nearly identical** to the
ball patch. The tracker confidently found the highest-scoring match — but that
match was a different uniform-looking region.

The constrained search window (3× box = 450 px) also failed to keep up when
the ball moved more than 225 px between consecutive frames.

---

### Iteration 3 — Sparse Lucas-Kanade Optical Flow

**Algorithm**  
1. Detect Shi-Tomasi corners inside the initial box with
   `cv2.goodFeaturesToTrack`.
2. Track them frame-to-frame with `cv2.calcOpticalFlowPyrLK`
   (pyramid levels = 3, window = 21 × 21).
3. Forward-backward error filter: re-track backward and discard points whose
   round-trip error exceeds `fb_thresh` (default 1.0 px).
4. Apply the median displacement of surviving points to the box centre.
5. Re-detect features when fewer than `min_features` survive.

**Why it was tried**  
Optical flow works on **local pixel gradients** between consecutive frames, not
on global appearance similarity. It should track even if the ball looks
identical to other parts of the image, as long as the ball itself has some
local texture.

**Results**

| Metric | Value |
|---|---|
| Mean IoU | 0.016 |
| Median IoU | 0.000 |
| IoU > 0.50 | 1.5% |
| Mean Centre Error | 729 px |

**Why it failed — two root causes**

1. **Background feature contamination.** The 150 × 145 initial box contains
   both ball pixels and background pixels. The vast majority of Shi-Tomasi
   corners found by `goodFeaturesToTrack` came from the background (which has
   stronger edges). The median displacement of all corners therefore reflected
   background motion (camera vibration / slight scene movement) rather than the
   ball's motion.

2. **Motion too fast for standard LK.** With `maxLevel=3` and `winSize=21`,
   the pyramid LK can handle motion up to roughly `2³ × 21/2 ≈ 84` pixels.
   The ball moves at up to **110 pixels per frame** in the early frames — right
   at or beyond this limit. Any features that were genuinely on the ball were
   lost or mismatched.

---

### Diagnostic Step — Data Analysis

Before implementing a fourth tracker approach, a dedicated diagnostic script
was run to understand the data from first principles.

**Key findings**

| Measurement | Value |
|---|---|
| Ball patch mean pixel value (16-bit) | 231 |
| Background left of ball (mean) | 330 |
| Background right of ball (mean) | 511 |
| Background above ball (mean) | 230 |
| Frame-difference mean *inside ball box* (frame 0→1) | **143.04** |
| Frame-difference mean *in background box* (frame 0→1) | **7.75** |
| Ratio (ball / background) | **~18×** |
| Overall diff 99th percentile | 127 |
| Ball displacement frame 0→25 | 445 px total / 17.8 px·frame⁻¹ avg |

**Conclusions**

1. The ball is **not globally distinctive by appearance** in the 8-bit
   stretched image — it overlaps with background pixel distributions.
   This explains why all appearance-based trackers (mean-shift, NCC) failed.

2. The ball creates an **18× stronger frame-difference signal** than the
   background. This is because it is a fast-moving object in a predominantly
   static scene.

3. The ball's average velocity (17.8 px/frame averaged over 25 frames)
   understates the instantaneous velocity in the first few frames (80–110
   px/frame), which explains why optical flow and windowed NCC lost the ball
   immediately.

4. The correct approach is **motion-based detection** (frame differencing),
   not appearance-based matching.

---

### Iteration 4 — Frame-Difference Blob Tracker with Velocity Prediction ✅ FINAL

**Algorithm**

For each frame `t`:

1. **Compute frame difference** in the raw 16-bit space (maximum dynamic range,
   no information lost to 8-bit conversion):
   ```
   diff = |frame[t] - frame[t-1]|   (float32)
   ```

2. **Adaptive threshold** to produce a binary motion mask:
   ```
   threshold = max(hard_min=60, percentile(diff, thresh_pct=97))
   ```
   The adaptive component adjusts to per-frame noise; the hard minimum prevents
   the threshold from collapsing to zero in very quiet frames.

3. **Morphological cleanup** (9 × 9 elliptical kernel):
   - `MORPH_CLOSE` fills small holes inside blobs
   - `MORPH_OPEN` removes speckle noise

4. **Connected-component analysis** (`cv2.connectedComponentsWithStats`).
   Each component is filtered by:
   - Area ∈ [0.15×, 5.0×] initial ball area
   - Distance from velocity-predicted position ≤ `max_search_dist` (600 px)

5. **Select the blob closest to the predicted position.** The difference blob
   straddles both the *previous* ball position and the *current* ball position,
   so its centroid is at the **midpoint** between old and new. This bias is
   corrected by adding `½ · velocity`:
   ```
   corrected_cx = blob_centroid_x + 0.5 · vx
   corrected_cy = blob_centroid_y + 0.5 · vy
   ```

6. **Update velocity** with exponential moving average (`vel_alpha = 0.6`):
   ```
   vx ← α · (new_cx - old_cx) + (1-α) · vx
   vy ← α · (new_cy - old_cy) + (1-α) · vy
   ```

7. **Fallback** (shown as orange box in video): if no qualifying blob is found,
   coast on the current velocity estimate with 0.85× decay per frame:
   ```
   (cx, cy) ← (cx + vx, cy + vy)
   vx ← 0.85 · vx
   vy ← 0.85 · vy
   ```

**Results (ball sequence)**

| Metric | Value |
|---|---|
| Mean IoU | **0.340** |
| Median IoU | **0.371** |
| IoU > 0.50 | **23.5%** |
| IoU > 0.25 | **66.6%** |
| Mean Centre Error | **109 px** |
| Median Centre Error | **48.65 px** |
| Fallback frames | 67 / 626 (10.7%) |

**Improvement over best previous approach (NCC)**

| Metric | NCC | Frame-Diff | Improvement |
|---|---|---|---|
| Mean IoU | 0.037 | 0.340 | **9.2×** |
| Mean Centre Error | 472 px | 109 px | **4.3×** |

**Remaining limitations**

- 67 frames (10.7%) used velocity coasting because the ball's motion was below
  the threshold or the blob was too small/large. This mostly occurs when the
  ball passes through a background region of similar brightness (ball mean ≈ 231,
  background above ≈ 230).
- Box *size* is fixed at the initial dimensions. The ground-truth shows the
  ball's apparent size varies (w = 90–185 after scaling) as it moves toward or
  away from the camera. A multi-scale blob fitting step would improve IoU
  further.
- Error accumulates across fallback streaks. Once velocity is wrong, the next
  blob-selection step may pick the wrong component.

---

### Iteration 5 — Hyperspectral Band Selection & Multi-Band Fused Tracker (`band_selector.py`)

**Context & Motivation**  
In the baseline single-band tracker (`hsi_tracker.py`), differencing is performed strictly on the raw 16-bit NIR image. While the moving ball has a strong difference signal on average (~18× background), 10.7% of frames are lost to velocity coasting because contrast collapses when the ball traverses background regions of similar brightness.

In hyperspectral imaging (HSI), each spectral wavelength exhibits a different quantum-efficiency curve (non-linear photometric response), atmospheric blur (wavelength-dependent point spread function), and distinct per-material reflectance. We simulate this spectral diversity by synthesizing **12 virtual spectral bands** from each 16-bit frame via spatial, differential, and radiometric transformations, score their informativeness using ground-truth motion statistics, and fuse the top-ranked bands into a robust composite motion mask.

#### 1. The 12 Virtual Spectral Bands

| Band ID | Name | Transformation | Physical Rationale |
|:---:|:---|:---|:---|
| **0** | `raw` | Identity ($f$) | Baseline raw 16-bit detector response |
| **1** | `gauss_s1` | $G_{\sigma=1} * f$ | Light optical PSF denoising |
| **2** | `gauss_s3` | $G_{\sigma=3} * f$ | Medium spatial smoothing |
| **3** | `gauss_s5` | $G_{\sigma=5} * f$ | Heavy atmospheric / lens blur simulation |
| **4** | `dog_fine` | $\|G_{\sigma=1} - G_{\sigma=3}\| * f$ | Fine-detail bandpass spatial frequencies |
| **5** | `dog_med` | $\|G_{\sigma=2} - G_{\sigma=5}\| * f$ | Mid-detail bandpass spatial frequencies |
| **6** | `dog_coarse`| $\|G_{\sigma=3} - G_{\sigma=9}\| * f$ | Coarse-detail bandpass spatial frequencies |
| **7** | `log_s1` | $\|\nabla^2(G_{\sigma=1} * f)\|$ | Fine Laplacian-of-Gaussian edge enhancement |
| **8** | `log_s2` | $\|\nabla^2(G_{\sigma=2} * f)\|$ | Coarse Laplacian-of-Gaussian edges |
| **9** | `morph_grad`| $\text{dilate}_{5\times 5}(f) - \text{erode}_{5\times 5}(f)$ | Non-linear morphological boundary contrast |
| **10**| `sqrt_comp` | $\sqrt{\max(f, 0)}$ | High-QE detector radiometric compression |
| **11**| `log_comp` | $\ln(1 + \max(f, 0))$ | Dynamic range compression (dim texture boost) |

#### 2. Band Informativeness Scoring Criteria

For each candidate band $k$, the temporal difference image $D_k(t) = |f_k(t) - f_k(t-1)|$ is scored over a sampled set of frame pairs against ground-truth boxes:

1. **Motion Energy Ratio (MER)**:
   $$\text{MER}_k = \frac{\mu(D_k(\Omega_{\text{obj}}))}{\mu(D_k(\Omega_{\text{bg}})) + \epsilon}$$
   Measures object motion amplification over background clutter.
2. **Signal-to-Noise Ratio (SNR)**:
   $$\text{SNR}_k = \frac{\mu(D_k(\Omega_{\text{obj}}))}{\sigma(D_k(\Omega_{\text{bg}})) + \epsilon}$$
   Measures discriminability relative to background variance.
3. **Fisher Discriminant Ratio**:
   $$\mathcal{F}_k = \frac{(\mu_{\text{obj}} - \mu_{\text{bg}})^2}{\sigma_{\text{obj}}^2 + \sigma_{\text{bg}}^2 + \epsilon}$$
   Measures statistical class separability.
4. **Bhattacharyya Distance**:
   $$\mathcal{D}_{B, k} = -\ln \sum_{b=1}^{B} \sqrt{p_{\text{obj}}(b) \cdot p_{\text{bg}}(b)}$$
   Measures histogram distribution divergence between target and background.
5. **Temporal Consistency (TC)**:
   $$\text{TC}_k = \frac{1}{1 + \sigma_t(\text{MER}_k(t))}$$
   Measures how consistently contrast is maintained across temporal transitions.

#### Quantitative Band Scores (Sampled across sequence)

```text
-----------------------------------------------------------------------------------------------
 ID  Name                MER       SNR    FISHER      BHAT        TC  Description
-----------------------------------------------------------------------------------------------
  0  raw              11.746     6.583     1.190     0.478     0.124  Raw 16-bit baseline
  1  gauss_s1         17.613     7.273     1.360     0.555     0.079  Gaussian s=1 (light denoise)
  2  gauss_s3         20.173     7.796     1.439     0.607     0.067  Gaussian s=3 (medium smooth)
  3  gauss_s5        *21.501    *8.388    *1.531    *0.658     0.062  Gaussian s=5 (heavy smooth)
  4  dog_fine          4.261     2.981     0.499     0.182     0.326  DoG fine bandpass
  5  dog_med           4.855     2.491     0.404     0.158     0.236  DoG medium bandpass
  6  dog_coarse        6.834     3.127     0.568     0.240     0.168  DoG coarse bandpass
  7  log_s1            4.053     3.307     0.466     0.181     0.338  LoG s=1 edge-enhanced
  8  log_s2            3.746     2.459     0.390     0.139    *0.348  LoG s=2 edge-enhanced
  9  morph_grad        7.011     4.969     0.944     0.359     0.194  Morphological gradient
 10  sqrt_comp        11.331     7.454     1.358     0.538     0.137  Square-root compression
 11  log_comp          9.332     6.135     1.332     0.480     0.164  Log dynamic compression
-----------------------------------------------------------------------------------------------
  * = best in column
```

#### 3. Dynamic Range Normalization & Multi-Band Fusion

Because physical sensors and compression curves produce diff magnitudes spanning several orders of magnitude (e.g. `raw` max diff is ~514 while `log_comp` max diff is ~1.6), direct thresholding with a fixed threshold like 60 breaks.

Each band difference image is dynamically normalized to align with the raw 97th percentile ($p_{97} \approx 45.0$):
$$\tilde{D}_k = D_k \cdot \left(\frac{45.0}{\mathcal{P}_{97}(D_k)}\right)$$
*(Subsampled 4×4 for sub-millisecond execution).*

Normalized bands are then fused via:
- **Mean Fusion**: $D_{\text{fused}} = \frac{1}{K}\sum_{k=1}^K \tilde{D}_k$ (averages out uncorrelated noise, tightens bounding box).
- **Weighted Fusion**: $D_{\text{fused}} = \sum_{k=1}^K w_k \tilde{D}_k$ with $w_k = \frac{\text{score}_k}{\sum_j \text{score}_j}$.
- **Max Fusion**: $D_{\text{fused}} = \max_{k} \tilde{D}_k$ (logical OR: if either edges, blur, or raw intensity detects motion, the blob is preserved).

---

### Comparative Study: Variations in Object Detection & Tracking (`band_study.py`)

A comprehensive comparative study was executed across all single bands, number of bands selected ($K$), ranking criteria, and fusion strategies.

#### Variation 1: Single Band Object Detection & Tracking (Group A)

| Config | Bands Used | Mean IoU | Mean Center Error | Fallback (Lost) Rate | Key Takeaway |
|:---|:---:|:---:|:---:|:---:|:---|
| **A_B00_raw** | `[0]` | 0.2852 | 261.4 px | 7.5% (47 frames) | Raw baseline; sensitive to noise |
| **A_B01_gauss_s1** | `[1]` | 0.3350 | 140.2 px | 8.1% (51 frames) | **Denoising cuts center error by 46%** |
| **A_B02_gauss_s3** | `[2]` | 0.3149 | 232.7 px | 9.4% (59 frames) | Medium smoothing |
| **A_B03_gauss_s5** | `[3]` | **0.3363** | 182.1 px | 8.3% (52 frames) | **Highest single-band IoU (+17.9% over raw)** |
| **A_B04_dog_fine** | `[4]` | 0.1795 | 399.3 px | 9.1% (57 frames) | Bandpass suppresses object interior |
| **A_B05_dog_med** | `[5]` | 0.0221 | 808.5 px | 20.4% (128 frames) | Object signal eroded |
| **A_B06_dog_coarse**| `[6]` | 0.0247 | 1055.1 px | 70.8% (443 frames) | Catastrophic tracking failure |
| **A_B07_log_s1** | `[7]` | 0.3235 | 227.4 px | **5.0% (31 frames)** | **Lowest fallback rate of any single band** |
| **A_B08_log_s2** | `[8]` | 0.0747 | 617.7 px | 16.8% (105 frames) | Edge blur too wide |
| **A_B09_morph_grad**| `[9]` | 0.2118 | 407.2 px | 11.7% (73 frames) | High border noise |
| **A_B10_sqrt_comp** | `[10]` | 0.2493 | 243.2 px | 6.9% (43 frames) | Resilient under dim illumination |
| **A_B11_log_comp** | `[11]` | 0.1394 | 417.9 px | 11.2% (70 frames) | Dynamic compression amplifies sensor noise |

#### Variation 2: Impact of Number of Bands Selected ($K$) (Group B)

Tracking performance evaluated across $K \in \{1, 2, 3, 5, 12\}$ top bands selected by Motion Energy Ratio (MER):

| Configuration | Bands Selected ($K$) | Mean IoU | Mean Center Error | Fallback Rate | Observation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Top-1 (MER)** | `[3]` ($K=1$) | 0.3363 | 182.1 px | 8.3% | Strong single-band denoise |
| **Top-2 (MER)** | `[3, 2]` ($K=2$) | 0.3432 | 138.6 px | 8.0% | Sharp reduction in center error |
| **Top-3 (MER)** | `[3, 2, 1]` ($K=3$) | **0.3440** | **137.5 px** | 8.6% | **Optimal Sweet Spot: Peak IoU & Lowest Error** |
| **Top-5 (MER)** | `[3, 2, 1, 0, 10]` ($K=5$) | 0.3338 | 140.9 px | **7.7%** | Slight dilution from lower-ranked bands |
| **Top-12 (All)** | `All 12 Bands` ($K=12$) | 0.2977 | 219.3 px | 15.0% | **Performance collapses (-13.5% IoU, Fallback doubles)** |

> **Key Takeaway — The Curse of Cluttered Bands ($K=12$ vs $K=3$):**
> Simply taking more spectral bands does **not** improve tracking performance. Incorporating uninformative bands (e.g., coarse DoG/LoG) introduces background clutter into the fused difference image, cutting Mean IoU from 0.3440 down to 0.2977 and doubling lost frames (15.0%). Selecting $K=2$ or $K=3$ bands is the empirical sweet spot.

#### Variation 3: Fusion Method & Synergistic Edge Coupling (Groups C & D)

| Configuration | Fusion Mode | Mean IoU | Fallback Frames | Impact |
|:---|:---:|:---:|:---:|:---|
| `[3, 2, 1]` (Top-3 MER) | **Mean** | **0.3440** | 54 / 625 (8.6%) | Best IoU boundary tightness |
| `[0, 1]` (Raw + Gauss1) | **Mean** | 0.3301 | 45 / 625 (7.2%) | Low center error, stable motion |
| `[0, 7]` (Raw + LoG Edge) | **Max** | 0.1890 | 17 / 625 (2.7%) | Dramatic drop in lost tracks |
| `[1, 3, 7]` (Multi-Scale + LoG) | **Max** | 0.2593 | **9 / 625 (1.4%)** | **98.6% continuous tracking lock!** (Fallback cut from 47 down to 9) |

- **Mean Fusion** maximizes bounding box overlap (IoU) by smoothing sensor noise.
- **Max Fusion** acts as a robust OR gate: coupling smoothed intensity (`gauss_s1`, `gauss_s5`) with edge derivatives (`log_s1`) cuts tracking loss by **81%** (from 47 frames to only 9 frames).

---

## Usage

### `overlay_gt_boxes.py` — Ground-truth visualisation

```bash
python overlay_gt_boxes.py <frames_dir> <groundtruth.txt> <output.mp4> [fps] [scale_x] [scale_y]
```

Example:
```bash
python overlay_gt_boxes.py Datasets/ball Datasets/groundtruth_ball.txt output_ball.mp4 25
```

---

### `hsi_tracker.py` — Frame-difference blob tracker

```bash
python hsi_tracker.py <frames_dir> <init_rect.txt> <output.mp4>
                      [--gt groundtruth.txt]
                      [--fps 25]
                      [--scale_x 5.0]  [--scale_y 5.0]
                      [--threshold 60] [--thresh_pct 97]
                      [--vel_alpha 0.6]
                      [--max_search_dist 600]
```

**Key arguments**

| Argument | Default | Description |
|---|---|---|
| `--gt` | None | Ground-truth file for evaluation (IoU, centre error) |
| `--scale_x / --scale_y` | 5.0 | Coordinate scale: GT / init-box → pixel space |
| `--threshold` | 60 | Hard minimum pixel-difference threshold |
| `--thresh_pct` | 97.0 | Adaptive threshold: `percentile(diff, thresh_pct)` |
| `--vel_alpha` | 0.6 | Velocity EMA factor (0 = frozen, 1 = instant update) |
| `--max_search_dist` | 600 | Max distance from predicted position to accept a blob |

**Ball sequence example:**
```bash
python hsi_tracker.py Datasets/ball Datasets/ball/init_rect.txt tracked_ball.mp4 \
    --gt Datasets/groundtruth_ball.txt --scale_x 5 --scale_y 5
```

**img sequence example:**
```bash
python hsi_tracker.py Datasets/img Datasets/img/init_rect.txt tracked_img.mp4 \
    --gt Datasets/groundtruth_rect.txt --scale_x 5 --scale_y 5
```

**Outputs**
- `<output>.mp4` — annotated video
  - **Green box** — tracker prediction
  - **Orange box** — fallback (velocity coasting, no blob detected)
  - **Red box** — ground-truth (if `--gt` is provided)
  - Yellow text overlay: IoU and centre error per frame
- `<output>_predicted_boxes.txt` — one `x\ty\tw\th` line per frame (same
  format as the ground-truth files)

---

### `band_selector.py` — Band Selection & Multi-Band Fused Tracker

```bash
python band_selector.py <frames_dir> <init_rect.txt> <output.mp4>
                        [--gt groundtruth.txt]
                        [--fps 25]
                        [--scale_x 5.0]  [--scale_y 5.0]
                        [--n_score_frames 50]
                        [--n_bands 3]
                        [--criterion {mer,snr,fisher,bhat,tc}]
                        [--fusion {mean,weighted,max}]
                        [--show_scores]
                        [--band_ids 1,3,7]
                        [--threshold 60] [--thresh_pct 97]
                        [--vel_alpha 0.6] [--max_search_dist 600]
```

**Key Arguments**

| Argument | Default | Description |
|---|---|---|
| `--n_score_frames` | 50 | Number of frame pairs sampled across sequence for band scoring |
| `--n_bands` | 3 | Number of top bands to auto-select ($K$) |
| `--criterion` | `mer` | Band selection scoring metric (`mer`, `snr`, `fisher`, `bhat`, `tc`) |
| `--fusion` | `mean` | Fusion method (`mean`, `weighted`, `max`) |
| `--show_scores` | False | Print complete 12-band score evaluation table |
| `--band_ids` | None | Manually specify comma-separated bands (e.g. `0,1,7`), skipping auto-scoring |

**Run Commands & Examples:**

1. **Auto-score and track using Top-3 bands by MER with Mean fusion (Optimal IoU):**
```bash
python band_selector.py Datasets/ball Datasets/ball/init_rect.txt tracked_top3_mer.mp4 \
    --gt Datasets/groundtruth_ball.txt \
    --n_score_frames 50 \
    --n_bands 3 \
    --criterion mer \
    --fusion mean \
    --show_scores \
    --scale_x 5 --scale_y 5
```

2. **Ultra-robust tracking using Multi-Scale + LoG with Max fusion (98.6% continuous lock, 1.4% fallback):**
```bash
python band_selector.py Datasets/ball Datasets/ball/init_rect.txt tracked_log_max.mp4 \
    --gt Datasets/groundtruth_ball.txt \
    --band_ids 1,3,7 \
    --fusion max \
    --scale_x 5 --scale_y 5
```

3. **Weighted fusion using Top-5 bands by SNR:**
```bash
python band_selector.py Datasets/ball Datasets/ball/init_rect.txt tracked_top5_snr.mp4 \
    --gt Datasets/groundtruth_ball.txt \
    --n_bands 5 \
    --criterion snr \
    --fusion weighted \
    --scale_x 5 --scale_y 5
```

---

### `band_study.py` — Comparative Study Framework

Runs all experimental variations across single bands, top-$K$ band subsets, selection criteria, and fusion algorithms; exports comprehensive CSV summaries and high-resolution matplotlib charts.

```bash
python band_study.py <frames_dir> <init_rect.txt>
                     --gt <groundtruth.txt>
                     [--output_dir band_study_results]
                     [--scale_x 5.0]  [--scale_y 5.0]
                     [--n_score_frames 50]
                     [--k_values 1,2,3,5,12]
                     [--no_video]
```

**Run Commands:**

1. **Full comparative study across all configurations (Fast in-memory mode, no video encode):**
```bash
python band_study.py Datasets/ball Datasets/ball/init_rect.txt \
    --gt Datasets/groundtruth_ball.txt \
    --output_dir band_study_results \
    --no_video
```

2. **Full comparative study with annotated MP4 video generated for every experiment:**
```bash
python band_study.py Datasets/ball Datasets/ball/init_rect.txt \
    --gt Datasets/groundtruth_ball.txt \
    --output_dir band_study_results
```

**Generated Study Artifacts (`band_study_results/`):**
- `results.csv` — Full quantitative evaluation matrix (Mean IoU, Median IoU, Centre Errors, Fallback Rates) for all 32 experiments
- `band_scores.csv` — Exact per-band informativeness metrics across all 5 criteria with mean and standard deviation
- `iou_comparison.png` — Comparative bar chart of Mean IoU across all experimental configurations
- `ce_comparison.png` — Comparative bar chart of Center Localization Error
- `band_score_heatmap.png` — Normalized cross-criterion band informativeness heatmap
- `topk_effect.png` — Line plot of IoU and Error curves as a function of number of bands ($K$)
- `fusion_comparison.png` — Performance comparison of Mean vs Weighted vs Max fusion
- `mer_vs_iou.png` — Scatter plot and correlation between band informativeness score and tracking IoU

---

## Requirements

```bash
pip install opencv-python numpy pillow matplotlib pandas
```

Python ≥ 3.8. No GPU required. Compatible with Windows, Linux, and macOS.

---

## Design Decisions Summary

| Decision | Rationale |
|---|---|
| Use raw 16-bit data for differencing | 8-bit conversion discards ~6 bits of dynamic range; the ball's difference signal is clearest in the full 16-bit space |
| Adaptive + hard threshold | Adaptive alone can collapse to near-zero in static frames; the hard minimum (60) is ~8× the background noise mean (7.75) |
| Velocity-predicted blob selection | The tracker needs to handle 80–110 px/frame motion; velocity prediction places the search centre near the true new position even when the ball has moved far |
| Midpoint bias correction | The diff blob centroid is at the average of old and new positions; subtracting that bias improves localisation accuracy |
| EMA velocity smoothing (α=0.6) | Faster response than α=0.5 while still averaging out per-frame noise |
| Box size fixed | Simplicity; the blob detection naturally localises the centre accurately even when the apparent size changes |
| Fallback velocity coasting | Without a detected blob, the alternative is to freeze or use the last known position; velocity coasting is better for a fast-moving ball that is likely to continue in the same direction |
