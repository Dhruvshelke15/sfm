# 3D Reconstruction using Structure from Motion

A complete incremental SfM pipeline built from scratch in Python. Takes a collection of 2D photographs and reconstructs a sparse 3D point cloud with estimated camera poses.

**Results on Middlebury DinoRing (48 images):**
- 48/48 cameras registered (100%)
- 3,028 sparse 3D points
- 0.17 px mean reprojection error after bundle adjustment

---

## Project Structure

```
sfm_project/
├── main.py                    # Full pipeline entry point (stages 1-7)
├── demo.py                    # 2D → 3D demo figure generator
├── requirements.txt
├── README.md
├── data/
│   └── dino_ring/             # Middlebury DinoRing dataset
│       ├── dinoR0001.png
│       ├── ...
│       └── K.txt              # Camera intrinsics (you create this)
├── output/                    # All visualizations saved here
└── sfm/
    ├── __init__.py
    ├── features.py            # SIFT/ORB extraction + FLANN matching
    ├── geometry.py            # RANSAC verification + two-view reconstruction
    ├── reconstruction.py      # Incremental PnP registration
    ├── bundle_adjustment.py   # Sparse bundle adjustment (scipy)
    └── utils.py               # Image loading, visualization, histograms
```

---

## Pipeline

The pipeline runs 7 sequential stages:

| Stage | Description | Key technique |
|-------|-------------|---------------|
| 1 | Image loading | OpenCV |
| 2 | Feature extraction | SIFT (128-dim) or ORB (binary) |
| 3 | Feature matching | FLANN + Lowe ratio test (0.75) |
| 4 | Geometric verification | RANSAC + Fundamental Matrix |
| 5 | Two-view reconstruction | Essential Matrix + DLT triangulation |
| 6 | Incremental reconstruction | PnP + RANSAC (EPnP) |
| 7 | Bundle adjustment | scipy least_squares, sparse Jacobian |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Use `opencv-contrib-python`, not `opencv-python` -- SIFT is in the contrib module.
> If you have `opencv-python` installed already: `pip uninstall opencv-python` first.

---

## Dataset

Download the Middlebury DinoRing dataset:

```bash
cd data/
curl -O https://vision.middlebury.edu/mview/data/data/dinoRing.zip
unzip dinoRing.zip
mv dinoRing dino_ring
```

Create `K.txt` from the provided `.par` file. The DinoRing intrinsics are:

```bash
python -c "
import numpy as np
K = np.array([
    [3310.4,    0.0,   316.73],
    [   0.0, 3325.5,   200.55],
    [   0.0,    0.0,     1.0 ],
], dtype=np.float64)
np.savetxt('data/dino_ring/K.txt', K)
print('Saved K.txt')
"
```

---

## Running the Pipeline

### Quick test (no intrinsics, 10 images)
```bash
python main.py --images data/dino_ring --max 10 --visualize
```

### Full SIFT run (recommended)
```bash
python main.py \
  --images data/dino_ring \
  --intrinsics data/dino_ring/K.txt \
  --visualize \
  --min_matches 10 \
  --ratio_thresh 0.80 \
  --pnp_reproj 8.0 \
  --tri_reproj 8.0 \
  --stall 15
```

### Full run with bundle adjustment
```bash
python main.py \
  --images data/dino_ring \
  --intrinsics data/dino_ring/K.txt \
  --visualize \
  --min_matches 10 \
  --ratio_thresh 0.80 \
  --pnp_reproj 8.0 \
  --tri_reproj 8.0 \
  --stall 15 \
  --bundle_adjust \
  --ba_max_points 3000
```

### ORB ablation
```bash
python main.py \
  --images data/dino_ring \
  --intrinsics data/dino_ring/K.txt \
  --method orb \
  --visualize \
  --min_matches 10 \
  --ratio_thresh 0.80 \
  --pnp_reproj 8.0 \
  --tri_reproj 8.0 \
  --stall 15
```

### Demo figure (2D → 3D)
```bash
python demo.py \
  --images data/dino_ring \
  --intrinsics data/dino_ring/K.txt \
  --bundle_adjust \
  --ba_max_points 3000
```

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | required | Path to image directory |
| `--intrinsics` | None | Path to 3x3 K.txt file |
| `--method` | `sift` | Feature detector: `sift` or `orb` |
| `--n_features` | `8000` | Max keypoints per image |
| `--ratio_thresh` | `0.75` | Lowe ratio test threshold |
| `--min_matches` | `20` | Min matches to keep a pair |
| `--ransac_thresh` | `1.0` | RANSAC inlier threshold (px) |
| `--pnp_reproj` | `8.0` | PnP RANSAC reprojection threshold (px) |
| `--tri_reproj` | `8.0` | Max triangulation reprojection error (px) |
| `--stall` | `15` | Max stall iterations before stopping |
| `--max` | None | Cap number of images (quick tests) |
| `--visualize` | False | Save all visualizations to `--output` |
| `--bundle_adjust` | False | Run bundle adjustment after incremental reconstruction |
| `--ba_max_points` | `3000` | Max points used in bundle adjustment |
| `--output` | `output` | Directory for saved visualizations |

---

## Output Files

After a full run with `--visualize`, the `output/` folder contains:

| File | Description |
|------|-------------|
| `keypoints_0_sift.png` | SIFT keypoints on image 0 |
| `keypoints_0_orb.png` | ORB keypoints on image 0 (ORB run) |
| `matches_i_j.png` | Feature matches for best image pair |
| `point_cloud_final.png` | Full 3D point cloud with camera trajectory |
| `reprojection_histogram.png` | Distribution of per-point reprojection errors |
| `results_summary.txt` | All pipeline stats: images, points, errors |

`demo.py` additionally produces:

| File | Description |
|------|-------------|
| `demo_2d_to_3d.png` | Side-by-side 2D inputs → 3D output figure |

---

## Results

### SIFT (main results)

| Metric | Value |
|--------|-------|
| Images registered | 48 / 48 (100%) |
| Triangulated 3D points | 3,028 |
| Seed reprojection error | 0.224 px |
| BA error (before) | 0.493 px |
| BA error (after) | **0.166 px** |

### SIFT vs ORB ablation

| Metric | SIFT | ORB |
|--------|------|-----|
| Avg keypoints / image | 220 | 471 |
| Best pair matches | 242 | 546 |
| Cameras registered | **48/48** | 26/48 |
| Mean reproj. error | **0.17 px** | 1.01 px |
| Points < 1 px error | **~85%** | 69.9% |

ORB detects 2x more keypoints and raw matches but SIFT achieves complete camera registration and 6x lower reprojection error. SIFT's 128-dimensional float descriptors are significantly more robust for wide-baseline matching, where ORB's binary descriptors fail to find reliable correspondences.

---

## Code Walkthrough

### `sfm/features.py`

**`FeatureExtractor`** wraps `cv2.SIFT_create()` or `cv2.ORB_create()`. Key config: `contrast_threshold=0.02` (relaxed from 0.04) for denser keypoints on low-contrast images. Returns `ImageFeatures` dataclass per image.

**`FeatureMatcher`** uses FLANN with KD-tree (SIFT) or LSH (ORB). Applies Lowe's ratio test and returns `MatchResult` with filtered point pairs. `match_all_pairs()` exhaustively matches all image combinations.

### `sfm/geometry.py`

**`GeometricVerifier`** runs `cv2.findFundamentalMat` with RANSAC per matched pair. Drops pairs below `min_inliers=15` or `inlier_ratio=0.3`. Returns `VerifiedMatch` with only epipolar-consistent correspondences.

**`TwoViewReconstructor`** computes the Essential Matrix `E = K^T F K`, decomposes via SVD into (R, t), builds projection matrices P1/P2, triangulates with DLT, filters by cheirality. `rank_initial_pairs()` scores candidates by `num_inliers * (1 - homography_ratio)` to avoid degenerate planar seeds.

### `sfm/reconstruction.py`

**`IncrementalReconstructor`** grows the reconstruction image by image:
1. Seeds from the two-view result
2. Each iteration: find unregistered images with 2D-3D correspondences via the match graph
3. Run `cv2.solvePnPRansac` (EPnP) to estimate pose
4. Register the best candidate, triangulate new points
5. Repeat until all images registered or stall limit reached
Also extracts RGB colors from images for colored point clouds.

### `sfm/bundle_adjustment.py`

**`BundleAdjuster`** minimizes total reprojection error jointly over all camera poses and 3D points using `scipy.optimize.least_squares` with Trust Region Reflective and a sparse Jacobian. Cameras are parameterized as (rvec, tvec) ∈ R^6. Sparse Jacobian structure is built explicitly for efficiency.

### `sfm/utils.py`

- `load_images()` -- directory loader with optional resize and grayscale
- `load_intrinsics()` -- reads 3x3 K matrix from .txt
- `filter_outliers()` -- removes points beyond 2.5 std from centroid
- `center_reconstruction()` -- translates cloud to origin for clean visualization
- `visualize_point_cloud()` -- PyVista interactive viewer with camera trajectory lines; falls back to matplotlib
- `plot_reprojection_histogram()` -- dark-themed error distribution with mean/median lines

---

## Troubleshooting

**`SIFT not available` error**
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

**`No pairs survived geometric verification`**
Relax thresholds:
```bash
python main.py --images data/dino_ring --ransac_thresh 3.0 --min_matches 10
```

**BA hangs / takes too long**
Lower `--ba_max_points`:
```bash
--bundle_adjust --ba_max_points 500
```

**PyVista window doesn't open on macOS**
```bash
pip install pyvistaqt
```

**Point cloud looks like a straight line**
This is a viewing angle issue. The reconstruction is 3D -- rotate the PyVista window to see the ring shape. The `center_reconstruction()` function centers the cloud at origin automatically.

---

## Dependencies

```
opencv-contrib-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
tqdm>=4.65.0
pyvista>=0.43.0
```

---

## References

1. Schonberger & Frahm. *Structure-from-Motion Revisited.* CVPR 2016.
2. Hartley & Zisserman. *Multiple View Geometry in Computer Vision.* Cambridge, 2003.
3. Lowe. *Distinctive Image Features from Scale-Invariant Keypoints.* IJCV 2004.
4. Rublee et al. *ORB: An Efficient Alternative to SIFT or SURF.* ICCV 2011.
5. Fischler & Bolles. *Random Sample Consensus.* CACM 1981.
6. Muja & Lowe. *Fast Approximate Nearest Neighbors.* VISAPP 2009.
7. Middlebury Multi-View Dataset: https://vision.middlebury.edu/mview/data/
