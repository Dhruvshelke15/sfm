import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from sfm.features import FeatureExtractor, FeatureMatcher
from sfm.geometry import GeometricVerifier, TwoViewReconstructor
from sfm.reconstruction import IncrementalReconstructor
from sfm.bundle_adjustment import BundleAdjuster
from sfm.utils import load_images, load_intrinsics, filter_outliers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images",        type=str, required=True)
    p.add_argument("--intrinsics",    type=str, default=None)
    p.add_argument("--output",        type=str, default="output")
    p.add_argument("--min_matches",   type=int, default=10)
    p.add_argument("--ratio_thresh",  type=float, default=0.80)
    p.add_argument("--pnp_reproj",    type=float, default=8.0)
    p.add_argument("--tri_reproj",    type=float, default=8.0)
    p.add_argument("--stall",         type=int, default=15)
    p.add_argument("--bundle_adjust", action="store_true")
    p.add_argument("--ba_max_points", type=int, default=3000)
    return p.parse_args()


def build_default_K(images):
    h, w = images[0].shape[:2]
    f = max(h, w)
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)


def run_pipeline(args, images, paths):
    K = load_intrinsics(args.intrinsics) if args.intrinsics else build_default_K(images)

    extractor = FeatureExtractor(method="sift", n_features=8000)
    all_features = extractor.extract_from_image_list(images, paths)

    matcher = FeatureMatcher(method="sift", ratio_thresh=args.ratio_thresh, min_matches=args.min_matches)
    match_results = matcher.match_all_pairs(all_features)

    verifier = GeometricVerifier()
    verified = verifier.verify_all(match_results)

    two_view = TwoViewReconstructor(K)
    candidates = two_view.rank_initial_pairs(verified)

    init_result = None
    for (ci, cj), _, _ in candidates[:10]:
        r = two_view.reconstruct(verified[(ci, cj)])
        if r is not None:
            init_result = r
            break

    if init_result is None:
        raise RuntimeError("Two-view reconstruction failed.")

    inc = IncrementalReconstructor(K, pnp_reprojection=args.pnp_reproj, tri_max_reproj=args.tri_reproj, max_stall=args.stall)
    inc_result = inc.run(init_result, all_features, verified, images=images)

    if args.bundle_adjust:
        ba = BundleAdjuster(K)
        ba_result = ba.run(inc_result, all_features, max_points=args.ba_max_points)
        final_pts = ba_result.points_3d
        final_poses = ba_result.camera_poses
    else:
        final_pts = inc_result.points_3d
        final_poses = inc_result.camera_poses

    return final_pts, final_poses, inc_result.point_colors, inc_result


def make_demo_figure(images, final_pts, final_poses, colors, output_path):
    """Create a publication-quality 2D → 3D demo figure."""
    pts = filter_outliers(final_pts, std_ratio=2.5)
    centroid = pts.mean(axis=0)
    pts = pts - centroid

    cam_centers = np.array([(-p["R"].T @ p["t"]).ravel() - centroid for p in final_poses.values()])

    # Sample 6 evenly spaced input images
    n = len(images)
    sample_indices = np.linspace(0, n - 1, 6, dtype=int)

    fig = plt.figure(figsize=(18, 10), facecolor="#0a0a0a")
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.04, hspace=0.08,
                           left=0.02, right=0.98, top=0.92, bottom=0.04)

    # Input images on the left (2x3 grid)
    for idx, img_i in enumerate(sample_indices):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        img_rgb = cv2.cvtColor(images[img_i], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"Image {img_i+1}", color="white", fontsize=8, pad=2)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Arrow in the middle
    arrow_ax = fig.add_axes([0.495, 0.42, 0.015, 0.16], facecolor="none")
    arrow_ax.annotate("", xy=(0.5, 0.95), xytext=(0.5, 0.05),
                      arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.3",
                                     color="white", lw=2.5))
    arrow_ax.text(0.5, 0.5, "SfM", ha="center", va="center",
                  color="white", fontsize=11, fontweight="bold",
                  transform=arrow_ax.transAxes)
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis("off")

    # 3D point cloud on the right (spans both rows)
    ax3d = fig.add_subplot(gs[:, 3], projection="3d")
    ax3d.set_facecolor("#0a0a0a")

    # Plot point cloud
    if colors is not None and len(colors) == len(final_pts):
        # Apply same outlier mask
        mean = np.mean(final_pts, axis=0)
        std  = np.std(final_pts, axis=0)
        mask = np.all(np.abs(final_pts - mean) < 2.5 * std, axis=1)
        c = colors[mask].astype(np.float32) / 255.0
        pts_c = (final_pts[mask] - centroid)
        ax3d.scatter(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], c=c, s=1.0, alpha=0.8)
    else:
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="white", s=0.8, alpha=0.7)

    # Camera centers
    ax3d.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
                 c="red", s=20, zorder=5, label="Cameras")
    ax3d.plot(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
              "r-", alpha=0.3, linewidth=0.8)

    ax3d.set_title(f"3D Reconstruction\n{len(pts)} points  |  {len(final_poses)} cameras",
                   color="white", fontsize=10, pad=6)
    ax3d.tick_params(colors="#444444", labelsize=6)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#222222")
    ax3d.yaxis.pane.set_edgecolor("#222222")
    ax3d.zaxis.pane.set_edgecolor("#222222")
    ax3d.grid(False)
    ax3d.set_xlabel("X", color="#555555", fontsize=7)
    ax3d.set_ylabel("Y", color="#555555", fontsize=7)
    ax3d.set_zlabel("Z", color="#555555", fontsize=7)

    # Title
    fig.text(0.5, 0.96, "3D Reconstruction from 2D Images using Structure from Motion",
             ha="center", va="top", color="white", fontsize=14, fontweight="bold")

    fig.text(0.25, 0.96, "Input: 2D Images", ha="center", va="top", color="#aaaaaa", fontsize=10)
    fig.text(0.82, 0.96, "Output: 3D Point Cloud", ha="center", va="top", color="#aaaaaa", fontsize=10)

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
    print(f"Saved demo figure to {output_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("Loading images...")
    images, paths = load_images(args.images)

    print("Running SfM pipeline...")
    final_pts, final_poses, colors, inc_result = run_pipeline(args, images, paths)

    print(f"\nReconstruction: {len(inc_result.registered_images)}/{len(images)} cameras, {len(final_pts)} points")

    print("Generating demo figure...")
    make_demo_figure(
        images, final_pts, final_poses, colors,
        output_path=os.path.join(args.output, "demo_2d_to_3d.png"),
    )


if __name__ == "__main__":
    main()