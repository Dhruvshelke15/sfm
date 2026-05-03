import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"}


def load_images(
    source,
    grayscale: bool = False,
    max_images: Optional[int] = None,
    resize_to: Optional[tuple] = None,
):
    source = Path(source)
    paths = []
    if source.is_dir():
        for ext in SUPPORTED_EXTENSIONS:
            paths.extend(source.glob(f"*{ext}"))
            paths.extend(source.glob(f"*{ext.upper()}"))
        paths = sorted(set(paths))
    elif source.is_file():
        paths = [source]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    if max_images is not None:
        paths = paths[:max_images]

    images, valid_paths = [], []
    for p in paths:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(p), flag)
        if img is None:
            print(f"  [WARNING] Could not load: {p}")
            continue
        if resize_to is not None:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        images.append(img)
        valid_paths.append(str(p))

    print(f"Loaded {len(images)} images from '{source}'")
    return images, valid_paths


def load_intrinsics(path) -> np.ndarray:
    K = np.loadtxt(str(path), dtype=np.float64)
    assert K.shape == (3, 3), f"Expected 3x3 matrix, got {K.shape}"
    return K


def filter_outliers(points_3d: np.ndarray, std_ratio: float = 2.0) -> np.ndarray:
    """Remove statistical outliers using mean + std_ratio * std per axis."""
    if len(points_3d) == 0:
        return points_3d
    mean = np.mean(points_3d, axis=0)
    std  = np.std(points_3d, axis=0)
    mask = np.all(np.abs(points_3d - mean) < std_ratio * std, axis=1)
    n_removed = len(points_3d) - mask.sum()
    if n_removed > 0:
        print(f"  Outlier filter: removed {n_removed} points ({n_removed/len(points_3d)*100:.1f}%)")
    return points_3d[mask]


def center_reconstruction(points_3d: np.ndarray, camera_centers: list) -> tuple:
    """Translate everything so the centroid of the point cloud is at the origin."""
    centroid = np.mean(points_3d, axis=0)
    pts_centered = points_3d - centroid
    cams_centered = [c - centroid for c in camera_centers]
    return pts_centered, cams_centered


def visualize_keypoints(image, keypoints, title="Keypoints", save_path=None, show=True):
    vis = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(vis_rgb)
    plt.title(f"{title}  ({len(keypoints)} keypoints)")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved keypoint visualization to {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_matches(img1, kps1, img2, kps2, matches, title="Matches", max_draw=100, save_path=None, show=True):
    vis = cv2.drawMatches(img1, kps1, img2, kps2, matches[:max_draw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 5))
    plt.imshow(vis_rgb)
    plt.title(f"{title}  (showing {min(len(matches), max_draw)}/{len(matches)} matches)")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved match visualization to {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_point_cloud(
    points_3d: np.ndarray,
    colors: Optional[np.ndarray] = None,
    camera_centers: Optional[list] = None,
    title: str = "Sparse Point Cloud",
    point_size: float = 3.0,
    save_path: Optional[str] = None,
    show: bool = True,
):
    if len(points_3d) == 0:
        print("[visualize_point_cloud] No points to display.")
        return

    pts = filter_outliers(points_3d, std_ratio=2.5)
    clrs = None
    if colors is not None and len(colors) == len(points_3d):
        # Apply same outlier mask
        mean = np.mean(points_3d, axis=0)
        std  = np.std(points_3d, axis=0)
        mask = np.all(np.abs(points_3d - mean) < 2.5 * std, axis=1)
        clrs = colors[mask]

    cams = camera_centers or []
    pts, cams = center_reconstruction(pts, cams)

    if PYVISTA_AVAILABLE:
        _visualize_pyvista(pts, clrs, cams, title, point_size, save_path, show)
    else:
        print("[INFO] PyVista not found, falling back to matplotlib.")
        _visualize_matplotlib_3d(pts, cams, title, save_path, show)


def _visualize_pyvista(pts, colors, camera_centers, title, point_size, save_path, show):
    cloud = pv.PolyData(pts.astype(np.float32))
    if colors is not None and len(colors) == len(pts):
        cloud["colors"] = colors.astype(np.uint8)

    plotter = pv.Plotter(title=title, off_screen=not show)
    plotter.set_background("black")

    if colors is not None and len(colors) == len(pts):
        plotter.add_points(cloud, scalars="colors", rgb=True, point_size=point_size, render_points_as_spheres=False)
    else:
        plotter.add_points(cloud, color="white", point_size=point_size, render_points_as_spheres=False)

    if camera_centers and len(camera_centers) > 0:
        centers = np.array(camera_centers, dtype=np.float32)
        plotter.add_points(pv.PolyData(centers), color="red", point_size=point_size * 5, render_points_as_spheres=True)

        # Draw lines connecting consecutive cameras to show trajectory
        if len(centers) > 1:
            lines = []
            for i in range(len(centers) - 1):
                lines.append([2, i, i + 1])
            lines = np.array(lines)
            traj = pv.PolyData(centers)
            traj.lines = np.hstack(lines)
            plotter.add_mesh(traj, color="red", line_width=1, opacity=0.4)

    # Top-down view to show the ring shape
    centroid = pts.mean(axis=0)
    extent = np.max(np.ptp(pts, axis=0)) * 1.5
    plotter.camera.position = (centroid[0], centroid[1], centroid[2] + extent)
    plotter.camera.focal_point = tuple(centroid)
    plotter.camera.up = (0, 1, 0)

    plotter.add_text(title, position="upper_left", font_size=10, color="white")
    plotter.show_axes()

    if save_path:
        plotter.show(auto_close=False)
        plotter.screenshot(save_path)
        plotter.close()
        print(f"Saved point cloud screenshot to {save_path}")
    else:
        plotter.show()


def _visualize_matplotlib_3d(pts, camera_centers, title, save_path, show):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c="white", alpha=0.6)
    if camera_centers and len(camera_centers) > 0:
        cams = np.array(camera_centers)
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=40, c="red", marker="^")
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], "r-", alpha=0.3)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_title(title, color="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"Saved point cloud figure to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_reprojection_histogram(errors: list, save_path: Optional[str] = None, show: bool = True):
    """Plot histogram of per-point reprojection errors."""
    errors = np.array(errors)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(np.mean(errors), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(errors):.3f} px")
    ax.axvline(np.median(errors), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {np.median(errors):.3f} px")
    ax.set_xlabel("Reprojection Error (px)")
    ax.set_ylabel("Count")
    ax.set_title("Reprojection Error Distribution")
    ax.legend()
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111111")
        print(f"Saved reprojection histogram to {save_path}")
    if show:
        plt.show()
    plt.close()
    return errors


def print_summary(all_features, match_results):
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"{'ID':<5} {'Path':<35} {'Keypoints':>10}")
    print("-" * 60)
    for feat in all_features:
        name = os.path.basename(feat.image_path)
        print(f"{feat.image_id:<5} {name:<35} {feat.num_keypoints:>10}")

    print("\n" + "=" * 60)
    print("MATCHING SUMMARY")
    print("=" * 60)
    print(f"{'Pair':<12} {'Matches':>10}")
    print("-" * 60)
    for (i, j), result in match_results.items():
        print(f"({i:>2}, {j:>2})      {result.num_matches:>10}")
    print("=" * 60)