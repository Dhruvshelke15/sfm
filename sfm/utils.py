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


# Image Loading

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".ppm"}


def load_images(
    source: str | Path,
    grayscale: bool = False,
    max_images: Optional[int] = None,
    resize_to: Optional[tuple[int, int]] = None,
) -> tuple[list[np.ndarray], list[str]]:
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


def load_intrinsics(path: str | Path) -> np.ndarray:
    K = np.loadtxt(str(path), dtype=np.float64)
    assert K.shape == (3, 3), f"Expected 3x3 matrix, got {K.shape}"
    return K


# 2D Visualization (matplotlib)


def visualize_keypoints(
    image: np.ndarray,
    keypoints: list,
    title: str = "Keypoints",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
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


def visualize_matches(
    img1: np.ndarray,
    kps1: list,
    img2: np.ndarray,
    kps2: list,
    matches: list,
    title: str = "Matches",
    max_draw: int = 100,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    vis = cv2.drawMatches(
        img1, kps1,
        img2, kps2,
        matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
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


# 3D Visualization (PyVista with matplotlib fallback)

def visualize_point_cloud(
    points_3d: np.ndarray,
    colors: Optional[np.ndarray] = None,
    camera_centers: Optional[list[np.ndarray]] = None,
    title: str = "Sparse Point Cloud",
    point_size: float = 2.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    if len(points_3d) == 0:
        print("[visualize_point_cloud] No points to display.")
        return

    # Strip extreme outliers (keep inner 98%) for cleaner rendering
    lo, hi = np.percentile(points_3d, [1, 99], axis=0)
    mask = np.all((points_3d >= lo) & (points_3d <= hi), axis=1)
    pts = points_3d[mask]
    clrs = colors[mask] if colors is not None else None

    if PYVISTA_AVAILABLE:
        _visualize_pyvista(pts, clrs, camera_centers, title, point_size, save_path, show)
    else:
        print("[INFO] PyVista not found -- falling back to matplotlib 3D viewer.")
        _visualize_matplotlib_3d(pts, camera_centers, title, save_path, show)


def _visualize_pyvista(
    pts: np.ndarray,
    colors: Optional[np.ndarray],
    camera_centers: Optional[list[np.ndarray]],
    title: str,
    point_size: float,
    save_path: Optional[str],
    show: bool,
) -> None:
    cloud = pv.PolyData(pts.astype(np.float32))

    if colors is not None:
        cloud["colors"] = colors.astype(np.uint8)

    plotter = pv.Plotter(title=title, off_screen=not show)
    plotter.set_background("black")

    if colors is not None:
        plotter.add_points(
            cloud,
            scalars="colors",
            rgb=True,
            point_size=point_size,
            render_points_as_spheres=False,
        )
    else:
        plotter.add_points(
            cloud,
            color="white",
            point_size=point_size,
            render_points_as_spheres=False,
        )

    # Overlay camera centers as red spheres with frustum axes
    if camera_centers:
        centers = np.array(camera_centers, dtype=np.float32)
        cam_cloud = pv.PolyData(centers)
        plotter.add_points(
            cam_cloud,
            color="red",
            point_size=point_size * 4,
            render_points_as_spheres=True,
            label="Camera Centers",
        )

    plotter.add_text(title, position="upper_left", font_size=10, color="white")
    plotter.show_axes()

    if save_path and not show:
        # Off-screen: render silently and save screenshot
        plotter.show(auto_close=False)
        plotter.screenshot(save_path)
        plotter.close()
        print(f"Saved point cloud screenshot to {save_path}")
    elif save_path and show:
        # Interactive: open window, then save screenshot after user closes it
        plotter.show(auto_close=False)
        plotter.screenshot(save_path)
        plotter.close()
        print(f"Saved point cloud screenshot to {save_path}")
    else:
        plotter.show()


def _visualize_matplotlib_3d(
    pts: np.ndarray,
    camera_centers: Optional[list[np.ndarray]],
    title: str,
    save_path: Optional[str],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c="white", alpha=0.6)

    if camera_centers:
        cams = np.array(camera_centers)
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=40, c="red", marker="^", label="Cameras")
        ax.legend()

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


# Console Summary

def print_summary(all_features: list, match_results: dict) -> None:
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