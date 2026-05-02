import argparse
import os
import numpy as np

from sfm.features import FeatureExtractor, FeatureMatcher
from sfm.geometry import GeometricVerifier, TwoViewReconstructor
from sfm.reconstruction import IncrementalReconstructor
from sfm.bundle_adjustment import BundleAdjuster
from sfm.utils import load_images, load_intrinsics, visualize_keypoints, visualize_matches, visualize_point_cloud, print_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images",        type=str, required=True)
    p.add_argument("--intrinsics",    type=str, default=None)
    p.add_argument("--method",        type=str, default="sift", choices=["sift", "orb"])
    p.add_argument("--n_features",    type=int, default=8000)
    p.add_argument("--ratio_thresh",  type=float, default=0.75)
    p.add_argument("--min_matches",   type=int, default=20)
    p.add_argument("--ransac_thresh", type=float, default=1.0)
    p.add_argument("--max",           type=int, default=None, dest="max_images")
    p.add_argument("--visualize",     action="store_true")
    p.add_argument("--bundle_adjust", action="store_true")
    p.add_argument("--ba_max_points", type=int, default=1000)
    p.add_argument("--pnp_reproj",   type=float, default=8.0, help="PnP RANSAC reprojection threshold (default 8.0)")
    p.add_argument("--tri_reproj",   type=float, default=8.0, help="Triangulation max reprojection error (default 8.0)")
    p.add_argument("--stall",        type=int, default=15, help="Max stall iterations before stopping (default 15)")
    p.add_argument("--output",       type=str, default="output")
    return p.parse_args()


def build_default_K(images):
    h, w = images[0].shape[:2]
    f = max(h, w)
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    print(f"\n[INFO] No intrinsics provided. Using estimated K:\n{K}")
    return K


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n=== Stage 1: Loading Images ===")
    images, paths = load_images(args.images, max_images=args.max_images)
    if len(images) < 2:
        print("Need at least 2 images.")
        return

    print(f"\n=== Stage 2: Feature Extraction ({args.method.upper()}) ===")
    extractor = FeatureExtractor(method=args.method, n_features=args.n_features)
    all_features = extractor.extract_from_image_list(images, paths)

    if args.visualize:
        visualize_keypoints(
            images[0], all_features[0].keypoints,
            title=f"Image 0 -- {args.method.upper()} Keypoints",
            save_path=os.path.join(args.output, f"keypoints_0_{args.method}.png"),
            show=False,
        )

    print(f"\n=== Stage 3: Feature Matching ===")
    matcher = FeatureMatcher(method=args.method, ratio_thresh=args.ratio_thresh, min_matches=args.min_matches)
    match_results = matcher.match_all_pairs(all_features)

    if args.visualize and match_results:
        best_pair = max(match_results, key=lambda k: match_results[k].num_matches)
        i, j = best_pair
        visualize_matches(
            images[i], all_features[i].keypoints,
            images[j], all_features[j].keypoints,
            match_results[best_pair].matches,
            title=f"Best Match Pair ({i},{j}) -- {args.method.upper()}",
            save_path=os.path.join(args.output, f"matches_{i}_{j}.png"),
            show=False,
        )

    print_summary(all_features, match_results)

    print(f"\n=== Stage 4: Geometric Verification ===")
    verifier = GeometricVerifier(ransac_threshold=args.ransac_thresh)
    verified_matches = verifier.verify_all(match_results)
    if not verified_matches:
        print("No pairs survived verification.")
        return

    print(f"\n=== Stage 5: Two-View Reconstruction ===")
    K = load_intrinsics(args.intrinsics) if args.intrinsics else build_default_K(images)
    two_view = TwoViewReconstructor(K)
    candidates = two_view.rank_initial_pairs(verified_matches)

    if not candidates:
        print("No valid initial pairs.")
        return

    print("\nTop-10 candidates:")
    for (ci, cj), score, h_ratio in candidates[:10]:
        print(f"  ({ci:>2},{cj:>2})  inliers={verified_matches[(ci,cj)].num_inliers:>4}  h_ratio={h_ratio:.2f}  score={score:.1f}")

    init_result, seed_pair = None, None
    for (ci, cj), _, _ in candidates[:10]:
        r = two_view.reconstruct(verified_matches[(ci, cj)])
        if r is not None:
            init_result, seed_pair = r, (ci, cj)
            print(f"\nSeed pair ({ci},{cj}): {len(r.points_3d)} pts, reproj={r.reprojection_error:.4f} px")
            break

    if init_result is None:
        print("All seed pairs failed.")
        return

    print(f"\n=== Stage 6: Incremental Reconstruction ===")
    inc = IncrementalReconstructor(K, pnp_reprojection=args.pnp_reproj, tri_max_reproj=args.tri_reproj, max_stall=args.stall)
    inc_result = inc.run(init_result, all_features, verified_matches, images=images)

    print(f"\nRegistered: {len(inc_result.registered_images)}/{len(images)}")
    print(f"Total points: {len(inc_result.points_3d)}")

    if args.bundle_adjust:
        print(f"\n=== Stage 7: Bundle Adjustment ===")
        ba = BundleAdjuster(K)
        ba_result = ba.run(inc_result, all_features, max_points=args.ba_max_points)
        final_pts = ba_result.points_3d
        final_poses = ba_result.camera_poses
    else:
        final_pts = inc_result.points_3d
        final_poses = inc_result.camera_poses

    if args.visualize:
        cam_centers = [(-p["R"].T @ p["t"]).ravel() for p in final_poses.values()]
        visualize_point_cloud(
            final_pts,
            colors=inc_result.point_colors if len(inc_result.point_colors) > 0 else None,
            camera_centers=cam_centers,
            title=f"Full Reconstruction -- {len(inc_result.registered_images)} cameras, {len(final_pts)} points",
            save_path=os.path.join(args.output, "point_cloud_final.png"),
            show=True,
        )

    summary_path = os.path.join(args.output, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Dataset:           {args.images}\n")
        f.write(f"Method:            {args.method.upper()}\n")
        f.write(f"Total images:      {len(images)}\n")
        f.write(f"Registered images: {len(inc_result.registered_images)}\n")
        f.write(f"Total 3D points:   {len(final_pts)}\n")
        f.write(f"Seed pair:         {seed_pair}\n")
        f.write(f"Seed reproj error: {init_result.reprojection_error:.4f} px\n")
        if args.bundle_adjust:
            f.write(f"BA error before:   {ba_result.reprojection_error_before:.4f} px\n")
            f.write(f"BA error after:    {ba_result.reprojection_error_after:.4f} px\n")

    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()