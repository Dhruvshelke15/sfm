import argparse
import os
import numpy as np

from sfm.features import FeatureExtractor, FeatureMatcher
from sfm.geometry import GeometricVerifier, TwoViewReconstructor
from sfm.utils import (
    load_images,
    load_intrinsics,
    visualize_keypoints,
    visualize_matches,
    visualize_point_cloud,
    print_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SfM Pipeline")
    parser.add_argument("--images",     type=str, required=True,  help="Path to image directory.")
    parser.add_argument("--intrinsics", type=str, default=None,   help="Path to K.txt intrinsics file (3x3).")
    parser.add_argument("--method",     type=str, default="sift", choices=["sift", "orb"])
    parser.add_argument("--n_features", type=int, default=8000)
    parser.add_argument("--ratio_thresh", type=float, default=0.75)
    parser.add_argument("--min_matches", type=int, default=20)
    parser.add_argument("--ransac_thresh", type=float, default=1.0, help="RANSAC reprojection threshold (px).")
    parser.add_argument("--max",        type=int, default=None, dest="max_images")
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--output",     type=str, default="output")
    return parser.parse_args()


def build_default_K(images: list) -> np.ndarray:
    
    h, w = images[0].shape[:2]
    f = max(h, w)
    K = np.array([
        [f,   0,  w / 2],
        [0,   f,  h / 2],
        [0,   0,  1    ],
    ], dtype=np.float64)
    print(f"\n[INFO] No intrinsics file provided. Using estimated K:")
    print(K)
    return K


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1. Load images
    print("\n=== Stage 1: Loading Images ===")
    images, paths = load_images(args.images, max_images=args.max_images)
    if len(images) < 2:
        print("Need at least 2 images. Exiting.")
        return

    # 2. Feature extraction
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

    # 3. Feature matching
    print(f"\n=== Stage 3: Feature Matching ===")
    matcher = FeatureMatcher(
        method=args.method,
        ratio_thresh=args.ratio_thresh,
        min_matches=args.min_matches,
    )
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

    # 4. Geometric verification
    print(f"\n=== Stage 4: Geometric Verification (RANSAC) ===")
    verifier = GeometricVerifier(ransac_threshold=args.ransac_thresh)
    verified_matches = verifier.verify_all(match_results)

    if not verified_matches:
        print("No pairs survived geometric verification. Exiting.")
        return

    # 5. Two-view reconstruction (initial pair)
    print(f"\n=== Stage 5: Two-View Reconstruction ===")
    K = load_intrinsics(args.intrinsics) if args.intrinsics else build_default_K(images)

    reconstructor = TwoViewReconstructor(K)
    candidates = reconstructor.rank_initial_pairs(verified_matches)

    if not candidates:
        print("No valid initial pair candidates found. Exiting.")
        return

    print(f"\nTop-10 initial pair candidates:")
    for (ci, cj), score, h_ratio in candidates[:10]:
        print(
            f"  Pair ({ci:>2},{cj:>2})  "
            f"inliers={verified_matches[(ci,cj)].num_inliers:>4}  "
            f"homography_ratio={h_ratio:.2f}  score={score:.1f}"
        )

    # Try candidates in ranked order until one succeeds
    result = None
    best_pair = None
    for (ci, cj), score, h_ratio in candidates[:10]:
        print(f"\n  Trying pair ({ci},{cj})...")
        r = reconstructor.reconstruct(verified_matches[(ci, cj)])
        if r is not None:
            result = r
            best_pair = (ci, cj)
            print(f"  Success!")
            break
        print(f"  Failed, trying next candidate...")

    if result is None:
        print("\nAll candidate pairs failed reconstruction. Exiting.")
        return

    i, j = best_pair
    print(f"\nTwo-view reconstruction complete:")
    print(f"  Pair:               ({i}, {j})")
    print(f"  Triangulated pts:   {len(result.points_3d)}")
    print(f"  Mean reproj error:  {result.reprojection_error:.4f} px")
    print(f"  Rotation (R):\n{result.R}")
    print(f"  Translation (t):    {result.t.ravel()}")

    if args.visualize:
        visualize_point_cloud(
            result.points_3d,
            camera_centers=[
                np.zeros(3),
                (-result.R.T @ result.t).ravel(),
            ],
            title=f"Initial Point Cloud -- Pair ({i},{j})",
            save_path=os.path.join(args.output, f"point_cloud_{i}_{j}.png"),
            show=True,
        )


if __name__ == "__main__":
    main()