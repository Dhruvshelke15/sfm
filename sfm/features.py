import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional

# Data containers

@dataclass
class ImageFeatures:
    image_id: int
    image_path: str
    keypoints: list          # list of cv2.KeyPoint
    descriptors: np.ndarray  # shape (N, 128) for SIFT, (N, 32) for ORB

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def keypoints_as_array(self) -> np.ndarray:
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float32)


@dataclass
class MatchResult:
    image_id_1: int
    image_id_2: int
    matches: list            # list of cv2.DMatch (after filtering)
    src_pts: np.ndarray      # (M, 2) matched points in image 1
    dst_pts: np.ndarray      # (M, 2) matched points in image 2

    @property
    def num_matches(self) -> int:
        return len(self.matches)


# Feature Extractor

class FeatureExtractor:
    

    def __init__(
        self,
        method: Literal["sift", "orb"] = "sift",
        n_features: int = 8000,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.02,
        edge_threshold: int = 10,
    ):
        self.method = method.lower()
        self.n_features = n_features

        if self.method == "sift":
            self.detector = cv2.SIFT_create(
                nfeatures=n_features,
                nOctaveLayers=n_octave_layers,
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold,
                sigma=1.6,
            )
        elif self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=n_features)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'sift' or 'orb'.")

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[list, np.ndarray]:
        
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)

        if descriptors is None:
            descriptors = np.empty((0, 128 if self.method == "sift" else 32), dtype=np.float32)

        return keypoints, descriptors

    def extract_from_image_list(
        self,
        images: list[np.ndarray],
        image_paths: Optional[list[str]] = None,
    ) -> list[ImageFeatures]:
        
        if image_paths is None:
            image_paths = [f"image_{i}" for i in range(len(images))]

        results = []
        for i, (img, path) in enumerate(zip(images, image_paths)):
            kps, descs = self.detect_and_compute(img)
            feat = ImageFeatures(
                image_id=i,
                image_path=path,
                keypoints=kps,
                descriptors=descs,
            )
            results.append(feat)
            print(f"  [{i+1}/{len(images)}] {path}  ->  {feat.num_keypoints} keypoints")

        return results


# Feature Matcher

class FeatureMatcher:
    

    # FLANN parameters for SIFT (float descriptors)
    _FLANN_INDEX_KDTREE = 1
    _FLANN_SIFT_INDEX_PARAMS  = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
    _FLANN_SEARCH_PARAMS      = {"checks": 50}

    # FLANN parameters for ORB (binary descriptors)
    _FLANN_INDEX_LSH = 6
    _FLANN_ORB_INDEX_PARAMS = {
        "algorithm": _FLANN_INDEX_LSH,
        "table_number": 6,
        "key_size": 12,
        "multi_probe_level": 1,
    }

    def __init__(
        self,
        method: Literal["sift", "orb"] = "sift",
        ratio_thresh: float = 0.75,
        min_matches: int = 20,
    ):
        self.method = method.lower()
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches

        if self.method == "sift":
            self.matcher = cv2.FlannBasedMatcher(
                self._FLANN_SIFT_INDEX_PARAMS,
                self._FLANN_SEARCH_PARAMS,
            )
        elif self.method == "orb":
            self.matcher = cv2.FlannBasedMatcher(
                self._FLANN_ORB_INDEX_PARAMS,
                self._FLANN_SEARCH_PARAMS,
            )
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'sift' or 'orb'.")

    def match(
        self,
        features1: ImageFeatures,
        features2: ImageFeatures,
    ) -> Optional[MatchResult]:
        
        if features1.num_keypoints < 2 or features2.num_keypoints < 2:
            return None

        # knnMatch returns 2 nearest neighbors per descriptor
        raw_matches = self.matcher.knnMatch(
            features1.descriptors,
            features2.descriptors,
            k=2,
        )

        # Lowe's ratio test: keep matches where best << second-best
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            return None

        # Extract matched point coordinates
        src_pts = np.array(
            [features1.keypoints[m.queryIdx].pt for m in good_matches],
            dtype=np.float32,
        )
        dst_pts = np.array(
            [features2.keypoints[m.trainIdx].pt for m in good_matches],
            dtype=np.float32,
        )

        return MatchResult(
            image_id_1=features1.image_id,
            image_id_2=features2.image_id,
            matches=good_matches,
            src_pts=src_pts,
            dst_pts=dst_pts,
        )

    def match_all_pairs(
        self,
        all_features: list[ImageFeatures],
    ) -> dict[tuple[int, int], MatchResult]:
        
        results = {}
        n = len(all_features)
        total_pairs = n * (n - 1) // 2

        print(f"\nMatching {total_pairs} image pairs...")
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                count += 1
                result = self.match(all_features[i], all_features[j])
                if result is not None:
                    results[(i, j)] = result
                    print(
                        f"  Pair ({i},{j})  ->  {result.num_matches} matches  [KEPT]"
                    )
                else:
                    print(f"  Pair ({i},{j})  ->  insufficient matches  [SKIPPED]")

        print(f"\nKept {len(results)}/{total_pairs} pairs with >= {self.min_matches} matches.")
        return results