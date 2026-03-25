import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .features import MatchResult


# Data containers

@dataclass
class VerifiedMatch:
    image_id_1: int
    image_id_2: int
    src_pts: np.ndarray       # (M, 2) inlier points in image 1
    dst_pts: np.ndarray       # (M, 2) inlier points in image 2
    F: np.ndarray             # (3, 3) Fundamental Matrix
    num_inliers: int
    inlier_ratio: float       # inliers / total matches before RANSAC

    @property
    def pair(self) -> tuple[int, int]:
        return (self.image_id_1, self.image_id_2)


@dataclass
class TwoViewResult:
    image_id_1: int
    image_id_2: int
    R: np.ndarray             # (3, 3) rotation of camera 2 relative to camera 1
    t: np.ndarray             # (3, 1) translation of camera 2 relative to camera 1
    points_3d: np.ndarray     # (N, 3) triangulated 3D points
    src_pts: np.ndarray       # (N, 2) corresponding 2D points in image 1
    dst_pts: np.ndarray       # (N, 2) corresponding 2D points in image 2
    reprojection_error: float # mean reprojection error across both views
    P1: np.ndarray            # (3, 4) projection matrix for camera 1
    P2: np.ndarray            # (3, 4) projection matrix for camera 2


# Stage 3: Geometric Verification

class GeometricVerifier:
    

    def __init__(
        self,
        ransac_threshold: float = 1.0,
        confidence: float = 0.999,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.3,
    ):
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio

    def verify(self, match: MatchResult) -> Optional[VerifiedMatch]:
        
        if match.num_matches < 8:
            # Fundamental Matrix requires at least 8 point correspondences
            return None

        F, mask = cv2.findFundamentalMat(
            match.src_pts,
            match.dst_pts,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence,
        )

        if F is None or mask is None:
            return None

        mask = mask.ravel().astype(bool)
        num_inliers = int(mask.sum())
        inlier_ratio = num_inliers / match.num_matches

        if num_inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None

        return VerifiedMatch(
            image_id_1=match.image_id_1,
            image_id_2=match.image_id_2,
            src_pts=match.src_pts[mask],
            dst_pts=match.dst_pts[mask],
            F=F,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
        )

    def verify_all(
        self,
        match_results: dict[tuple[int, int], MatchResult],
    ) -> dict[tuple[int, int], VerifiedMatch]:
        
        verified = {}
        total = len(match_results)

        print(f"\nGeometric verification on {total} pairs...")
        for (i, j), match in match_results.items():
            result = self.verify(match)
            if result is not None:
                verified[(i, j)] = result
                print(
                    f"  Pair ({i},{j})  {match.num_matches:>5} -> "
                    f"{result.num_inliers:>5} inliers  "
                    f"({result.inlier_ratio:.1%})  [KEPT]"
                )
            else:
                print(f"  Pair ({i},{j})  {match.num_matches:>5} -> failed verification  [DROPPED]")

        print(f"\nKept {len(verified)}/{total} pairs after geometric verification.")
        return verified


# Stage 4: Two-View Reconstruction

class TwoViewReconstructor:
    

    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)

    def reconstruct(self, verified: VerifiedMatch) -> Optional[TwoViewResult]:
        
        src = verified.src_pts.astype(np.float64)
        dst = verified.dst_pts.astype(np.float64)

        # Step 1: Essential Matrix
        E, e_mask = cv2.findEssentialMat(
            src, dst,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            return None

        e_mask = e_mask.ravel().astype(bool)
        src_e = src[e_mask]
        dst_e = dst[e_mask]

        if len(src_e) < 10:
            return None

        # Step 2 & 3: Recover pose (handles cheirality check internally)
        _, R, t, pose_mask = cv2.recoverPose(E, src_e, dst_e, cameraMatrix=self.K)

        pose_mask = pose_mask.ravel().astype(bool)
        src_final = src_e[pose_mask]
        dst_final = dst_e[pose_mask]

        if len(src_final) < 8:
            return None

        # Step 4: Build projection matrices and triangulate
        # Camera 1 is the world origin: P1 = K [I | 0]
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        # Camera 2: P2 = K [R | t]
        P2 = self.K @ np.hstack([R, t])

        # cv2.triangulatePoints expects (2, N) arrays
        pts4d = cv2.triangulatePoints(P1, P2, src_final.T, dst_final.T)

        # Convert from homogeneous to 3D: divide by w
        pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)

        # Filter points behind either camera
        pts3d, src_final, dst_final = self._filter_cheirality(
            pts3d, src_final, dst_final, R, t
        )

        if len(pts3d) < 6:
            return None

        # Step 5: Reprojection error
        error = self._reprojection_error(pts3d, src_final, dst_final, P1, P2)

        return TwoViewResult(
            image_id_1=verified.image_id_1,
            image_id_2=verified.image_id_2,
            R=R,
            t=t,
            points_3d=pts3d,
            src_pts=src_final,
            dst_pts=dst_final,
            reprojection_error=error,
            P1=P1,
            P2=P2,
        )

    # Helpers

    def _filter_cheirality(
        self,
        pts3d: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Depth in camera 1 (world frame): just z coordinate
        depth1 = pts3d[:, 2]

        # Depth in camera 2: transform points to camera 2 frame
        pts_cam2 = (R @ pts3d.T + t).T
        depth2 = pts_cam2[:, 2]

        mask = (depth1 > 0) & (depth2 > 0)
        return pts3d[mask], src[mask], dst[mask]

    def _reprojection_error(
        self,
        pts3d: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
    ) -> float:
        def project(P: np.ndarray, pts: np.ndarray) -> np.ndarray:
            n = pts.shape[0]
            pts_h = np.hstack([pts, np.ones((n, 1))])     # (N, 4)
            proj = (P @ pts_h.T).T                         # (N, 3)
            proj = proj[:, :2] / proj[:, 2:3]             # (N, 2)
            return proj

        proj1 = project(P1, pts3d)
        proj2 = project(P2, pts3d)

        err1 = np.linalg.norm(proj1 - src, axis=1).mean()
        err2 = np.linalg.norm(proj2 - dst, axis=1).mean()
        return float((err1 + err2) / 2.0)

    def rank_initial_pairs(
        self,
        verified_matches: dict[tuple[int, int], VerifiedMatch],
    ) -> list[tuple[tuple[int, int], float, float]]:
        
        candidates = []
        for (i, j), vm in verified_matches.items():
            if vm.num_inliers < 20:
                continue
            _, h_mask = cv2.findHomography(
                vm.src_pts, vm.dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if h_mask is None:
                continue
            h_inlier_ratio = float(h_mask.sum()) / len(h_mask)
            score = vm.num_inliers * (1.0 - h_inlier_ratio)
            candidates.append(((i, j), score, h_inlier_ratio))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates