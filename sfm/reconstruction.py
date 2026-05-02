import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .features import ImageFeatures
from .geometry import TwoViewResult


@dataclass
class ReconstructionResult:
    points_3d: np.ndarray
    point_colors: np.ndarray
    camera_poses: dict
    registered_images: list
    reprojection_error: float
    point_obs: list


class IncrementalReconstructor:

    def __init__(self, K: np.ndarray, pnp_min_inliers: int = 6, pnp_reprojection: float = 8.0, tri_max_reproj: float = 8.0, max_stall: int = 15):
        self.K = K.astype(np.float64)
        self.pnp_min_inliers = pnp_min_inliers
        self.pnp_reprojection = pnp_reprojection
        self.tri_max_reproj = tri_max_reproj
        self.max_stall = max_stall
        self.points_3d: list = []
        self.point_obs: list = []
        self.camera_poses: dict = {}
        self.registered: set = set()

    def run(self, init_result: TwoViewResult, all_features: list, verified_matches: dict, images: Optional[list] = None) -> ReconstructionResult:
        self._reset()

        id1, id2 = init_result.image_id_1, init_result.image_id_2

        self.camera_poses[id1] = {"R": np.eye(3), "t": np.zeros((3, 1)), "P": self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])}
        self.camera_poses[id2] = {"R": init_result.R, "t": init_result.t, "P": init_result.P2}
        self.registered.update([id1, id2])

        kp1_arr = all_features[id1].keypoints_as_array()
        kp2_arr = all_features[id2].keypoints_as_array()

        for pt3d, src, dst in zip(init_result.points_3d, init_result.src_pts, init_result.dst_pts):
            self.points_3d.append(pt3d)
            self.point_obs.append([
                (id1, self._nearest_kp(kp1_arr, src)),
                (id2, self._nearest_kp(kp2_arr, dst)),
            ])

        print(f"\nSeed: images ({id1},{id2}), {len(self.points_3d)} initial points")

        stall = 0
        while len(self.registered) < len(all_features) and stall < self.max_stall:
            best = self._find_best_next(all_features, verified_matches)
            if best is None:
                stall += 1
                continue

            new_id, R, t, n_inliers = best
            self.camera_poses[new_id] = {"R": R, "t": t, "P": self.K @ np.hstack([R, t])}
            self.registered.add(new_id)
            stall = 0

            n_new = self._triangulate_new(new_id, all_features, verified_matches)
            print(f"  Registered {new_id:>3}  PnP inliers={n_inliers:>4}  new pts={n_new:>5}  total={len(self.points_3d):>6}")

        print(f"\nDone: {len(self.registered)}/{len(all_features)} images, {len(self.points_3d)} points")

        pts3d = np.array(self.points_3d, dtype=np.float64)
        colors = self._extract_colors(pts3d, images) if images else np.zeros((len(pts3d), 3), dtype=np.uint8)

        return ReconstructionResult(
            points_3d=pts3d,
            point_colors=colors,
            camera_poses=self.camera_poses,
            registered_images=list(self.registered),
            reprojection_error=0.0,
            point_obs=self.point_obs,
        )

    def _reset(self):
        self.points_3d, self.point_obs, self.camera_poses, self.registered = [], [], {}, set()

    def _nearest_kp(self, kp_arr: np.ndarray, pt: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(kp_arr - pt, axis=1)))

    def _get_2d3d(self, candidate_id: int, all_features: list, verified_matches: dict):
        obs_lookup = {(obs_id, kp_idx): pt_idx for pt_idx, obs_list in enumerate(self.point_obs) for obs_id, kp_idx in obs_list}

        pts2d, pts3d = [], []
        for reg_id in self.registered:
            key = (min(reg_id, candidate_id), max(reg_id, candidate_id))
            if key not in verified_matches:
                continue
            vm = verified_matches[key]
            reg_pts, cand_pts = (vm.src_pts, vm.dst_pts) if reg_id < candidate_id else (vm.dst_pts, vm.src_pts)
            reg_kp_arr = all_features[reg_id].keypoints_as_array()

            for rp, cp in zip(reg_pts, cand_pts):
                lk = (reg_id, self._nearest_kp(reg_kp_arr, rp))
                if lk in obs_lookup:
                    pts2d.append(cp)
                    pts3d.append(self.points_3d[obs_lookup[lk]])

        if len(pts2d) < 4:
            return np.empty((0, 2)), np.empty((0, 3))
        return np.array(pts2d, dtype=np.float64), np.array(pts3d, dtype=np.float64)

    def _find_best_next(self, all_features: list, verified_matches: dict) -> Optional[tuple]:
        best, best_n = None, self.pnp_min_inliers - 1

        for cand_id in range(len(all_features)):
            if cand_id in self.registered:
                continue
            pts2d, pts3d = self._get_2d3d(cand_id, all_features, verified_matches)
            if len(pts2d) < 6:
                continue

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, self.K, None,
                iterationsCount=200, reprojectionError=self.pnp_reprojection,
                confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) <= best_n:
                continue

            R, _ = cv2.Rodrigues(rvec)
            best_n = len(inliers)
            best = (cand_id, R, tvec, best_n)

        return best

    def _triangulate_new(self, new_id: int, all_features: list, verified_matches: dict) -> int:
        P_new = self.camera_poses[new_id]["P"]
        kp_new = all_features[new_id].keypoints_as_array()
        observed = {pair for obs_list in self.point_obs for pair in obs_list}
        n_new = 0

        for reg_id in list(self.registered - {new_id}):
            key = (min(reg_id, new_id), max(reg_id, new_id))
            if key not in verified_matches:
                continue
            vm = verified_matches[key]
            P_reg = self.camera_poses[reg_id]["P"]
            kp_reg = all_features[reg_id].keypoints_as_array()
            pts_reg, pts_new = (vm.src_pts, vm.dst_pts) if reg_id < new_id else (vm.dst_pts, vm.src_pts)

            for pr, pn in zip(pts_reg, pts_new):
                ri = self._nearest_kp(kp_reg, pr)
                ni = self._nearest_kp(kp_new, pn)
                if (reg_id, ri) in observed or (new_id, ni) in observed:
                    continue

                pts4d = cv2.triangulatePoints(P_reg, P_new, pr.reshape(2, 1), pn.reshape(2, 1))
                pt3d = (pts4d[:3] / pts4d[3]).ravel()

                R_reg = self.camera_poses[reg_id]["R"]
                t_reg = self.camera_poses[reg_id]["t"]
                R_new = self.camera_poses[new_id]["R"]
                t_new = self.camera_poses[new_id]["t"]

                if (R_reg @ pt3d + t_reg.ravel())[2] <= 0 or (R_new @ pt3d + t_new.ravel())[2] <= 0:
                    continue

                ph = np.append(pt3d, 1.0)
                q1, q2 = P_reg @ ph, P_new @ ph
                if q1[2] <= 0 or q2[2] <= 0:
                    continue

                err = (np.linalg.norm(q1[:2] / q1[2] - pr) + np.linalg.norm(q2[:2] / q2[2] - pn)) / 2.0
                if err > self.tri_max_reproj:
                    continue

                self.points_3d.append(pt3d)
                self.point_obs.append([(reg_id, ri), (new_id, ni)])
                observed.add((reg_id, ri))
                observed.add((new_id, ni))
                n_new += 1

        return n_new

    def _extract_colors(self, pts3d: np.ndarray, images: list) -> np.ndarray:
        colors = np.zeros((len(pts3d), 3), dtype=np.uint8)
        for pt_idx, (pt3d, obs_list) in enumerate(zip(pts3d, self.point_obs)):
            for img_id, _ in obs_list:
                if img_id not in self.camera_poses or img_id >= len(images):
                    continue
                P = self.camera_poses[img_id]["P"]
                q = P @ np.append(pt3d, 1.0)
                if q[2] <= 0:
                    continue
                x, y = int(q[0] / q[2]), int(q[1] / q[2])
                h, w = images[img_id].shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    colors[pt_idx] = images[img_id][y, x][::-1]
                    break
        return colors