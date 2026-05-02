import cv2
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .reconstruction import ReconstructionResult


@dataclass
class BundleAdjustmentResult:
    points_3d: np.ndarray
    camera_poses: dict
    reprojection_error_before: float
    reprojection_error_after: float
    n_observations: int


class BundleAdjuster:

    def __init__(self, K: np.ndarray, max_iter: int = 100, ftol: float = 1e-4):
        self.K = K.astype(np.float64)
        self.max_iter = max_iter
        self.ftol = ftol

    def run(self, result: ReconstructionResult, all_features: list, max_points: int = 2000) -> BundleAdjustmentResult:
        camera_ids = sorted(result.camera_poses.keys())
        cam_to_idx = {cid: i for i, cid in enumerate(camera_ids)}

        pts3d = result.points_3d.copy()
        sub_idx = None
        if len(pts3d) > max_points:
            sub_idx = np.random.choice(len(pts3d), max_points, replace=False)
            pts3d = pts3d[sub_idx]
            point_obs = [result.point_obs[i] for i in sub_idx]
        else:
            point_obs = result.point_obs

        observations = []
        for pt_idx, obs_list in enumerate(point_obs):
            for img_id, kp_idx in obs_list:
                if img_id not in cam_to_idx or img_id >= len(all_features):
                    continue
                kps = all_features[img_id].keypoints
                if kp_idx >= len(kps):
                    continue
                x, y = kps[kp_idx].pt
                observations.append((cam_to_idx[img_id], pt_idx, x, y))

        if len(observations) < 10:
            print("[BA] Not enough observations, skipping.")
            return BundleAdjustmentResult(
                points_3d=result.points_3d,
                camera_poses=result.camera_poses,
                reprojection_error_before=result.reprojection_error,
                reprojection_error_after=result.reprojection_error,
                n_observations=len(observations),
            )

        obs_arr = np.array(observations)
        cam_indices = obs_arr[:, 0].astype(int)
        pt_indices  = obs_arr[:, 1].astype(int)
        obs_2d      = obs_arr[:, 2:4]

        n_cams, n_pts = len(camera_ids), len(pts3d)
        cam_params = np.zeros((n_cams, 6), dtype=np.float64)
        for i, cid in enumerate(camera_ids):
            rvec, _ = cv2.Rodrigues(result.camera_poses[cid]["R"])
            cam_params[i, :3] = rvec.ravel()
            cam_params[i, 3:] = result.camera_poses[cid]["t"].ravel()

        x0 = np.concatenate([cam_params.ravel(), pts3d.ravel()])

        err_before = float(np.sqrt(np.mean(self._residuals(x0, n_cams, n_pts, cam_indices, pt_indices, obs_2d) ** 2)))
        print(f"[BA] {n_cams} cameras, {n_pts} points, {len(observations)} observations")
        print(f"[BA] Error before: {err_before:.4f} px")

        A = self._sparsity(n_cams, n_pts, cam_indices, pt_indices)
        opt = least_squares(
            self._residuals, x0,
            jac_sparsity=A, verbose=1, x_scale="jac",
            ftol=self.ftol, method="trf",
            max_nfev=500,
            args=(n_cams, n_pts, cam_indices, pt_indices, obs_2d),
        )

        err_after = float(np.sqrt(np.mean(opt.fun ** 2)))
        print(f"[BA] Error after:  {err_after:.4f} px")

        refined_cams = opt.x[: n_cams * 6].reshape(n_cams, 6)
        refined_pts  = opt.x[n_cams * 6 :].reshape(n_pts, 3)

        refined_poses = {}
        for i, cid in enumerate(camera_ids):
            R, _ = cv2.Rodrigues(refined_cams[i, :3])
            t = refined_cams[i, 3:].reshape(3, 1)
            refined_poses[cid] = {"R": R, "t": t, "P": self.K @ np.hstack([R, t])}

        all_pts = result.points_3d.copy()
        if sub_idx is not None:
            all_pts[sub_idx] = refined_pts
        else:
            all_pts = refined_pts

        return BundleAdjustmentResult(
            points_3d=all_pts,
            camera_poses={**result.camera_poses, **refined_poses},
            reprojection_error_before=err_before,
            reprojection_error_after=err_after,
            n_observations=len(observations),
        )

    def _project(self, cam_params: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(cam_params[:3])
        pts_cam = (R @ pts3d.T).T + cam_params[3:]
        pts_proj = pts_cam[:, :2] / pts_cam[:, 2:3]
        return pts_proj * np.array([self.K[0, 0], self.K[1, 1]]) + np.array([self.K[0, 2], self.K[1, 2]])

    def _residuals(self, params, n_cams, n_pts, cam_indices, pt_indices, obs_2d):
        cams = params[: n_cams * 6].reshape(n_cams, 6)
        pts  = params[n_cams * 6 :].reshape(n_pts, 3)
        res  = []
        for ci, pi, (ox, oy) in zip(cam_indices, pt_indices, obs_2d):
            p = self._project(cams[ci], pts[pi:pi+1])
            res.extend([p[0, 0] - ox, p[0, 1] - oy])
        return np.array(res, dtype=np.float64)

    def _sparsity(self, n_cams, n_pts, cam_indices, pt_indices):
        A = lil_matrix((len(cam_indices) * 2, n_cams * 6 + n_pts * 3), dtype=int)
        for i, (ci, pi) in enumerate(zip(cam_indices, pt_indices)):
            A[2*i,   ci*6 : ci*6+6] = 1
            A[2*i+1, ci*6 : ci*6+6] = 1
            A[2*i,   n_cams*6 + pi*3 : n_cams*6 + pi*3+3] = 1
            A[2*i+1, n_cams*6 + pi*3 : n_cams*6 + pi*3+3] = 1
        return A