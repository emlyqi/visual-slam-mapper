import cv2
import numpy as np
from src.utils.transforms import make_se3


def estimate_motion(points_3d, points_2d, K, reproj_threshold=2.0, iterations=100, min_inliers=10):
    """
    Estimate camera motion from 3D-2D correspondences using PnP.
    Args:
        points_3d: (N, 3) array of 3D points in the world frame
        points_2d: (N, 2) array of corresponding pixel coordinates in the current image
        K: (3, 3) camera intrinsic matrix
    Returns:
        T: (4, 4) SE(3) transformation matrix from world to camera frame
        inliers: indices of points that survived RANSAC outlier rejection
        success: boolean indicating if a valid pose was found
    """
    
    if len(points_3d) < 4:
        return np.eye(4), np.array([]), False  # not enough points for PnP
    
    # OpenCV's solvePnPRansac expects (N, 1, 3) and (N, 1, 2) shapes
    obj_points = points_3d.astype(np.float32).reshape(-1, 1, 3)
    img_points = points_2d.astype(np.float32).reshape(-1, 1, 2)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=K,
        distCoeffs=None, # KITTI images are rectified, so no distortion
        reprojectionError=reproj_threshold,
        iterationsCount=iterations,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < min_inliers:
        return np.eye(4), np.array([]), False  # PnP failed

    # convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()

    # construct the SE(3) transformation matrix
    T = make_se3(R, t)

    return T, inliers.flatten(), True