import cv2
import numpy as np


def compute_disparity(left_image, right_image, num_disparities=128, block_size=5):
    """
    Compute disparity map from rectified stereo pair.
    Returns float32 disparity array in pixels. Invalid pixels return -1.0; valid disparities are positive.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity


def triangulate_points(points_2d, disparity, K, baseline, min_depth=1.0, max_depth=80.0):
    """
    Triangulate 3D points from 2D pixel coordinates and disparity.
    Args:
        points_2d: (N, 2) array of pixel coordinates in the left image
        disparity: (N,) array of disparity values in pixels
        K: (3, 3) camera intrinsic matrix
        baseline: stereo baseline in meters
        min_depth: minimum depth for filtering
        max_depth: maximum depth for filtering
    Returns:
        points_3d: (N, 3) array of 3D points in the left camera frame.
            Entries where `valid` is False are not meaningful.
        valid: (N,) boolean array indicating which input points are valid after depth filtering
    """
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = points_2d[:, 0]
    v = points_2d[:, 1]

    # sample disparity at integer pixel locations (nearest neighbour)
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)

    # bounds check
    H, W = disparity.shape
    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    d = np.full(len(points_2d), -1, dtype=np.float32)
    d[in_bounds] = disparity[v_int[in_bounds], u_int[in_bounds]]

    # compute depth where disparity is valid
    valid = d > 0
    Z = np.zeros_like(d)
    Z[valid] = fx * baseline / d[valid]

    # filter by depth range
    valid &= (Z >= min_depth) & (Z <= max_depth)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d, valid