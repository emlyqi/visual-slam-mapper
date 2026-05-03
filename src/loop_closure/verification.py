"""
Geometric verification of loop closure candidates.

Given two candidate keyframes from BoW retrieval, this module:
1. Matches their ORB descriptors (Hamming distance, brute force, mutual best)
2. Filters matches by Lowe's ratio test
3. Recovers relative pose via PnP+RANSAC using stored 3D points
4. Returns the pose if enough inliers were found, otherwise rejects

This is the second filter in the loop closure pipeline. BoW retrieval is fast
but produces many false positives (perceptually similar but geometrically
different scenes). Geometric verification filters them out by requiring
real feature correspondences and a recoverable rigid transformation.
"""

import cv2
import numpy as np

from src.utils.transforms import make_se3, invert_se3


# brute-force matcher with Hamming distance for ORB binary descriptors
# crossCheck=False because we'll do mutual-best filtering manually with the ratio test
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_descriptors(desc_a, desc_b, ratio_threshold=0.75):
    """
    Match two sets of ORB descriptors using KNN + Lowe's ratio test.
    Args:
        desc_a: (M, 32) uint8 ORB descriptors from keyframe A
        desc_b: (N, 32) uint8 ORB descriptors from keyframe B
        ratio_threshold: Lowe's ratio test threshold, smaller = more strict
    Returns:
        (K, 2) int array. Row k = (idx_in_a, idx_in_b) for the k-th surviving match
    """
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), dtype=np.int32)

    # KNN match: for each descriptor in A, find its 2 nearest in B
    knn_matches = _matcher.knnMatch(desc_a, desc_b, k=2)

    # Lowe's ratio test: keep match if best distance is sufficiently better than 2nd best
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append((m.queryIdx, m.trainIdx))

    if len(good) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(good, dtype=np.int32)


def verify_pair(kf_a, kf_b, K, min_inliers=20, reproj_threshold=2.0):
    """
    Geometrically verify a loop closure candidate between two keyframes.
    Uses PnP-RANSAC with kf_a's 3D points (in its camera frame) as object points
    and kf_b's 2D features as image points. The recovered pose is T_b_from_a,
    where kf_b's camera is in kf_a's camera frame.
    Args:
        kf_a: dict with 'descriptors', 'points_3d' (in camera A frame)
        kf_b: dict with 'descriptors', 'points_2d'
        K: (3, 3) camera intrinsic matrix
        min_inliers: minimum PnP inliers to accept the loop
        reproj_threshold: pixel error threshold for RANSAC inliers
    Returns:
        success: bool, True if the loop is verified
        T_b_from_a: (4, 4) SE(3) pose, or identity if not verified
        n_inliers: int, number of PnP inliers (0 if not verified)
        n_matches: int, number of descriptor matches before PnP
    """
    desc_a = kf_a['descriptors']
    desc_b = kf_b['descriptors']
    pts3d_a = kf_a['points_3d']
    pts2d_b = kf_b['points_2d']

    # step 1: descriptor matching
    matches = match_descriptors(desc_a, desc_b)
    n_matches = len(matches)
    if n_matches < min_inliers:
        return False, np.eye(4, dtype=np.float32), 0, n_matches

    # step 2: gather 3D-2D correspondences
    obj_pts = pts3d_a[matches[:, 0]] # (n_matches, 3) in kf_a's camera frame
    img_pts = pts2d_b[matches[:, 1]] # (n_matches, 2) in kf_b's image plane

    # step 3: PnP-RANSAC geometric verification
    obj = obj_pts.astype(np.float32).reshape(-1, 1, 3)
    img = img_pts.astype(np.float32).reshape(-1, 1, 2)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=reproj_threshold,
        iterationsCount=200,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < min_inliers:
        return False, np.eye(4, dtype=np.float32), 0, n_matches

    # step 4: convert rvec+tvec to SE(3) pose
    R, _ = cv2.Rodrigues(rvec)
    # PnP returns the transform that takes points from kf_a's frame to kf_b's camera
    # (for projection into kf_b's image). For pose graph use, we want T_a_to_b in
    # the GTSAM BetweenFactor sense: pose_a.inverse() @ pose_b. These are inverses.
    T_pnp = make_se3(R, tvec.flatten())
    T_a_to_b = invert_se3(T_pnp)
    return True, T_a_to_b, len(inliers), n_matches