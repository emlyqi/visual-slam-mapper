"""
Build a GTSAM factor graph for bundle adjustment.

The factor graph has:
- N pose variables (one per keyframe)
- M point variables (one per landmark with ≥3 observations)
- K reprojection factors (one per observation of a landmark in a keyframe)
- 1 prior factor on the first pose (anchors the graph)

Each reprojection factor connects a pose + point and measures the pixel reprojection error.
"""

import gtsam
import numpy as np

from src.utils.transforms import se3_to_pose3, pose3_to_se3


def build_ba_graph(keyframes, lm_map, K_matrix, loops=None,
                   projection_sigma=1.0, min_observations=3,
                   loop_trans_sigma=0.2, loop_rot_sigma_deg=3.0):
    """
    Build a GTSAM bundle adjustment factor graph.
    Args:
        keyframes: list of dicts with 'pose' (4x4 world poses) and 'frame_idx' (KITTI frame index)
        lm_map: LandmarkMap instance with landmarks and observations
        K_matrix: (3, 3) camera intrinsic matrix
        loops: optional list of LoopClosure objects (same as PGO)
        projection_sigma: reprojection error noise (pixels), typically 1.0
        min_observations: filter landmarks with fewer observations than this
        loop_trans_sigma: loop closure translation noise (meters)
        loop_rot_sigma_deg: loop closure rotation noise (degrees)
    Returns:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values containing initial estimates for poses and points
        frame_idx_to_list_idx: dict mapping KITTI frame_idx to pose symbol index
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # filter landmarks to well-constrained ones
    lm_filtered = lm_map.filter_by_observations(min_observations=min_observations)
    print(f"Filtered to {lm_filtered.n_landmarks} landmarks with ≥{min_observations} observations")

    # build mapping from KITTI frame_idx to keyframe list index
    frame_idx_to_list_idx = {kf['frame_idx']: i for i, kf in enumerate(keyframes)}

    # create GTSAM calibration object
    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]
    K_gtsam = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    # prior on first pose: anchor graph at VO's frame0 pose
    print("Adding prior factor on first pose to anchor graph...")
    prior_sigmas = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])  # (r_x, r_y, r_z, t_x, t_y, t_z)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
    first_pose_sym = gtsam.symbol('x', 0)
    graph.add(gtsam.PriorFactorPose3(first_pose_sym, se3_to_pose3(keyframes[0]['pose']), prior_noise))

    # add pose variables
    print(f"Adding initial estimates for {len(keyframes)} keyframe poses...")
    for i, kf in enumerate(keyframes):
        pose_sym = gtsam.symbol('x', i)
        initial.insert(pose_sym, se3_to_pose3(kf['pose']))

    # add landmark variables
    print(f"Adding initial estimates for {lm_filtered.n_landmarks} landmarks...")
    for lid, lm in lm_filtered.landmarks.items():
        point_sym = gtsam.symbol('l', lid)
        initial.insert(point_sym, gtsam.Point3(lm['position_world']))

    # add reprojection factors
    print("Adding reprojection factors...")
    projection_noise = gtsam.noiseModel.Isotropic.Sigma(2, projection_sigma)
    n_factors = 0

    for lid, lm in lm_filtered.landmarks.items():
        pt_sym = gtsam.symbol('l', lid)

        for kf_frame_idx, pixel in lm['observations']:
            # convert KITTI frame_idx to keyframe list index
            if kf_frame_idx not in frame_idx_to_list_idx:
                continue

            kf_list_idx = frame_idx_to_list_idx[kf_frame_idx]
            pose_sym = gtsam.symbol('x', kf_list_idx)

            # create reprojection factor
            # GenericProjectionFactorCal3_S2(measured_pixel, noise, pose_key, point_key, K)
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(pixel[0], pixel[1]), projection_noise, 
                pose_sym, pt_sym, K_gtsam
            ))
            n_factors += 1

    # add loop closure factors (BetweenFactorPose3 constraints)
    if loops:
        print(f"Adding {len(loops)} loop closure factors...")
        loop_sigmas = np.array([
            np.deg2rad(loop_rot_sigma_deg),
            np.deg2rad(loop_rot_sigma_deg),
            np.deg2rad(loop_rot_sigma_deg),
            loop_trans_sigma,
            loop_trans_sigma,
            loop_trans_sigma
        ])
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)
        
        n_loop_factors = 0
        for loop in loops:
            # map KITTI frame indices to keyframe list indices
            if loop.kf_a not in frame_idx_to_list_idx or loop.kf_b not in frame_idx_to_list_idx:
                continue  # skip if loop references non-keyframe
            
            kf_a_idx = frame_idx_to_list_idx[loop.kf_a]
            kf_b_idx = frame_idx_to_list_idx[loop.kf_b]
            
            pose_a_sym = gtsam.symbol('x', kf_a_idx)
            pose_b_sym = gtsam.symbol('x', kf_b_idx)
            
            # add BetweenFactor
            graph.add(gtsam.BetweenFactorPose3(
                pose_a_sym, pose_b_sym,
                se3_to_pose3(loop.T_a_to_b),
                loop_noise
            ))
            n_loop_factors += 1
        
        print(f"  Added {n_loop_factors} loop closure BetweenFactors")
        n_factors += n_loop_factors

    print(f"Bundle adjustment graph: {len(keyframes)} poses, {lm_filtered.n_landmarks} landmarks, {n_factors} factors ({n_factors - (n_loop_factors if loops else 0)} projection + {n_loop_factors if loops else 0} loop)")
    
    return graph, initial, frame_idx_to_list_idx


def extract_optimized_poses(result, keyframes):
    """
    Extract optimized poses from GTSAM result.
    Args:
        result: gtsam.Values from optimization
        keyframes: list of keyframe dicts (to get the count) 
    Returns:
        poses: (N, 4, 4) array of optimized SE(3) poses
    """
    n = len(keyframes)
    poses = np.zeros((n, 4, 4))
    for i in range(n):
        pose_sym = gtsam.symbol('x', i)
        poses[i] = pose3_to_se3(result.atPose3(pose_sym))
    return poses


def extract_optimized_landmarks(result, landmark_ids):
    """
    Extract optimized landmark positions from GTSAM result.
    Args:
        result: gtsam.Values from optimization
        landmark_ids: list of landmark IDs to extract
    Returns:
        landmarks: dict {landmark_id: (3,) numpy array of optimized position}
    """
    landmarks = {}
    for lid in landmark_ids:
        point_sym = gtsam.symbol('l', lid)
        landmarks[lid] = result.atPoint3(point_sym)
    return landmarks