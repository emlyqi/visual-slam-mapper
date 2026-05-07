"""
Main visual odometry loop on KITTI.

For each frame:
- If no active keyframe: detect ORB + stereo + triangulate, set as new keyframe.
- Otherwise: track features from prev frame to curr frame via optical flow,
  run PnP against the keyframe's 3D points, and accumulate pose into trajectory.
- After PnP, check keyframe criterion. If triggered, next iteration will create a new keyframe.
"""

import numpy as np
from tqdm import tqdm

from src.data.kitti_loader import KittiSequence
from src.loop_closure.verification import match_descriptors
from src.utils.config import parse_config_arg
from src.utils.transforms import invert_se3
from src.utils.io import save_kitti_trajectory
from src.vo.features import detect_features, track_features
from src.vo.keyframes import should_make_keyframe
from src.vo.keyframe_logger import KeyframeLogger
from src.vo.landmarks import LandmarkMap
from src.vo.motion import estimate_motion
from src.vo.stereo import compute_disparity, triangulate_points
from src.vo.trajectory import Trajectory


def run_vo(data_path, output_path, kf_output_path=None, lm_output_path=None, n_features=2000):
    seq = KittiSequence(data_path)
    K, baseline = seq.K, seq.baseline

    # trajectory initialized w identity pose at frame 0
    traj = Trajectory()
    kf_logger = KeyframeLogger()
    lm_map = LandmarkMap()

    # keyframe state - set when a keyframe is created, used until it expires
    kf_pose = np.eye(4) # keyframe's pose in world coords (T_world_from_kf)
    kf_3d = None # (N, 3) 3D points in keyframe's camera frame
    n_kf_features = 0 # number of features in keyframe (for keyframe criterion)
    prev_kf_frame_idx = -1 # most recent keyframe's frame index (for landmark matching)

    # per-frame tracking state - updated every iteration
    prev_image = None # most recent frame's left image
    prev_2d = None # (M, 2) 2D pixel locations of surviving features tracked in prev frame

    def initialize_keyframe(frame_idx, left_img, right_img, world_pose, ts):
        """
        Detect features in the current frame, triangulate, log keyframe, and register
        with the landmark map (matching against the previous keyframe's descriptors
        to inherit landmark IDs where possible).
        Side effects: updates kf_3d, n_kf_features, kf_pose, prev_kf_frame_idx
        Returns: pts2d_valid, pts3d_valid (the new keyframe's features)
        """
        nonlocal kf_3d, n_kf_features, kf_pose, prev_kf_frame_idx

        pts2d, descriptors = detect_features(left_img, n_features=n_features, return_descriptors=True)
        disp = compute_disparity(left_img, right_img)
        pts3d, valid = triangulate_points(pts2d, disp, K, baseline)

        pts2d_valid = pts2d[valid]
        pts3d_valid = pts3d[valid]
        descriptors_valid = descriptors[valid]

        # log keyframe
        kf_logger.add(
            frame_idx=frame_idx,
            pose=world_pose,
            points_2d=pts2d_valid,
            points_3d=pts3d_valid,
            descriptors=descriptors_valid,
            timestamp=ts,
        )

        # build inherited_landmark_ids by matching against previous keyframe's descriptors
        inherited = [-1] * len(pts2d_valid)
        if prev_kf_frame_idx >= 0 and len(kf_logger.keyframes) >= 2:
            prev_kf = kf_logger.keyframes[-2]
            prev_descriptors = prev_kf['descriptors']
            prev_local_to_landmark = lm_map.get_landmark_ids_for_keyframe(prev_kf_frame_idx)

            # match new descriptors to prev keyframe's descriptors
            # match_descriptors returns (K, 2) array of (idx_in_a, idx_in_b)
            # here a = prev, b = new
            matches = match_descriptors(prev_descriptors, descriptors_valid, ratio_threshold=0.85)

            # for each (prev_idx, new_idx), inherit the landmark from prev_idx
            for prev_idx, new_idx in matches:
                prev_idx = int(prev_idx)
                new_idx = int(new_idx)
                if prev_idx in prev_local_to_landmark:
                    inherited[new_idx] = prev_local_to_landmark[prev_idx]

        lm_map.add_keyframe(
            kf_frame_idx=frame_idx,
            kf_pose_world=world_pose,
            points_2d=pts2d_valid,
            points_3d_cam=pts3d_valid,
            inherited_landmark_ids=inherited,
        )

        kf_3d = pts3d_valid
        n_kf_features = len(kf_3d)
        kf_pose = world_pose
        prev_kf_frame_idx = frame_idx
        return pts2d_valid, pts3d_valid

    for i in tqdm(range(len(seq)), desc="VO"):
        left, right, ts = seq[i]

        if i == 0:
            pts2d_valid, _ = initialize_keyframe(i, left, right, np.eye(4), ts)
            prev_image = left
            prev_2d = pts2d_valid
            continue

        # optical flow + PnP
        _, curr_2d, idx = track_features(prev_image, left, prev_2d)
        kf_3d_alive = kf_3d[idx]
        T_curr_from_kf, inliers, success = estimate_motion(kf_3d_alive, curr_2d, K)

        if not success:
            # PnP failed - repeat last pose, force keyframe refresh
            traj.poses.append(traj.poses[-1].copy())
            pts2d_valid, _ = initialize_keyframe(i, left, right, traj.poses[-1].copy(), ts)
            prev_image = left
            prev_2d = pts2d_valid
            continue

        # accumulate trajectory
        T_motion_kf_to_curr = invert_se3(T_curr_from_kf)
        pose_curr = kf_pose @ T_motion_kf_to_curr
        traj.poses.append(pose_curr)

        # keyframe criterion
        if should_make_keyframe(T_motion_kf_to_curr, len(curr_2d), n_kf_features):
            pts2d_valid, _ = initialize_keyframe(i, left, right, pose_curr, ts)
            prev_image = left
            prev_2d = pts2d_valid
        else:
            # no keyframe - keep accumulating motion from the same keyframe
            kf_3d = kf_3d_alive
            prev_image = left
            prev_2d = curr_2d

    ### save trajectory
    traj.save_kitti_trajectory(output_path)
    print(f"Saved trajectory to {output_path} ({len(traj)} poses)")

    ### save keyframes and landmarks
    if kf_output_path is not None:
        kf_logger.save(kf_output_path)
    if lm_output_path is not None:
        lm_map.save(lm_output_path)

    return traj


if __name__ == "__main__":
    cfg = parse_config_arg()
    run_vo(
        cfg.data_dir,
        str(cfg.vo_trajectory_path),
        kf_output_path=str(cfg.keyframes_basename),
        lm_output_path=str(cfg.landmarks_basename),
        n_features=cfg.n_features,
    )