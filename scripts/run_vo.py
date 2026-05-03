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
from src.utils.config import parse_config_arg
from src.utils.transforms import invert_se3
from src.vo.features import detect_features, track_features
from src.vo.keyframes import should_make_keyframe
from src.vo.keyframe_logger import KeyframeLogger
from src.vo.motion import estimate_motion
from src.vo.stereo import compute_disparity, triangulate_points
from src.vo.trajectory import Trajectory


def run_vo(data_path, output_path, kf_output_path=None, n_features=2000):
    seq = KittiSequence(data_path)
    K, baseline = seq.K, seq.baseline

    # trajectory initialized w identity pose at frame 0
    traj = Trajectory()
    kf_logger = KeyframeLogger()

    # keyframe state - set when a keyframe is created, used until it expires
    kf_pose = np.eye(4) # keyframe's pose in world coords (T_world_from_kf)
    kf_3d = None # (N, 3) 3D points in keyframe's camera frame
    n_kf_features = 0 # number of features in keyframe (for keyframe criterion)

    # per-frame tracking state - updated every iteration
    prev_image = None # most recent frame's left image
    prev_2d = None # (M, 2) 2D pixel locations of surviving features tracked in prev frame

    for i in tqdm(range(len(seq)), desc="VO"):
        left, right, ts = seq[i]

        ### first frame: just initialize keyframe state
        if i == 0:
            pts2d, descriptors = detect_features(left, n_features=n_features, return_descriptors=True)
            disp = compute_disparity(left, right)
            pts3d, valid = triangulate_points(pts2d, disp, K, baseline)
            kf_3d = pts3d[valid]
            n_kf_features = len(kf_3d)
            prev_image = left
            prev_2d = pts2d[valid]
            # trajectory already has identity pose from ctor

            # log the initial keyframe (frame 0, identity pose)
            kf_logger.add(
                frame_idx=i,
                pose=np.eye(4),
                points_2d=pts2d[valid],
                points_3d=pts3d[valid],
                descriptors=descriptors[valid],
                timestamp=ts,
            )
            continue

        ### every other frame: PnP first, ALWAYS!
        # optical flow: prev_image -> left
        _, curr_2d, idx = track_features(prev_image, left, prev_2d)

        # filter 3D points to match those tracked 2D points
        kf_3d_alive = kf_3d[idx]

        # PnP: kf_3d_alive -> curr_2d
        T_curr_from_kf, inliers, success = estimate_motion(kf_3d_alive, curr_2d, K)

        if not success:
            # PnP failed - repeat last pose, force keyframe refresh
            traj.poses.append(traj.poses[-1].copy())
            pts2d, descriptors = detect_features(left, n_features=n_features, return_descriptors=True)
            disp = compute_disparity(left, right)
            pts3d, valid = triangulate_points(pts2d, disp, K, baseline)
            kf_3d = pts3d[valid]
            n_kf_features = len(kf_3d)
            kf_pose = traj.poses[-1].copy() # keep same world pose since we didn't move
            prev_image = left
            prev_2d = pts2d[valid]

            # log the recovery keyframe
            kf_logger.add(
                frame_idx=i,
                pose=kf_pose,
                points_2d=pts2d[valid],
                points_3d=pts3d[valid],
                descriptors=descriptors[valid],
                timestamp=ts,
            )
            continue

        # compute and append this frame's pose
        # camera motion (kf -> curr) is the inverse of point transform (curr from kf)
        T_motion_kf_to_curr = invert_se3(T_curr_from_kf)

        # accumulate trajectory: world pose of curr = kf's world pose @ camera motion
        pose_curr = kf_pose @ T_motion_kf_to_curr
        traj.poses.append(pose_curr)

        ### check keyframe criterion
        if should_make_keyframe(T_motion_kf_to_curr, len(curr_2d), n_kf_features):
            # this frame becomes new keyframe - redetect + retriangulate to refresh 3D points
            pts2d, descriptors = detect_features(left, n_features=n_features, return_descriptors=True)
            disp = compute_disparity(left, right)
            pts3d, valid = triangulate_points(pts2d, disp, K, baseline)
            kf_3d = pts3d[valid]
            n_kf_features = len(kf_3d)
            kf_pose = pose_curr # new keyframe's world pose will be curr frame's world pose
            prev_image = left
            prev_2d = pts2d[valid] 

            # log the new keyframe
            kf_logger.add(
                frame_idx=i,
                pose=kf_pose,
                points_2d=pts2d[valid],
                points_3d=pts3d[valid],
                descriptors=descriptors[valid],
                timestamp=ts,
            )
        else :
            # keep tracking from this frame next iteration
            kf_3d = kf_3d_alive
            prev_image = left
            prev_2d = curr_2d

    ### save trajectory
    traj.save_kitti(output_path)
    print(f"Saved trajectory to {output_path} ({len(traj)} poses)")

    ### save keyframes
    if kf_output_path is not None:
        kf_logger.save(kf_output_path)

    return traj


if __name__ == "__main__":
    cfg = parse_config_arg()
    run_vo(
        cfg.data_dir,
        str(cfg.vo_trajectory_path),
        kf_output_path=str(cfg.keyframes_basename),
        n_features=cfg.n_features,
    )