"""
Interpolate optimized keyframe poses back to all 1101 KITTI frames.

For non-keyframe frames f sitting between keyframes a and b, the original VO
gave us T_a_to_f (relative transform from kf_a to frame f). After PGO, kf_a's
world pose was updated. The new world pose for f is:

    new_pose[f] = optimized_pose[a] @ T_a_to_f

This propagates the optimization correction to all frames so we can compare
against the full 1101-frame KITTI ground truth.
"""

import json
from pathlib import Path
import numpy as np
import argparse

from src.utils.config import load_config
from src.utils.transforms import invert_se3
from src.utils.io import load_kitti_trajectory, save_kitti_trajectory
from src.vo.keyframe_logger import load_keyframes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kitti_05.yaml")
    parser.add_argument("--input", default=None, help="Trajectory to interpolate (default: optimized.txt)")
    parser.add_argument("--output", default=None, help="Output path (default: optimized_full.txt)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # use provided paths or fall back to defaults
    input_path = args.input if args.input else str(cfg.optimized_trajectory_path)
    output_path = args.output if args.output else str(cfg.optimized_full_trajectory_path)
    
    # load original VO trajectory (all 1101 frames)
    print("Loading original VO trajectory...")
    vo_poses = load_kitti_trajectory(str(cfg.vo_trajectory_path))
    n_frames = len(vo_poses)
    print(f"  Loaded {n_frames} VO frames")

    # load keyframes
    print("Loading keyframes...")
    keyframes = load_keyframes(str(cfg.keyframes_path))
    kf_indices = np.array([kf['frame_idx'] for kf in keyframes])
    n_keyframes = len(keyframes)
    print(f"  Loaded {n_keyframes} keyframes")

    # load optimized keyframe poses
    print("Loading optimized keyframe poses...")
    optimized_kf_poses = load_kitti_trajectory(input_path)
    assert len(optimized_kf_poses) == n_keyframes, \
        f"Expected {n_keyframes} optimized poses, got {len(optimized_kf_poses)}"
    print(f"  Loaded optimized poses for {len(optimized_kf_poses)} keyframes")

    # build full optimized trajectory (1101 frames) by interpolating between optimized keyframes
    print("Propagating optimization to full trajectory...")
    full_poses = np.zeros((n_frames, 4, 4))

    # for each frame, find its nearest preceding keyframe
    # frame_kf_idx[f] = index of nearest keyframe at or before frame f
    frame_kf_idx = np.zeros(n_frames, dtype=np.int32)
    current_kf = 0
    for f in range(n_frames):
        # advance current_kf while next keyframe is at or before frame f
        while current_kf + 1 < n_keyframes and kf_indices[current_kf + 1] <= f:
            current_kf += 1
        frame_kf_idx[f] = current_kf

    # for each frame, compute its pose as optimized[anchor_kf] @ relative_vo[anchor_kf_to_f]
    for f in range(n_frames):
        kf = frame_kf_idx[f]
        anchor_frame_idx = kf_indices[kf]

        # relative transofrm from anchor keyframe to this frame, as VO computed
        T_anchor_to_f = invert_se3(vo_poses[anchor_frame_idx]) @ vo_poses[f]

        # apply to optimized anchor pose
        full_poses[f] = optimized_kf_poses[kf] @ T_anchor_to_f

    # save
    save_kitti_trajectory(full_poses, output_path)  # output_path already set above
    print(f"Saved {n_frames}-frame optimized trajectory to {output_path}")


if __name__ == "__main__":
    main()