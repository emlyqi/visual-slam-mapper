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

from src.utils.transforms import invert_se3
from src.vo.keyframe_logger import load_keyframes


def load_kitti_trajectory(path):
    """Load (N, 4, 4) trajectory from KITTI format (one line per pose, 12 floats)."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(v) for v in line.split()]
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return np.stack(poses)


def save_kitti_trajectory(poses_4x4, output_path):
    """Save (N, 4, 4) optimized poses in KITTI format (one line per pose, 12 floats)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for T in poses_4x4:
            row = T[:3, :].flatten()
            f.write(' '.join(f"{v:.6e}" for v in row) + '\n')


def main():
    # load original VO trajectory (all 1101 frames)
    print("Loading original VO trajectory...")
    vo_poses = load_kitti_trajectory("results/trajectories/kitti_07_vo.txt")
    n_frames = len(vo_poses)
    print(f"  Loaded {n_frames} VO frames")

    # load keyframes
    print("Loading keyframes...")
    keyframes = load_keyframes("results/keyframes/kitti_07.npz")
    kf_indices = np.array([kf['frame_idx'] for kf in keyframes])
    n_keyframes = len(keyframes)
    print(f"  Loaded {n_keyframes} keyframes")

    # load optimized keyframe poses
    print("Loading optimized keyframe poses...")
    optimized_kf_poses = load_kitti_trajectory("results/trajectories/kitti_07_optimized.txt")
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
    output_path = "results/trajectories/kitti_07_optimized_full.txt"
    save_kitti_trajectory(full_poses, output_path)
    print(f"Saved {n_frames}-frame optimized trajectory to {output_path}")


if __name__ == "__main__":
    main()