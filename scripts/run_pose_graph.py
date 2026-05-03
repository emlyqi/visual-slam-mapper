"""
Build pose graph from keyframes + loops, optimize, save corrected trajectory.

Compares ATE before/after optimization.
"""

import json
from pathlib import Path
import numpy as np

from src.loop_closure.detector import LoopClosure
from src.pose_graph.builder import build_pose_graph, extract_poses
from src.pose_graph.optimizer import optimize
from src.utils.config import parse_config_arg
from src.vo.keyframe_logger import load_keyframes


def load_loops(path):
    """Load loop closures from JSON file saved by detect_loops.py."""
    with open(path, 'r') as f:
        data = json.load(f)
    loops = []
    for loop in data['loops']:
        loops.append(LoopClosure(
            kf_a=loop['kf_a'],
            kf_b=loop['kf_b'],
            T_a_to_b=np.array(loop['T_a_to_b']),
            n_inliers=loop['n_inliers'],
            n_matches=loop['n_matches'],
            bow_score=loop['bow_score'],
        ))

    return loops


def save_kitti_trajectory(poses_4x4, output_path):
    """Save (N, 4, 4) optimized poses in KITTI format (one line per pose, 12 floats)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for T in poses_4x4:
            row = T[:3, :].flatten()
            f.write(' '.join(f"{v:.6e}" for v in row) + '\n')


def main():
    cfg = parse_config_arg()

    # load inputs
    print("Loading keyframes and loops...")
    keyframes = load_keyframes(str(cfg.keyframes_path))
    loops = load_loops(str(cfg.loops_path))
    print(f"  {len(keyframes)} keyframes, {len(loops)} verified loops")

    # save un-optimized keyframe trajectory for comparison
    initial_poses = np.stack([kf['pose'] for kf in keyframes])
    save_kitti_trajectory(initial_poses, str(cfg.initial_trajectory_path))
    print(f"Saved un-optimized trajectory to {cfg.initial_trajectory_path}")

    # build pose graph
    print("\nBuilding pose graph...")
    graph, initial = build_pose_graph(
        keyframes, loops,
        odom_trans_sigma=cfg.odom_trans_sigma,
        odom_rot_sigma_deg=cfg.odom_rot_sigma_deg,
        loop_trans_sigma_base=cfg.loop_trans_sigma_base,
        loop_rot_sigma_deg_base=cfg.loop_rot_sigma_deg_base,
        loop_inlier_ref=cfg.loop_inlier_ref,
    )

    # optimize
    print("\nOptimizing pose graph...")
    result, info = optimize(graph, initial)

    # extract optimized poses
    optimized_poses = extract_poses(result, n=len(keyframes))

    # save optimized trajectory
    save_kitti_trajectory(optimized_poses, str(cfg.optimized_trajectory_path))
    print(f"Saved optimized trajectory to {cfg.optimized_trajectory_path}")

    # save optimization info
    info_path = cfg.optimization_info_path
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved optimization info to {info_path}")

    print("\n=== Done ===")
    print("Next: interpolate to full trajectory, then evaluate with evo")
    print(f"  python -m scripts.interpolate_full_trajectory --config configs/kitti_{cfg.sequence_id}.yaml")
    print(f"  evo_ape kitti {cfg.gt_path} {cfg.vo_trajectory_path} --align")
    print(f"  evo_ape kitti {cfg.gt_path} {cfg.optimized_full_trajectory_path} --align")

if __name__ == "__main__":
    main()