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
    # load inputs
    print("Loading keyframes and loops...")
    keyframes = load_keyframes("results/keyframes/kitti_07.npz")
    loops = load_loops("results/loops/kitti_07_loops.json")
    print(f"  {len(keyframes)} keyframes, {len(loops)} verified loops")

    # save un-optimized keyframe trajectory for comparison
    initial_poses = np.stack([kf['pose'] for kf in keyframes])
    save_kitti_trajectory(initial_poses, "results/trajectories/kitti_07_initial.txt")
    print("Saved un-optimized trajectory to results/trajectories/kitti_07_initial.txt")

    # build pose graph
    print("\nBuilding pose graph...")
    graph, initial = build_pose_graph(keyframes, loops)

    # optimize
    print("\nOptimizing pose graph...")
    result, info = optimize(graph, initial)

    # extract optimized poses
    optimized_poses = extract_poses(result, n=len(keyframes))

    # save optimized trajectory
    save_kitti_trajectory(optimized_poses, "results/trajectories/kitti_07_optimized.txt")
    print("Saved optimized trajectory to results/trajectories/kitti_07_optimized.txt")

    # save optimization info
    info_path = Path("results/trajectories/kitti_07_optimization_info.json")
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved optimization info to {info_path}")

    print("\n=== Done ===")
    print("Next: interpolate to full trajectory, then evaluate with evo")
    print("  python -m scripts.interpolate_full_trajectory")
    print("  evo_ape kitti data/kitti/poses/07.txt results/trajectories/kitti_07_vo.txt --align")
    print("  evo_ape kitti data/kitti/poses/07.txt results/trajectories/kitti_07_optimized_full.txt --align")

if __name__ == "__main__":
    main()