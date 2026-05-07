"""
Run bundle adjustment on keyframes and landmarks, save optimized trajectory.
"""

import json
from pathlib import Path
import numpy as np

from src.bundle_adjustment.builder import build_ba_graph, extract_optimized_poses
from src.pose_graph.optimizer import optimize  # reuse existing optimizer
from src.utils.config import parse_config_arg
from src.utils.io import save_kitti_trajectory
from src.vo.keyframe_logger import load_keyframes
from src.vo.landmarks import load_landmarks
from src.loop_closure.detector import LoopClosure 


def load_loops(path):
    """Load loop closures from JSON file."""
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


def main():
    cfg = parse_config_arg()

    # load keyframes
    print("Loading keyframes...")
    keyframes = load_keyframes(str(cfg.keyframes_path))
    print(f"  {len(keyframes)} keyframes")

    # load landmarks
    print("Loading landmarks...")
    lm_map = load_landmarks(str(cfg.landmarks_path))
    print(f"  {lm_map.n_landmarks} landmarks (before filtering)")

    # load loop closures
    print("Loading loop closures...")
    loops = load_loops(str(cfg.loops_path))
    print(f"  {len(loops)} loop closures")

    # get camera intrinsics from config
    K = cfg.K

    # save un-optimized keyframe trajectory for comparison
    initial_poses = np.stack([kf['pose'] for kf in keyframes])
    save_kitti_trajectory(initial_poses, str(cfg.initial_trajectory_path))
    print(f"Saved un-optimized trajectory to {cfg.initial_trajectory_path}")

    # build bundle adjustment graph
    print("\nBuilding bundle adjustment graph...")
    graph, initial, frame_idx_map = build_ba_graph(
        keyframes=keyframes,
        lm_map=lm_map,
        K_matrix=K,
        loops=None, # BA+loops degrades performance (15.6m) vs BA alone (7.5m); use PGO for loop handling
        projection_sigma=10.0,
        min_observations=10,
    )
        
    # optimize
    print("\nOptimizing...")
    result, info = optimize(graph, initial, max_iterations=50, verbose=True)

    # extract optimized poses and save full trajectory in KITTI format
    optimized_poses = extract_optimized_poses(result, keyframes)
    ba_output_path = str(cfg.ba_optimized_trajectory_path)
    save_kitti_trajectory(optimized_poses, str(ba_output_path))
    print(f"\nSaved BA-optimized trajectory to {ba_output_path}")

    # save optimization info
    info_output_path = str(cfg.ba_optimization_info_path)
    with open(info_output_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved optimization info to {info_output_path}")

    print("\n=== Done ===")
    print("Evaluate with:")
    print(f"  evo_ape kitti {cfg.gt_path} {cfg.vo_trajectory_path} --align --plot")
    print(f"  evo_ape kitti {cfg.gt_path} {ba_output_path} --align --plot")


if __name__ == "__main__":
    main()