"""
Build a GTSAM pose graph from VO keyframes + verified loop closures.

The factor graph has:
- N pose variables (one per keyframe)
- N-1 odometry factors (consecutive keyframes, from VO relative motion)
- M loop closure factors (from verified loops, weighted by inlier count)
- 1 prior factor on the first pose (anchors the graph at origin)

Each factor is a BetweenFactorPose3 representing a relative SE(3) constraint.
The information matrix (inverse covariance) controls how strongly each constraint
is enforced during optimization.
"""

import gtsam
import numpy as np

from src.utils.transforms import invert_se3


def _se3_to_pose3(T):
    """Convert a 4x4 SE(3) matrix to a gtsam.Pose3."""
    R = T[:3, :3]
    t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*t))
    

def _pose3_to_se3(pose):
    """Convert a gtsam.Pose3 to a 4x4 SE(3) matrix."""
    R = pose.rotation().matrix()
    t = pose.translation()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# sigma = standard deviation of noise in each dimension = expected error of the measurement
# odom_trans_sigma=0.1                 : 0.1 meters of translation noise per odometry edge
# odom_rot_sigma_deg=2.0               : 2 degrees of rotation noise per odometry edge
# loop_trans_sigma_base=0.2            : 0.2 meters base for loop edges (scales with inliers)
# loop_rot_sigma_deg_base=3.0          : 3 degrees base for loop edges
# bigger loop closure errors bc they come from more uncertain place recognition + PnP
#      vs odometry which is more locally accurate
def build_pose_graph(keyframes, loops,
                     odom_trans_sigma=0.1, odom_rot_sigma_deg=2.0,
                     loop_trans_sigma_base=0.2, loop_rot_sigma_deg_base=3.0,
                     loop_inlier_ref=100):
    """
    Build a GTSAM pose graph.
    Args:
        keyframes: list of dicts with 'pose' (4x4 world poses from VO)
        loops: list of LoopClosure objects (kf_a, kf_b, T_a_to_b, n_inliers)
        odom_*: noise sigmas for odometry edges (translation in meters, rotation in degrees)
        loop_*_base: base noise sigmas for loop edges, scaled down by inlier confidence
        loop_inlier_ref: reference inlier count where loop sigma equals base sigma.
            Loops with more inliers get smaller sigma (higher information)
    Returns:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values containing initial estimates for all poses
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    n = len(keyframes)

    # prior on first pose: anchor graph at VO's frame0 pose
    # without this, the optimization would have a free 6-DOF gauge freedom
    print("Adding prior factor on first pose to anchor graph...")
    prior_sigmas = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])  # (r_x, r_y, r_z, t_x, t_y, t_z)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
    graph.add(gtsam.PriorFactorPose3(0, _se3_to_pose3(keyframes[0]['pose']), prior_noise))

    # add initial estimates for all poses (VO-derived starting point)
    print(f"Adding initial estimates for {n} keyframe poses...")
    for i, kf in enumerate(keyframes):
        initial.insert(i, _se3_to_pose3(kf['pose']))

    # odometry factors between consecutive keyframes
    print(f"Adding {n-1} odometry factors to graph...")
    odom_sigmas = np.array([
        np.deg2rad(odom_rot_sigma_deg),  # rotation noise in radians
        np.deg2rad(odom_rot_sigma_deg),
        np.deg2rad(odom_rot_sigma_deg),
        odom_trans_sigma,  # translation noise in meters
        odom_trans_sigma,
        odom_trans_sigma
    ])
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)

    for i in range(n - 1):
        T_world_a = keyframes[i]['pose']
        T_world_b = keyframes[i + 1]['pose']
        # relative transform: T_a_to_b = T_world_a^-1 @ T_world_b
        T_a_to_b = invert_se3(T_world_a) @ T_world_b
        graph.add(gtsam.BetweenFactorPose3(i, i + 1, _se3_to_pose3(T_a_to_b), odom_noise))

    # loop closure factors, weighed by inlier count
    print(f"Adding {len(loops)} loop closures to graph...")
    for loop in loops:
        # scale sigma by sqrt(inlier_ref / inliers): more inliers -> smaller sigma (higher confidence)
        scale = np.sqrt(loop_inlier_ref / max(loop.n_inliers, 1))  # avoid div by zero
        scale = np.clip(scale, 0.5, 2.0)  # don't let extreme values dominate

        loop_sigmas = np.array([
            np.deg2rad(loop_rot_sigma_deg_base) * scale,
            np.deg2rad(loop_rot_sigma_deg_base) * scale,
            np.deg2rad(loop_rot_sigma_deg_base) * scale,
            loop_trans_sigma_base * scale,
            loop_trans_sigma_base * scale,
            loop_trans_sigma_base * scale
        ])
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)

        # T_a_to_b from verification step is already the relative transform from a to b
        graph.add(gtsam.BetweenFactorPose3(
            loop.kf_a, loop.kf_b, 
            _se3_to_pose3(loop.T_a_to_b), 
            loop_noise,
        ))
        
    print(f"Pose graph: {n} nodes, {n-1} odometry edges, {len(loops)} loop edges")
    return graph, initial


def extract_poses(values, n):
    """Extract optimized poses as a (n, 4, 4) array of SE(3) matrices."""
    poses = np.zeros((n, 4, 4))
    for i in range(n):
        poses[i] = _pose3_to_se3(values.atPose3(i))
    return poses