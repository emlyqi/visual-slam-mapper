import numpy as np


def make_se3(R, t):
    """Build 4x4 SE(3) transformation matrix from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()
    return T


def invert_se3(T):
    """Invert a 4x4 SE(3) transformation matrix."""
    # faster than np.linalg.inv
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_se3(R_inv, t_inv)


def se3_to_pose3(T):
    """Convert a 4x4 SE(3) matrix to a gtsam.Pose3."""
    import gtsam
    R = T[:3, :3]
    t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*t))


def pose3_to_se3(pose):
    """Convert a gtsam.Pose3 to a 4x4 SE(3) matrix."""
    R = pose.rotation().matrix()
    t = pose.translation()
    return make_se3(R, t)