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