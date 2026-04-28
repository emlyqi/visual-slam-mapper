import numpy as np
from pathlib import Path

from src.utils.transforms import invert_se3


class Trajectory:
    """Accumulates camera poses over time, starting from identity at the first frame."""
    
    def __init__(self):
        # poses[i] = camera pose at frame i in world frame (4x4 SE(3) matrix)
        self.poses = [np.eye(4)]

    def add_motion(self, T_curr_from_prev):
        """Append a new pose given the relative point transform from PnP."""
        T_motion = invert_se3(T_curr_from_prev)  # camera-to-camera motion
        T_new = self.poses[-1] @ T_motion  # accumulate new pose in world frame
        self.poses.append(T_new)

    def positions(self):
        """Return (N, 3) array of camera positions over time."""
        return np.array([pose[:3, 3] for pose in self.poses])
    
    def __len__(self):
        return len(self.poses)

    def save_kitti(self, path):
        """Save trajectory in KITTI format (3x4 row-major per line)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for pose in self.poses:
                row = pose[:3, :].flatten() # take top 3 rows -> (3, 4) -> flatten to (12,)
                f.write(" ".join(f"{x:.6e}" for x in row) + "\n")