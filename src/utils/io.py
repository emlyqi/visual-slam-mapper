from pathlib import Path
import numpy as np


def save_kitti_trajectory(poses_4x4, output_path):
    """Save (N, 4, 4) poses in KITTI format (one line per pose, 12 floats)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for T in poses_4x4:
            row = T[:3, :].flatten()
            f.write(' '.join(f"{v:.6e}" for v in row) + '\n')


def load_kitti_trajectory(path):
    """Load (N, 4, 4) trajectory from KITTI format."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(v) for v in line.split()]
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return np.stack(poses)