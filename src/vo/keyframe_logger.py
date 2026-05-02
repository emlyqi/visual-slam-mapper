import json
from pathlib import Path

import numpy as np


class KeyframeLogger:
    """Accumulates per-keyframe artifacts during VO and saves them as a single .npz file
    plus a sidecar .json with metadata."""
    def __init__(self):
        self.keyframes = []

    def add(self, frame_idx, pose, points_2d, points_3d, descriptors, timestamp=None):
        """
        Record a keyframe.

        Args:
            frame_idx: KITTI frame index (int)
            pose: (4, 4) world-from-camera SE(3) pose at this keyframe
            points_2d: (N, 2) float32 pixel coords of features at this keyframe
            points_3d: (N, 3) float32 3D points in camera frame, triangulated from stereo
            descriptors: (N, 32) uint8 ORB binary descriptors
            timestamp: float, optional
        """
        self.keyframes.append({
            'frame_idx': int(frame_idx),
            'pose': np.asarray(pose, dtype=np.float32),
            'points_2d': np.asarray(points_2d, dtype=np.float32),
            'points_3d': np.asarray(points_3d, dtype=np.float32),
            'descriptors': np.asarray(descriptors, dtype=np.uint8),
            'timestamp': float(timestamp) if timestamp is not None else -1.0,
        })

    def save(self, output_path):
        """
        Save keyframes to disk.

        Writes:
            {output_path}.npz - all numerical arrays, packed
            {output_path}.json - metadata (frame indices, timestamps, sizes)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # arrays of mixed sizes go in object dtype; numpy supports this in npz
        npz_data = {
            'frame_indices': np.array([k['frame_idx'] for k in self.keyframes], dtype=np.int32),
            'poses': np.stack([k['pose'] for k in self.keyframes]),  # (K, 4, 4)
            'timestamps': np.array([k['timestamp'] for k in self.keyframes], dtype=np.float64),
            # variable-length per-keyframe arrays stored as object arrays
            'points_2d': np.array([k['points_2d'] for k in self.keyframes], dtype=object),
            'points_3d': np.array([k['points_3d'] for k in self.keyframes], dtype=object),
            'descriptors': np.array([k['descriptors'] for k in self.keyframes], dtype=object),
        }
        np.savez_compressed(str(output_path) + '.npz', **npz_data)

        # human-readable sidecar
        meta = {
            'n_keyframes': len(self.keyframes),
            'frame_indices': [k['frame_idx'] for k in self.keyframes],
            'feature_counts': [len(k['points_2d']) for k in self.keyframes],
        }
        with open(str(output_path) + '.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {len(self.keyframes)} keyframes to {output_path}.npz (+ .json)")

    def __len__(self):
        return len(self.keyframes)


def load_keyframes(npz_path):
    """
    Load keyframes saved by KeyframeLogger.save.

    Returns a list of dicts, same structure as keyframes added via .add().
    """
    path = Path(npz_path)
    if path.suffix != '.npz':
        path = path.with_suffix('.npz')

    data = np.load(str(path), allow_pickle=True)
    n = len(data['frame_indices'])
    keyframes = []
    for i in range(n):
        keyframes.append({
            'frame_idx': int(data['frame_indices'][i]),
            'pose': data['poses'][i],
            'points_2d': data['points_2d'][i],
            'points_3d': data['points_3d'][i],
            'descriptors': data['descriptors'][i],
            'timestamp': float(data['timestamps'][i]),
        })
    return keyframes