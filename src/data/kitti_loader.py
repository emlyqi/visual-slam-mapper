from pathlib import Path
import numpy as np
import cv2


class KittiSequence:
    def __init__(self, sequence_dir):
        self.seq_dir = Path(sequence_dir)
        self.left_dir = self.seq_dir / "image_0"
        self.right_dir = self.seq_dir / "image_1"
        self.K, self.baseline = self._load_calib(self.seq_dir / "calib.txt")
        self.frame_ids = sorted([p.stem for p in self.left_dir.glob("*.png")])
    
    def _load_calib(self, calib_path):
        # KITTI calib: P0, P1, P2, P3 - 3x4 projection matrices
        # P0 = left grayscale, P1 = right grayscale
        # K is the upper-left 3x3 of P0
        # Baseline = -P1[0,3] / P1[0,0] (in meters)
        with open(calib_path) as f:
            lines = f.readlines()
        P0 = np.array(lines[0].strip().split()[1:], dtype=float).reshape(3, 4)
        P1 = np.array(lines[1].strip().split()[1:], dtype=float).reshape(3, 4)
        K = P0[:3, :3]
        baseline = -P1[0, 3] / P1[0, 0]
        return K, baseline
    
    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        left_img = cv2.imread(str(self.left_dir / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(self.right_dir / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        return left_img, right_img, int(frame_id)