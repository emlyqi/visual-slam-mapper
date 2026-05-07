"""Load YAML config and standardize result paths by sequence ID."""

import argparse
from pathlib import Path

import yaml


class Config:
    """Wrapper around a YAML config dict that builds standard result paths."""

    def __init__(self, data: dict):
        self.data = data
        self.sequence_id = data['sequence_id']
        self.data_dir = data['data_dir']
        self.gt_path = data['gt_path']
        self._results_root = Path("results") / f"kitti_{self.sequence_id}"

    def __getattr__(self, key):
        # fall through to data dict for params not explicitly named above
        if key in self.data:
            return self.data[key]
        raise AttributeError(f"Config has no key '{key}'")

    @property
    def K(self):
        """Load camera intrinsics from KITTI calib file."""
        from src.data.kitti_loader import KittiSequence
        seq = KittiSequence(self.data_dir)
        return seq.K

    @property
    def keyframes_path(self) -> Path:
        return self._results_root / "keyframes" / "keyframes.npz"

    @property
    def keyframes_meta_path(self) -> Path:
        return self._results_root / "keyframes" / "keyframes.json"

    @property
    def keyframes_basename(self) -> Path:
        # for KeyframeLogger.save() which appends .npz/.json itself
        return self._results_root / "keyframes" / "keyframes"

    @property
    def vocab_path(self) -> Path:
        return self._results_root / "vocab" / "vocab.npz"

    @property
    def loops_path(self) -> Path:
        return self._results_root / "loops" / "loops.json"

    @property
    def vo_trajectory_path(self) -> Path:
        return self._results_root / "trajectories" / "vo.txt"

    @property
    def initial_trajectory_path(self) -> Path:
        return self._results_root / "trajectories" / "initial.txt"

    @property
    def optimized_trajectory_path(self) -> Path:
        return self._results_root / "trajectories" / "optimized.txt"

    @property
    def optimized_full_trajectory_path(self) -> Path:
        return self._results_root / "trajectories" / "optimized_full.txt"

    @property
    def optimization_info_path(self) -> Path:
        return self._results_root / "trajectories" / "optimization_info.json"

    @property
    def landmarks_path(self):
        return self._results_root / "landmarks" / "landmarks.npz"

    @property
    def landmarks_basename(self) -> Path:
        return self._results_root / "landmarks" / "landmarks"

    @property
    def ba_optimized_trajectory_path(self) -> Path:
        return self._results_root / "trajectories" / "ba_optimized.txt"

    @property
    def ba_optimized_full_trajectory_path(self) -> Path:
        return self._results_root / "trajectories" / "ba_optimized_full.txt"
        
    @property  
    def ba_optimization_info_path(self) -> Path:
        return self._results_root / "trajectories" / "ba_optimization_info.json"


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data)


def parse_config_arg(default="configs/kitti_07.yaml") -> Config:
    """Standard arg parsing for scripts: `script.py --config <path>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default, help="Path to YAML config")
    args = parser.parse_args()
    return load_config(args.config)