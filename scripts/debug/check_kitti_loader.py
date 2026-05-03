"""Verify KittiSequence loads frames and calibration correctly."""

from src.data.kitti_loader import KittiSequence
from src.utils.config import parse_config_arg


def main():
    cfg = parse_config_arg()
    seq = KittiSequence(cfg.data_dir)

    print(f"Sequence: KITTI {cfg.sequence_id}")
    print(f"Frames: {len(seq)}")
    print(f"K:\n{seq.K}")
    print(f"Baseline: {seq.baseline:.4f} m")

    left, right, fid = seq[0]
    print(f"Frame {fid}: left {left.shape}, right {right.shape}")


if __name__ == "__main__":
    main()