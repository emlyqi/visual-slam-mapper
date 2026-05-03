"""Verify optical flow tracks features across frames with reasonable retention.
Checks 1-frame and 10-frame gaps."""

from src.data.kitti_loader import KittiSequence
from src.utils.config import parse_config_arg
from src.vo.features import detect_features, track_features


def main():
    cfg = parse_config_arg()
    seq = KittiSequence(cfg.data_dir)

    left0, _, _ = seq[0]
    pts0 = detect_features(left0)
    print(f"Sequence: KITTI {cfg.sequence_id}")
    print(f"Detected: {len(pts0)} features")

    left1, _, _ = seq[1]
    prev_kept, curr_kept, _ = track_features(left0, left1, pts0)
    print(f"1-frame gap: {len(curr_kept)} / {len(pts0)} ({100*len(curr_kept)/len(pts0):.1f}%)")

    left10, _, _ = seq[10]
    prev_kept, curr_kept, _ = track_features(left0, left10, pts0)
    print(f"10-frame gap: {len(curr_kept)} / {len(pts0)} ({100*len(curr_kept)/len(pts0):.1f}%)")


if __name__ == "__main__":
    main()