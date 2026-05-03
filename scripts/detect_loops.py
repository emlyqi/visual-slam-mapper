"""Run BoW + geometric vertification across all keyframes on KITTI 07."""

import json
from pathlib import Path
import numpy as np

from src.data.kitti_loader import KittiSequence
from src.loop_closure.database import BowDatabase
from src.loop_closure.detector import detect_loops
from src.loop_closure.vocabulary import Vocabulary
from src.utils.config import parse_config_arg
from src.vo.keyframe_logger import load_keyframes


def main():
    cfg = parse_config_arg()

    seq = KittiSequence(cfg.data_dir)
    K = seq.K

    print("Loading vocabulary...")
    vocab = Vocabulary().load(str(cfg.vocab_path))

    print("Loading keyframes...")
    keyframes = load_keyframes(str(cfg.keyframes_path))
    print(f"Loaded {len(keyframes)} keyframes")

    print("Building BoW database...")
    db = BowDatabase(vocab)
    db.build(keyframes)

    print("Detecting loops...")
    loops = detect_loops(
        keyframes=keyframes,
        database=db,
        K=K,
        top_k=cfg.loop_top_k,
        min_bow_score=cfg.loop_min_bow_score,
        temporal_window=cfg.loop_temporal_window,
        min_inliers=cfg.loop_min_inliers,
        reproj_threshold=cfg.loop_reproj_threshold,
    )

    print(f"\nFound {len(loops)} verified loop closures")
    for loop in sorted(loops, key=lambda l: -l.n_inliers)[:20]: # print top 20 loops by inlier count
        t = loop.T_a_to_b[:3, 3]
        print(f"  kf {loop.kf_a:3d} <-> kf {loop.kf_b:3d}: "
            f"inliers={loop.n_inliers:3d}/{loop.n_matches:3d} "
            f"bow={loop.bow_score:.2f} "
            f"trans=[{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}]")

    # save
    out_path = cfg.loops_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "n_loops": len(loops),
        "params": {
            "top_k": cfg.loop_top_k, "min_bow_score": cfg.loop_min_bow_score,
            "temporal_window": cfg.loop_temporal_window,
            "min_inliers": cfg.loop_min_inliers, "reproj_threshold": cfg.loop_reproj_threshold,
        },
        "loops": [
            {
                "kf_a": l.kf_a, "kf_b": l.kf_b,
                "n_inliers": l.n_inliers, "n_matches": l.n_matches,
                "bow_score": l.bow_score,
                "T_a_to_b": l.T_a_to_b.tolist(),
            }
            for l in loops
        ],
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved loop closures to {out_path}")


if __name__ == "__main__":
    main()