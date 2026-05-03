"""
Sanity check the BoW database.

For a few sample keyframes, print their top-5 most similar keyframes (excluding
self + temporal neighbors). For non-loop keyframes, the top matches should look
random (low scores). For keyframes near a loop, the top match should be the
corresponding revisit keyframe (high score).
"""

import numpy as np

from src.loop_closure.database import BowDatabase
from src.loop_closure.vocabulary import Vocabulary
from src.vo.keyframe_logger import load_keyframes


def main():
    vocab = Vocabulary().load("results/vocab/kitti_07_vocab.npz")
    keyframes = load_keyframes("results/keyframes/kitti_07.npz")
    print(f"Loaded {len(keyframes)} keyframes")

    db = BowDatabase(vocab)
    db.build(keyframes)

    print("\n=== Self-similarity check (should be ~1.0) ===")
    # query each keyframe with its OWN descriptors via the descriptor query path
    # this tests the encoding pipeline end-to-end
    for i in [0, 100, 250, 400]:
        top_idx, top_scores = db.query(keyframes[i]['descriptors'], top_k=5,
                                       exclude_indices=None)
        # top_idx[0] should be i, top_scores[0] should be ~1.0
        print(f"  keyframe {i}: top-5 = {top_idx} scores = {top_scores.round(3)}")
        assert top_idx[0] == i, f"keyframe {i} should be most similar to itself"
        assert top_scores[0] > 0.99, f"self-similarity should be ~1.0"

    print("\n=== Loop closure search (excluding temporal neighbors) ===")
    # KITTI 07 has its loop near the end where the trajectory revisits the start
    # query late keyframes, look for high-score matches at low keyframe indices
    print("Format: query_kf -> [(matched_kf, score), ...]")
    for query_kf in [0, 100, 250, 400, 500, 525]:
        top_idx, top_scores = db.query_by_index(query_kf, top_k=3, temporal_window=20)
        results = list(zip(top_idx.tolist(), top_scores.round(3).tolist()))
        print(f"  kf {query_kf:3d} -> {results}")


if __name__ == "__main__":
    main()