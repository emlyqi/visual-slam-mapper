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
from src.utils.config import parse_config_arg
from src.vo.keyframe_logger import load_keyframes


def main():
    cfg = parse_config_arg()

    vocab = Vocabulary().load(str(cfg.vocab_path))
    keyframes = load_keyframes(str(cfg.keyframes_path))
    print(f"Loaded {len(keyframes)} keyframes")

    db = BowDatabase(vocab)
    db.build(keyframes)

    print("\n=== Self-similarity check (should be ~1.0) ===")
    # query each keyframe with its OWN descriptors via the descriptor query path
    # this tests the encoding pipeline end-to-end
    n = len(keyframes)
    sample_queries = [0, n // 4, n // 2, 3 * n // 4]
    for i in sample_queries:
        top_idx, top_scores = db.query(keyframes[i]['descriptors'], top_k=5,
                                       exclude_indices=None)
        # top_idx[0] should be i, top_scores[0] should be ~1.0
        print(f"  keyframe {i}: top-5 = {top_idx} scores = {top_scores.round(3)}")
        assert top_idx[0] == i, f"keyframe {i} should be most similar to itself"
        assert top_scores[0] > 0.99, f"self-similarity should be ~1.0"

    print("\n=== Loop closure search (excluding temporal neighbors) ===")
    print("Format: query_kf -> [(matched_kf, score), ...]")
    # sample queries spread across the sequence
    for query_kf in [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 5]:
        top_idx, top_scores = db.query_by_index(query_kf, top_k=3, temporal_window=20)
        results = list(zip(top_idx.tolist(), top_scores.round(3).tolist()))
        print(f"  kf {query_kf:3d} -> {results}")


if __name__ == "__main__":
    main()