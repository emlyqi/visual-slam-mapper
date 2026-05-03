"""
Full loop closure detection pipeline.

Combines BoW retrieval (coarse filter) with geometric verification (fine filter)
to produce a list of high-confidence loop closures across all keyframes.
"""

from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from src.loop_closure.database import BowDatabase
from src.loop_closure.verification import verify_pair


@dataclass
class LoopClosure:
    """One verified loop closure between two keyframes."""
    kf_a: int # query keyframe index
    kf_b: int # matched keyframe index
    T_a_to_b: np.ndarray # (4, 4) SE(3) transform from kf_a's camera frame to kf_b's camera frame
    n_inliers: int # number of inliers in geometric verification
    n_matches: int # number of descriptor matches before geometric verification
    bow_score: float # BoW similarity score between the two keyframes


def detect_loops(keyframes, database: BowDatabase, K, top_k=5,
                min_bow_score=0.5, temporal_window=20, min_inliers=20, reproj_threshold=2.0):
    """
    Detect loop closures across all keyframes.
    For each keyframe, query the BoW database, then geometrically verify the
    top candidates. Returns a list of verified loop closures.
    Args:
        keyframes: list of keyframe dicts (with descriptors, points_3d, points_2d)
        database: prebuilt BowDatabase
        K: (3, 3) camera intrinsics
        top_k: how many BoW candidates to verify per query
        min_bow_score: skip candidates below this BoW similarity
        temporal_window: skip candidates within +/- this many indices
        min_inliers: PnP inlier threshold
        reproj_threshold: PnP reprojection error threshold
    Returns:
        List[LoopClosure]
    """
    loops = []
    n = len(keyframes)

    print(f"Detecting loops: {n} keyframes, top_k={top_k}, "
          f"min_bow={min_bow_score}, min_inliers={min_inliers}")
    
    for query_idx in tqdm(range(n), desc="loop detection"):
        # BoW retrieval (excludes self + temporal neighbors)
        cand_idx, cand_scores = database.query_by_index(
            query_idx, top_k=top_k, temporal_window=temporal_window
        )

        for kf_b_idx, score in zip(cand_idx, cand_scores):
            if score < min_bow_score:
                continue

            # geometric verification
            success, T_a_to_b, n_inliers, n_matches = verify_pair(
                kf_a=keyframes[query_idx],
                kf_b=keyframes[kf_b_idx],
                K=K,
                min_inliers=min_inliers,
                reproj_threshold=reproj_threshold,
            )

            if success:
                loops.append(LoopClosure(
                    kf_a=int(query_idx),
                    kf_b=int(kf_b_idx),
                    T_a_to_b=T_a_to_b,
                    n_inliers=n_inliers,
                    n_matches=n_matches,
                    bow_score=float(score),
                ))

    # deduplicate loops (e.g., if A->B and B->A both pass verification, keep only one)
    loops = _deduplicate_loops(loops)
    return loops


def _deduplicate_loops(loops):
    """Keep one representative per (a, b) pair, regardless of direction."""
    best = {} # frozenset({a, b}) -> LoopClosure with highest n_inliers
    for loop in loops:
        key = frozenset({loop.kf_a, loop.kf_b}) # immutable
        if key not in best or loop.n_inliers > best[key].n_inliers:
            best[key] = loop
    return list(best.values())