import numpy as np

from src.loop_closure.bow import compute_idf, encode_tfidf
from src.loop_closure.vocabulary import Vocabulary


class BowDatabase:
    """Stores BoW encodings of keyframes and supports similarity queries."""

    def __init__(self, vocabulary: Vocabulary):
        self.vocab = vocabulary
        self.k = vocabulary.k

        self.idf = None # (k,) IDF weights, set in build()
        self.encodings = None # (N, k) TF-IDF encodings

    def build(self, keyframes):
        """
        Build the database from a list of keyframes.
        Args:
            keyframes: list of dicts with 'descriptors' field (from KeyframeLogger)
        """
        N = len(keyframes)
        print(f"Building BoW database: {N} keyframes, k={self.k}")
    
        # step 1: convert each keyframe's descriptors to word indices
        print("  computing word indices...")
        all_word_indices = [self.vocab.transform(kf['descriptors']) for kf in keyframes]

        # step 2: compute IDF over corpus
        print("  computing IDF...")
        self.idf = compute_idf(all_word_indices, self.k)

        # step 3: encode each keyframe with TF-IDF
        print("  encoding keyframes with TF-IDF...")
        self.encodings = np.stack([
            encode_tfidf(words, self.idf, self.k) for words in all_word_indices
        ]) 

        print(f"  done. encodings shape: {self.encodings.shape}")

    def query(self, descriptors, top_k=10, exclude_indices=None):
        """
        Query the database with a new image's descriptors.
        Args:
            descriptors: (M, 32) ORB descriptors
            top_k: how many top results to return
            exclude_indices: optional list of keyframe indices to exclude from
                results (e.g., temporal neighbors of the query)

        Returns:
            (top_k,) array of keyframe indices, sorted by similarity (highest first)
            (top_k,) array of similarity scores in [0, 1]
        """
        if self.encodings is None:
            raise RuntimeError("Database not built. Call build() first.")

        # encode query
        words = self.vocab.transform(descriptors)
        query_vec = encode_tfidf(words, self.idf, self.k)

        # cosine similarity = dot product of L2-normalized vectors
        scores = self.encodings @ query_vec # (N,) similarity scores

        # exclude specified indices
        if exclude_indices is not None:
            scores = scores.copy()
            scores[list(exclude_indices)] = -1.0 # set excluded scores to -1 so they sort last

        # top_k by score
        n_results = min(top_k, len(scores))
        # argpartition is O(N), but argsort is O(N log N), so we do argpartition first to get top_k candidates, then sort those
        top_idx = np.argpartition(-scores, n_results - 1)[:n_results] # unsorted top_k indices
        # sort top_k properly (argpartition doesn't guarantee order)
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_scores = scores[top_idx]

        return top_idx, top_scores

    def query_by_index(self, query_idx, top_k=10, temporal_window=5):
        """
        Query the database with an existing keyframe, excluding temporal neighbors.
        Args:
            query_idx: keyframe index to use as query
            top_k: how many top results to return
            temporal_window: exclude keyframes within +/- this many indices of query

        Returns:
            (top_k,) array of keyframe indices, sorted by similarity
            (top_k,) array of similarity scores
        """
        if self.encodings is None:
            raise RuntimeError("Database not built. Call build() first.")
        
        query_vec = self.encodings[query_idx]
        scores = self.encodings @ query_vec
        scores = scores.copy() # avoid modifying original scores

        # exclude self and temporal neighbors
        lo = max(0, query_idx - temporal_window)
        hi = min(len(scores), query_idx + temporal_window + 1)
        scores[lo:hi] = -1.0 # set excluded scores to -1

        n_results = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, n_results - 1)[:n_results]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_scores = scores[top_idx]

        return top_idx, top_scores