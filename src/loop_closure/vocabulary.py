from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class Vocabulary:
    """K-means visual vocabulary over ORB descriptors, for bag-of-words loop closure."""
    def __init__(self, k=1000, random_state=42):
        self.k = k
        self.random_state = random_state
        self._kmeans = None # set after train() or load()

    def train(self, descriptors, batch_size=4096, verbose=True):
        """
        Train the vocabulary by clustering a stack of descriptors.

        Args:
            descriptors: (N, 32) uint8 array of ORB descriptors aggregated from
                many images. Convert binary descriptors to float for k-means
                (Hamming distance approximation via Euclidean in {0..255}^32).
            batch_size: minibatch size for MiniBatchKMeans.
            verbose: print progress.
        Returns:
            self (for chaining).
        """

        if descriptors.ndim != 2 or descriptors.shape[1] != 32:
            raise ValueError(f"Expected (N, 32) descriptors, got {descriptors.shape}")
        if len(descriptors) < self.k:
            raise ValueError(
                f"Cannot train k={self.k} clusters from only {len(descriptors)} descriptors. "
                f"Need at least k descriptors."
            )

        X = descriptors.astype(np.float32) # convert binary descriptors to float for k-means

        if verbose:
            print(f"Training vocabulary: k={self.k}, n_descriptors={len(X)}")

        self._kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            batch_size=batch_size,
            random_state=self.random_state,
            n_init=3,
            max_iter=100,
            verbose=int(verbose),
        )
        self._kmeans.fit(X) # train

        if verbose:
            print(f"Vocabulary training complete. Inertia: {self._kmeans.inertia_:.0f}")

        return self

    def transform(self, descriptors):
        """
        Assign each descriptor to its nearest visual word.

        Args:
            descriptors: (M, 32) uint8 array.
        Returns:
            (M,) int array of word indices in [0, k).
        """

        if self._kmeans is None:
            raise RuntimeError("Vocabulary not trained or loaded yet. Call train() or load() first.")
        if len(descriptors) == 0:
            return np.empty((0,), dtype=np.int32)
        if descriptors.ndim != 2 or descriptors.shape[1] != 32:
            raise ValueError(f"Expected (M, 32) descriptors, got {descriptors.shape}")
        
        # sklearn's predict() computes Euclid. distance from each input descriptor to each cluster center, returns idx of nearest
        return self._kmeans.predict(descriptors.astype(np.float32))

    def save(self, path):
        """Save vocabulary cluster centers to .npz."""
        if self._kmeans is None:
            raise RuntimeError("Cannot save untrained vocabulary.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            cluster_centers=self._kmeans.cluster_centers_,
            k=self.k,
        )
        print(f"Saved vocabulary to {path} (k={self.k})")

    def load(self, path):
        """Load vocabulary cluster centers from .npz."""
        path = Path(path)
        data = np.load(str(path))
        self.k = int(data['k'])
        
        # rebuild a usable kmeans object from saved centers
        self._kmeans = MiniBatchKMeans(n_clusters=self.k)
        self._kmeans.cluster_centers_ = data['cluster_centers']

        # MiniBatchKMeans needs these for predict() to work; we don't actually
        # use them but sklearn checks they exist
        self._kmeans._n_threads = 1
        
        return self

    @property
    def cluster_centers(self):
        """(k, 32) array of cluster centers in descriptor space."""
        if self._kmeans is None:
            return None
        return self._kmeans.cluster_centers_