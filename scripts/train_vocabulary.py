"""Train a BoW visual vocabulary from saved keyframe descriptors."""

import numpy as np

from src.loop_closure.vocabulary import Vocabulary
from src.vo.keyframe_logger import load_keyframes


def train_vocabulary(keyframes_path, output_path, k=1000):
    print(f"Loading keyframes from {keyframes_path}")
    keyframes = load_keyframes(keyframes_path)
    print(f"Loaded {len(keyframes)} keyframes")

    # stack all descriptors into one big (N_total, 32) array
    all_descriptors = np.concatenate([kf['descriptors'] for kf in keyframes], axis=0)
    print(f"Total descriptors: {len(all_descriptors)}")

    vocab = Vocabulary(k=k)
    vocab.train(all_descriptors)
    vocab.save(output_path)


if __name__ == "__main__":
    train_vocabulary(
        keyframes_path="results/keyframes/kitti_07.npz",
        output_path="results/vocab/kitti_07_vocab.npz",
        k=1000,
    )