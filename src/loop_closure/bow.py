import numpy as np


def encode_raw_histogram(word_indices, k):
    """
    Build a raw word-count histogram for one image.
    Args:
        word_indices: (M,) int array of word indices in [0, k)
        k: vocabulary size
    Returns:
        (k,) float32 array of word counts
    """
    hist = np.bincount(word_indices, minlength=k).astype(np.float32)
    return hist


def compute_idf(all_word_indices, k):
    """
    Compute inverse document frequency over a corpus of keyframes.
    IDF for word w = log(N / df(w)) where N is the total number of keyframes
    and df(w) is the number of keyframes that contain word w at least once
    Args:
        all_word_indices: list of (M_i,) int arrays, one per keyframe
        k: vocabulary size
    Returns:
        (k,) float32 array of IDF weights
    """
    N = len(all_word_indices)
    if N == 0:
        return np.zeros(k, dtype=np.float32)

    # df(w) = number of keyframes where word w appears at least once
    df = np.zeros(k, dtype=np.float32)
    for words in all_word_indices: # loop over keyframes
        unique_words = np.unique(words) # unique words in THIS keyframe
        df[unique_words] += 1.0 # +1 for each word that appeared in this keyframe

    # avoid log(N / 0) by adding 1 to df (smoothing)
    # words with df=0 get a small positive weight, words with high df get weight near 0
    idf = np.log(N / (df + 1.0)).astype(np.float32)
    return idf


def encode_tfidf(word_indices, idf, k):
    """
    Encode a single keyframe as a TF-IDF, L2-normalized vector.
    Args:
        word_indices: (M,) int array of word indices for this keyframe
        idf: (k,) array of IDF weights for the corpus
        k: vocabulary size
    Returns:
        (k,) float32 array, L2-normalized
    """
    if len(word_indices) == 0:
        return np.zeros(k, dtype=np.float32)

    # term frequency: count(w) / total_words_in_image
    raw = encode_raw_histogram(word_indices, k)
    tf = raw / len(word_indices)

    # TF-IDF 
    tfidf = tf * idf

    # L2 normalize so cosine similarity = dot product
    norm = np.linalg.norm(tfidf)
    if norm > 0:
        tfidf /= norm

    return tfidf