import numpy as np
from src.utils.transforms import make_se3, invert_se3


def test_invert_round_trip():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90° about z
    t = np.array([1, 2, 3], dtype=float)
    T = make_se3(R, t)
    assert np.allclose(T @ invert_se3(T), np.eye(4))
    assert np.allclose(invert_se3(T) @ T, np.eye(4))


if __name__ == "__main__":
    test_invert_round_trip()
    print("ok")