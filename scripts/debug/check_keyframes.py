import numpy as np
from src.utils.transforms import make_se3
from src.vo.keyframes import should_make_keyframe


# test 1: small motion, most features intact -> no new keyframe
T_small = make_se3(np.eye(3), np.array([0, 0, 0.1]))
print(should_make_keyframe(T_small, n_tracked=1900, n_kf_features=2000))

# test 2: big translation, but most features still tracked -> new keyframe
T_big_trans = make_se3(np.eye(3), np.array([0, 0, 2.0]))
print(should_make_keyframe(T_big_trans, n_tracked=1900, n_kf_features=2000))

# test 3: low feature ratio -> new keyframe
print(should_make_keyframe(T_small, n_tracked=1000, n_kf_features=2000))