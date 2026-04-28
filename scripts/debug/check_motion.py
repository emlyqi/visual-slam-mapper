import numpy as np
from src.data.kitti_loader import KittiSequence
from src.vo.features import detect_features, track_features
from src.vo.stereo import compute_disparity, triangulate_points
from src.vo.motion import estimate_motion
from src.utils.transforms import invert_se3


seq = KittiSequence("data/kitti")
left0, right0, _ = seq[0]
left1, _, _ = seq[1]

# frame 0: detect, triangulate
pts2d_0 = detect_features(left0)
disp_0 = compute_disparity(left0, right0)
pts3d_0, valid = triangulate_points(pts2d_0, disp_0, seq.K, seq.baseline)
pts2d_0_valid = pts2d_0[valid]   # 2D pixels for surviving 3D points

# track those 2D points into frame 1
pts2d_0_tracked, pts2d_1, idx = track_features(left0, left1, pts2d_0_valid)

# use indices to pick corersponding 3D points for PnP
pts3d_for_pnp = pts3d_0[idx]

# run PnP
T, inliers, success = estimate_motion(pts3d_for_pnp, pts2d_1, seq.K)

print(f"Success: {success}")
print(f"Inliers: {len(inliers)} / {len(pts3d_for_pnp)}")
print(f"\nT_curr_from_prev:\n{T}")

# camera motion in previous frame is inverse of point transform
T_motion = invert_se3(T)
print(f"\nCamera motion (translation): {T_motion[:3, 3]}")
print(f"Translation magnitude: {np.linalg.norm(T_motion[:3, 3]):.3f} m")