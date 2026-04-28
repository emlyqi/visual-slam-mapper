import numpy as np
from src.data.kitti_loader import KittiSequence
from src.vo.features import detect_features
from src.vo.stereo import compute_disparity, triangulate_points


seq = KittiSequence("data/kitti")
left, right, _ = seq[0]

# disparity map
disp = compute_disparity(left, right)
print(f"Disparity: shape={disp.shape}, "
      f"valid={(disp > 0).sum()}/{disp.size} ({100*(disp > 0).mean():.1f}%)")
print(f"Disparity range (valid): {disp[disp > 0].min():.2f} to {disp[disp > 0].max():.2f}")

# triangulate ORB keypoints
pts2d = detect_features(left)
pts3d, valid = triangulate_points(pts2d, disp, seq.K, seq.baseline)
print(f"3D points: {len(pts3d)} / {len(pts2d)} keypoints had valid depth")
print(f"Depth range: {pts3d[:, 2].min():.2f}m to {pts3d[:, 2].max():.2f}m")
print(f"Median depth: {np.median(pts3d[:, 2]):.2f}m")