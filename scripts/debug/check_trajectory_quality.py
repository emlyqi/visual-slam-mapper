import numpy as np


# load VO output
poses = np.loadtxt("results/trajectories/kitti_07_vo.txt").reshape(-1, 3, 4)
positions = poses[:, :, 3]

print("=" * 50)
print("VO trajectory")
print("=" * 50)
print(f"Start:  {positions[0]}")
print(f"End:    {positions[-1]}")
print(f"Path length: {np.linalg.norm(np.diff(positions, axis=0), axis=1).sum():.1f} m")
print(f"Loop closure gap (start to end): {np.linalg.norm(positions[-1] - positions[0]):.1f} m")

# per-frame motion stats
diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
print(f"\nFrames with zero motion: {(diffs < 0.001).sum()}")
print(f"Median frame motion: {np.median(diffs):.3f} m")
print(f"Max frame motion: {diffs.max():.3f} m")

# load GT for comparison
print("\n" + "=" * 50)
print("Ground truth")
print("=" * 50)
gt = np.loadtxt("data/kitti/poses/07.txt").reshape(-1, 3, 4)
gt_positions = gt[:, :, 3]

print(f"GT start: {gt_positions[0]}")
print(f"GT end:   {gt_positions[-1]}")
print(f"GT path length: {np.linalg.norm(np.diff(gt_positions, axis=0), axis=1).sum():.1f} m")
print(f"GT loop closure gap: {np.linalg.norm(gt_positions[-1] - gt_positions[0]):.1f} m")

# scale ratio
vo_len = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
gt_len = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1).sum()
print(f"\nScale ratio (VO / GT): {vo_len / gt_len:.3f}")