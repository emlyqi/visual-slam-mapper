from src.data.kitti_loader import KittiSequence


seq = KittiSequence("data/kitti")

print(f"Frames: {len(seq)}")
print(f"K:\n{seq.K}")
print(f"Baseline: {seq.baseline:.4f} m")

left, right, fid = seq[0]
print(f"Frame {fid}: left {left.shape}, right {right.shape}")