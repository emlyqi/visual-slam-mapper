"""Render trajectory as Open3D camera frustums + GT line."""

import numpy as np
import open3d as o3d

from src.data.kitti_loader import KittiSequence


# load
seq = KittiSequence("data/kitti")
K = seq.K
poses = np.loadtxt("results/trajectories/kitti_07_vo.txt").reshape(-1, 3, 4)
gt = np.loadtxt("data/kitti/poses/07.txt").reshape(-1, 3, 4)

geoms = []

# camera frustums for VO trajectory (subsample so it's not 1100 of them)
for T_3x4 in poses[::5]:
    T = np.eye(4)
    T[:3, :] = T_3x4
    # create_camera_visualization expects extrinsic = world->camera, so invert
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=1226, view_height_px=370,
        intrinsic=K,
        extrinsic=np.linalg.inv(T),
        scale=3.0,
    )
    frustum.paint_uniform_color([0.2, 0.5, 1.0])  # blue
    geoms.append(frustum)

# GT trajectory as a line
gt_pts = gt[:, :, 3]
gt_line = o3d.geometry.LineSet()
gt_line.points = o3d.utility.Vector3dVector(gt_pts)
gt_line.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(gt_pts) - 1)])
gt_line.paint_uniform_color([0.5, 0.5, 0.5])
geoms.append(gt_line)

# coordinate frame at origin for reference
geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))

# open viewer
o3d.visualization.draw_geometries(geoms, window_name="KITTI 07 VO Frustums")