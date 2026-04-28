# Visual SLAM Mapper

Stereo visual odometry on KITTI. The current module estimates 6-DOF camera trajectory from a stereo image stream using ORB features, Lucas-Kanade optical flow, and PnP-RANSAC pose estimation, with keyframe-anchored tracking.

Long-term goal: extend this into an indoor mapping system (loop closure, IMU fusion, dense reconstruction, 2D semantic floor plan), validated on my own custom sensor rig.

![KITTI 07 trajectory](results/plots/kitti_07_trajectory_trajectories.png)

## Results on KITTI 07

695m urban driving sequence, evaluated against GPS/IMU ground truth.

| Metric | Value |
|---|---|
| ATE RMSE | 2.4 m |
| RPE (100m) | 1.37 m / 100m (1.37%) |
| Path length ratio (VO/GT) | 1.001 |
| Runtime | ~2 min for 1101 frames on CPU |

## Setup

```bash
git clone https://github.com/[your-username]/visual-slam-mapper
cd visual-slam-mapper
python -m venv .venv && source .venv/Scripts/activate   # Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

Download KITTI Odometry sequence 07 (grayscale + calibration + ground truth poses) and arrange under `data/kitti/`:

data/kitti/
├── image_0, image_1, calib.txt, times.txt
└── poses/07.txt

## Run

```bash
python -m scripts.run_vo                          # produces trajectory file
python -m scripts.debug.check_trajectory_quality  # sanity-check stats vs GT
evo_ape kitti data/kitti/poses/07.txt results/trajectories/kitti_07_vo.txt --align --plot
evo_rpe kitti data/kitti/poses/07.txt results/trajectories/kitti_07_vo.txt --align --delta 100 --delta_unit m --plot
```

## Read more

[WRITEUP.md](WRITEUP.md) covers methodology, design decisions, and detailed results.