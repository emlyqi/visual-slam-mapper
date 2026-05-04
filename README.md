# SLAM from Scratch

Visual SLAM in Python, built from scratch and validated on KITTI Odometry. Implements stereo visual odometry, BoW loop closure detection, and pose graph optimization with GTSAM. IMU fusion in progress.

![KITTI 05 trajectory](results/kitti_05/plots/trajectory_comparison_trajectories.png)

## Results

## Results

### KITTI 05 — full pipeline

2.7km urban driving with multiple loop revisits, evaluated against GPS/IMU ground truth.

| Metric | VO baseline | After PGO |
|---|---|---|
| ATE RMSE | 7.65 m | **2.03 m** (73% reduction) |
| ATE max | 19.48 m | 4.56 m |
| RPE (100m) | 1.27 m / 100m (1.27%) | 1.70 m / 100m |
| Loops detected | — | 1447 |

### KITTI 07 — VO only

695m urban driving with loops only at trajectory endpoints. PGO does not improve ATE on this sequence (see writeup for analysis).

| Metric | Value |
|---|---|
| ATE RMSE | 2.42 m |
| RPE (100m) | 1.37 m / 100m (1.37%) |
| Path length ratio (VO/GT) | 1.001 |

## Requirements

This project requires **Linux, macOS, or WSL2**. The `gtsam` pip wheel has a known bug on Python 3.10 + Ubuntu 22.04 that causes segfaults on basic operations (see [borglab/gtsam#1880](https://github.com/borglab/gtsam/issues/1880)). The conda-forge build works correctly, so this project uses conda for `gtsam` and pip for everything else.

## Setup

Install miniconda if you don't have it:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate
```

Create the environment and install dependencies:

```bash
git clone https://github.com/emlyqi/slam-from-scratch
cd slam-from-scratch

conda env create -f environment.yml
conda activate slam

pip install -r requirements.txt
```

Download KITTI Odometry sequences (grayscale + calibration + ground truth poses) and arrange under `data/kitti/`:

```
data/kitti/
├── 05/
│   ├── image_0/
│   ├── image_1/
│   ├── poses/05.txt
│   ├── calib.txt
│   └── times.txt
└── 07/
└── (same structure)
```

## Run

The pipeline is config-driven. Replace `kitti_05` with `kitti_07` (or any new config) to switch sequences.

```bash
python -m scripts.run_vo --config configs/kitti_05.yaml
python -m scripts.train_vocabulary --config configs/kitti_05.yaml
python -m scripts.detect_loops --config configs/kitti_05.yaml
python -m scripts.run_pose_graph --config configs/kitti_05.yaml
python -m scripts.interpolate_full_trajectory --config configs/kitti_05.yaml

evo_ape kitti data/kitti/05/poses/05.txt results/kitti_05/trajectories/vo.txt --align --plot
evo_ape kitti data/kitti/05/poses/05.txt results/kitti_05/trajectories/optimized_full.txt --align --plot
evo_rpe kitti data/kitti/05/poses/05.txt results/kitti_05/trajectories/optimized_full.txt --align --delta 100 --delta_unit m --plot
```

## Read more

[WRITEUP.md](WRITEUP.md) covers methodology, design decisions, and detailed results.