# SLAM from Scratch

Visual SLAM in Python, built from scratch and validated on KITTI Odometry. Currently implements stereo visual odometry; loop closure, pose graph optimization, and IMU fusion in progress.

![KITTI 07 trajectory](results/plots/kitti_07_trajectory_trajectories.png)

## Results on KITTI 07

695m urban driving sequence, evaluated against GPS/IMU ground truth.

| Metric | Value |
|---|---|
| ATE RMSE | 2.4 m |
| RPE (100m) | 1.37 m / 100m (1.37%) |
| Path length ratio (VO/GT) | 1.001 |
| Runtime | ~2 min for 1101 frames on CPU |

## Requirements

This project requires **Linux, macOS, or WSL2**. The `gtsam` pip wheel has a known bug on Python 3.10 + Ubuntu 22.04 that causes segfaults on basic operations (see [borglab/gtsam#1880](https://github.com/borglab/gtsam/issues/1880)). The conda-forge build works correctly, so this project uses conda for `gtsam` and pip for everything else.

## Setup

Install miniconda if you don't have it:

​```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate
​```

Create the environment and install dependencies:

​```bash
git clone https://github.com/emlyqi/slam-from-scratch
cd slam-from-scratch

conda env create -f environment.yml
conda activate slam

pip install -r requirements.txt
​```

Download KITTI Odometry sequence 07 (grayscale + calibration + ground truth poses) and arrange under `data/kitti/`:

​```
data/kitti/
├── image_0/
├── image_1/
├── calib.txt
├── times.txt
└── poses/07.txt
​```

## Run

```bash
python -m scripts.run_vo                          # produces trajectory file
python -m scripts.debug.check_trajectory_quality  # sanity-check stats vs GT
evo_ape kitti data/kitti/poses/07.txt results/trajectories/kitti_07_vo.txt --align --plot
evo_rpe kitti data/kitti/poses/07.txt results/trajectories/kitti_07_vo.txt --align --delta 100 --delta_unit m --plot
```

## Read more

[WRITEUP.md](WRITEUP.md) covers methodology, design decisions, and detailed results.
