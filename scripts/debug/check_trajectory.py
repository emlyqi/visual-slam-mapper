import numpy as np
from src.utils.transforms import make_se3
from src.vo.trajectory import Trajectory


traj = Trajectory()

# simulate 5 frames moving forward 1m each, no rotation
# T_curr_from_prev: in current frame, the previous frame is at z = -1
# (because we moved forward by 1, so prev point is now behind us)
R = np.eye(3)
t = np.array([0, 0, -1.0]) # ← point transform: prev point is 1m behind in curr frame
T_curr_from_prev = make_se3(R, t)

for _ in range(5):
    traj.add_motion(T_curr_from_prev)

print("Positions:")
print(traj.positions())