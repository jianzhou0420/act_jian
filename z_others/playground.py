
import h5py
import os
import numpy as np


data_dir = "/media/jian/data/rlbench_hdf5/train/close_jar"
JOINT_POSITIONS_LIMITS = np.array([[-2.8973, 2.8973],
                                   [-1.7628, 1.7628],
                                   [-2.8973, 2.8973],
                                   [-3.0718, -0.0698],
                                   [-2.8973, 2.8973],
                                   [-0.0175, 3.7525],
                                   [-2.8973, 2.8973]])


with h5py.File(os.path.join(data_dir, f"episode_{0}.h5"), 'r') as f:
    actions_all_joints = f['joint_positions'][100]
    pass
normalized_actions_all_joints = (actions_all_joints - JOINT_POSITIONS_LIMITS[:, 0]) / (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0])
test = normalized_actions_all_joints * (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0]) + JOINT_POSITIONS_LIMITS[:, 0]
pass
