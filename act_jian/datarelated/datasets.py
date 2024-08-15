import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import h5py

# utils
from act_jian.utils_all.utils_depth import natural_sort_key
from act_jian.utils_all.constants import JOINT_POSITIONS_LIMITS
from act_jian.utils_all.utils_obs import normalize_position, normalize_image
import clip


class JianRLBenchDataset(Dataset):
    '''
    load hdf5 files from the specified directory
    '''

    def __init__(self, action_chunk_size=200, data_dir=None):
        # TODO: config task
        self.action_chunk_size = action_chunk_size
        self.data_dir = data_dir
        self.episodes = sorted([d for d in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, d))], key=natural_sort_key)
        self.iteration_each_episode = np.zeros(len(self.episodes), dtype=int)

        joint_position_all = []
        for i in range(len(self.episodes)):
            with h5py.File(os.path.join(self.data_dir, f"episode_{i}.h5"), 'r') as f:
                self.iteration_each_episode[i] = f['joint_positions'].shape[0]
                joint_position_all.append(f['joint_positions'][:])
        self.accumulated_iteration = np.cumsum(self.iteration_each_episode)
        joint_position_all = np.concatenate(joint_position_all, axis=0)

        # check if all joint_position_all is within the limits
        assert np.all(joint_position_all >= JOINT_POSITIONS_LIMITS[:, 0]), "joint_position_limit error"
        assert np.all(joint_position_all <= JOINT_POSITIONS_LIMITS[:, 1]), "joint_position_limit error"

    def __len__(self):
        return self.accumulated_iteration[-1]

    def __getitem__(self, idx):
        return_dict = dict()  # a dict to give out 1. images, 2. current_position, 3. future_position, 4. is_data_mask

        # find the epoisode idx

        episode_idx = np.argmax(self.accumulated_iteration > idx)
        read_path = os.path.join(self.data_dir, f"episode_{episode_idx}.h5")
        frame_idx = idx - self.accumulated_iteration[episode_idx - 1] if episode_idx > 0 else idx
        # print(episode_idx, frame_idx)
        # Firstly, retrieve data
        with h5py.File(read_path, 'r') as f:
            # get all images
            # need to firstly get some data to find the length of episode, despite unefficient
            actions_all_joints = f['joint_positions'][:]
            actions_all_gripper = f['gripper_open'][:].reshape(-1, 1)
            actions_all = np.concatenate((actions_all_joints, actions_all_gripper), axis=1)
            # sample one frame            # get other data
            # first of all: image
            images_all = []
            left_shoulder_rgb = f['left_shoulder_rgb'][frame_idx]
            # left_shoulder_depth = np.expand_dims(f['left_shoulder_depth'][sample_frame], axis=0)
            right_shoulder_rgb = f['right_shoulder_rgb'][frame_idx]
            # right_shoulder_depth = np.expand_dims(f['right_shoulder_depth'][sample_frame], axis=0)
            wrist_rgb = f['wrist_rgb'][frame_idx]
            # wrist_depth = np.expand_dims(f['wrist_depth'][sample_frame], axis=0)
            front_rgb = f['front_rgb'][frame_idx]
            # front_depth = np.expand_dims(f['front_depth'][sample_frame], axis=0)
            overhead_rgb = f['overhead_rgb'][frame_idx]
            # overhead_depth = np.expand_dims(f['overhead_depth'][sample_frame], axis=0)

            # RGBD实验
            # images_all.append(np.concatenate((left_shoulder_rgb, left_shoulder_depth), axis=0))
            # images_all.append(np.concatenate((right_shoulder_rgb, right_shoulder_depth), axis=0))
            # images_all.append(np.concatenate((wrist_rgb, wrist_depth), axis=0))
            # images_all.append(np.concatenate((front_rgb, front_depth), axis=0))
            # images_all.append(np.concatenate((overhead_rgb, overhead_depth), axis=0))

            # Pure ACT 的RGB输入
            images_all.append(left_shoulder_rgb)
            images_all.append(right_shoulder_rgb)
            images_all.append(wrist_rgb)
            images_all.append(front_rgb)
            images_all.append(overhead_rgb)

            images_all = np.stack(images_all, axis=0)  # [num_cameras, channels(RGBD), height, width]

            semantics = f['variation_descriptions '][:]
        # Secondly, prepare data
        # 先padding成action chunk的大小
        current_position = actions_all[frame_idx]
        future_position = actions_all[frame_idx:]
        length_future = len(future_position)

        # padding to action_chunk_size
        if length_future < self.action_chunk_size:
            future_position = np.concatenate((future_position, np.zeros((self.action_chunk_size - length_future, future_position.shape[1]))), axis=0)
            is_pad = np.full(self.action_chunk_size, False)
            is_pad[length_future:] = True
        else:
            future_position = future_position[:self.action_chunk_size]
            is_pad = np.full(self.action_chunk_size, False)
            is_pad[:] = False

        # norm with mean and std
        images_all = normalize_image(images_all)
        current_position[:7] = normalize_position(current_position[:7])
        future_position[:, :7] = normalize_position(future_position[:, :7])

        # Tokenize semantics
        text = clip.tokenize(str(semantics[np.random.randint(len(semantics))])).squeeze(0)

        # return
        return_dict['images'] = torch.tensor(images_all).float()
        return_dict['current_position'] = torch.tensor(current_position).float()
        return_dict['future_position'] = torch.tensor(future_position).float()
        return_dict['is_pad'] = torch.tensor(is_pad).bool()
        return_dict['idx'] = idx
        return_dict['semantics'] = text

        return return_dict


if __name__ == "__main__":
    '''
    For Debug
    '''
    # # validate
    test = JianRLBenchDataset(data_dir='/media/jian/data/rlbench_hdf5/train/close_jar')
    DataLoader1 = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)

    for i, data_dict in enumerate(DataLoader1):
        this_images = data_dict['images'][0][0]
        # np.save('/home/jian/git_all/git_manipulation/act_jian/z_others/image_example.npy', this_images)
        break
        pass
    # test hdf5
    # with h5py.File('/media/jian/data/rlbench_hdf5/val/close_jar/episode_0.h5', 'r') as f:
    #     print(f.keys())
    #     print(f['variation_descriptions '][:])
