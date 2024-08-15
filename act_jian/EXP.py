# framework package
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# My core package
from act_jian.datarelated.datasets import JianRLBenchDataset
from act_jian.policy import ACTPolicy

# RLbench package
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import MT15_V1, CloseJar
from rlbench.observation_config import ObservationConfig

# utils package
from utils_all.utils_obs import normalize_position, denormalize_position, normalize_image
import yaml

'''
Jian: To make my code clean, this file only contain the code of trainning and evaluation.
      Details of the model can be found in policy.py
'''


class jianact(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.policy_config = config['policy_config']
        self.policy = ACTPolicy(self.policy_config)

    def configure_optimizers(self):
        optimizer = self.policy.configure_optimizers()
        return optimizer

    def training_step(self, data):
        self.policy.train()
        forward_dict = self._forward_pass(data, self.policy)
        loss = forward_dict['loss']
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, data):
        self.policy.eval()
        forward_dict = self._forward_pass(data, self.policy)
        loss = forward_dict['loss']
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def _forward_pass(self, data, policy):
        image_data = data['images']
        qpos_data = data['current_position']
        action_data = data['future_position']
        is_pad = data['is_pad']
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        result = policy(qpos_data, image_data, action_data, is_pad)
        return result

    # lightningModel hooks, lightningmodules has +20 hooks to keep all the flexibility
    @torch.no_grad()
    def evaluate_env(self):
        # 1. Set parameters
        # 1.1 same parameters of evaluation
        eval_episodes = 2
        temporal_flag = True
        max_time_steps = 250
        state_dim = 8
        num_queries = 200  # chunk size
        self.policy.eval()

        if temporal_flag:
            query_frequncy = 1
            all_time_actions = torch.zeros([max_time_steps, max_time_steps + num_queries, state_dim]).cuda()
        else:
            query_frequncy = 10
        # 1.2 functions that are only used here

        def process_obs(obs):
            ''' retrive the images and current position from the observation'''
            # get images
            images = []
            images.append(obs.left_shoulder_rgb)
            images.append(obs.right_shoulder_rgb)
            images.append(obs.wrist_rgb)
            images.append(obs.front_rgb)
            images.append(obs.overhead_rgb)
            images = torch.tensor(np.stack(images, axis=0)).float()
            images = images.permute(0, 3, 1, 2)
            images = normalize_image(images)
            images = images.unsqueeze(0)
            images = images.cuda()

            # get current position
            joint_position = obs.joint_positions
            gripper = np.array([obs.gripper_open])
            current_position = torch.tensor(np.concatenate((joint_position, gripper), axis=0)).float()
            current_position[:7] = normalize_position(current_position[:7])
            current_position = current_position.unsqueeze(0)
            current_position = current_position.cuda()

            return current_position, images

        # 2. Set up simulators
        # 2.1 Environment setup
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(absolute_mode=True),
            gripper_action_mode=Discrete()
        )

        observation_config = ObservationConfig()
        observation_config.gripper_joint_positions = True

        env = Environment(action_mode, obs_config=observation_config)
        env.launch()

        # 2.2 Task setup
        task = env.get_task(CloseJar)
        task.sample_variation()  # random variation
        des, obs = task.reset()
        terminate = False

        # 3. main loop
        for episode in range(eval_episodes):
            for t in range(max_time_steps):
                if terminate == False:
                    qpos, curr_image = process_obs(obs)
                    if t % query_frequncy == 0:  # sample every query_frequncy
                        all_actions = self.policy(qpos, curr_image)

                    # 3.1 predict normalnized actions
                    if temporal_flag:
                        all_time_actions[t, t:t + num_queries, :] = all_actions  # store all actions for each time step
                        actions_for_curr_step = all_time_actions[:, t]
                        action_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[action_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        normalized_actions = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        normalized_actions = all_actions.squeeze()[t % query_frequncy]

                    # 3.2 denormalize actions
                    normalized_actions = normalized_actions.squeeze().detach().cpu().numpy()
                    normalized_actions[:7] = denormalize_position(normalized_actions[:7])
                    real_world_action = normalized_actions

                    # 3.3 proceed the action
                    obs, reward, terminate = task.step(real_world_action)
                    terminate = terminate.T
            success, _ = task._task.success()
            print(f'Episode {episode} success: {success}')
            task.sample_variation()  # random variation
            des, obs = task.reset()


config_path = '/home/jian/git_all/git_manipulation/act_jian/act_jian/jianact.yaml'
load_ckpt_path = '/media/jian/second/single_task_close_jar_ckpt/expriment4_2epochepoch=529.ckpt'

train_dataset_path = '/media/jian/data/rlbench_hdf5/train/close_jar'
val_dataset_path = '/media/jian/data/rlbench_hdf5/val/close_jar'

ckpt_save_path = '/media/jian/second/1/'
expriment_name = 'exp6'  # ckpt name depends on this

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# Load and preprocess the MNIST dataset
# Dataloader

train_dataset = JianRLBenchDataset(config['policy_config']['chunk_size'], train_dataset_path)
val_dataset = JianRLBenchDataset(config['policy_config']['chunk_size'], val_dataset_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)

checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    save_top_k=-1,  # Save all checkpoints
    dirpath=ckpt_save_path,  # Directory to save the checkpoints
    filename=expriment_name + '{epoch:03d}'  # Checkpoint filename
)


def train():
    model = jianact(config)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=30000, devices='auto')
    trainer.fit(model, train_loader, test_loader)


def eval():
    model = jianact.load_from_checkpoint(load_ckpt_path, config=config)
    model.evaluate_env()


def train_from_ckpt():
    model = jianact.load_from_checkpoint(load_ckpt_path, config=config)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=30000, devices='auto')
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    # train()
    eval()
