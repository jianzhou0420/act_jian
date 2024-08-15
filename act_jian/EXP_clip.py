# framework package
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.nn import functional as F

# My core package
from act_jian.datarelated.datasets import JianRLBenchDataset
from act_jian.detr.models.CVAE_clip import CVAE
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
import clip
'''
Jian: To make my code clean, this file only contain the code of trainning and evaluation.
      Details of the model can be found in CVAE.py
'''

torch.set_float32_matmul_precision('medium')


class jianact(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        model = CVAE(config)
        model.cuda()
        # param_dicts = [
        #     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        #     {
        #         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        #         "lr": config['lr_backbone'],
        #     },
        # ]
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                      weight_decay=config['weight_decay'])

        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = config['kl_weight']

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def training_step(self, data):
        # ZJA:data中有4个tensor，0：[8,1,3,480,640] 1:[8,1,14] 2:[8,400,14] 3:[8,400] bool,image_data, qpos_data, action_data, is_pad(在forwatd_pass中写了)
        self.model.train()
        forward_dict = self._forward_pass(data)
        loss = forward_dict['loss']
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, data):
        self.model.train()
        forward_dict = self._forward_pass(data)
        loss = forward_dict['loss']
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def _forward_pass(self, data):
        # ZJA:data中有4个tensor，0：[8,1,3,480,640] 1:[8,1,14] 2:[8,400,14] 3:[8,400] bool,image_data, qpos_data, action_data, is_pad(在forwatd_pass中写了)
        images = data['images']
        curr_pos = data['current_position']
        actions = data['future_position']
        is_pad = data['is_pad']
        text = data['semantics']  # already tokenized
        images, curr_pos, actions, is_pad, text = images.cuda(), curr_pos.cuda(), actions.cuda(), is_pad.cuda(), text.cuda()
        # Go Model

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, (mu, logvar) = self.model(images, text, curr_pos, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = self._kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(images, text, curr_pos)  # no action, sample from prior
            return a_hat

        # lightningModel hooks, lightningmodules has +20 hooks to keep all the flexibility
    def _kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

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

        env = Environment(action_mode, obs_config=observation_config, headless=False)
        env.launch()

        # 2.2 Task setup
        task = env.get_task(CloseJar)
        task.sample_variation()  # random variation
        des, obs = task.reset()
        text = clip.tokenize(str(des[np.random.randint(len(des))])).cuda()
        terminate = False

        # 3. main loop
        for episode in range(eval_episodes):
            for t in range(max_time_steps):
                if terminate == False:
                    qpos, curr_image = process_obs(obs)
                    if t % query_frequncy == 0:  # sample every query_frequncy
                        all_actions = self.policy(qpos, curr_image, text)

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
            text = clip.tokenize(str(des[np.random.randint(len(des))])).cuda()


config_path = '/home/jian/git_all/git_manipulation/act_jian/act_jian/jianact.yaml'
load_ckpt_path = '/media/jian/data/ckpt_tmp/exp6_2CLIPepoch=009.ckpt'

train_dataset_path = '/media/jian/data/rlbench_hdf5/train/close_jar'
val_dataset_path = '/media/jian/data/rlbench_hdf5/val/close_jar'

ckpt_save_path = '/media/jian/data/ckpt_tmp/'
expriment_name = 'exp6_2CLIP'  # ckpt name depends on this

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# Load and preprocess the MNIST dataset
# Dataloader

train_dataset = JianRLBenchDataset(config['chunk_size'], train_dataset_path)
val_dataset = JianRLBenchDataset(config['chunk_size'], val_dataset_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

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
    train()
    # eval()
    # train_from_ckpt()
