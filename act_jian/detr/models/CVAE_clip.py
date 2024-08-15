
'''
Modified from TonyZhao's ACT, Link:https://github.com/tonyzhaozh/act
V0.1: 2024/08/10: Running in the RLBench Datset
V0.2: 2024/08/14: Added Clip text encoder and repleaced the image encoder with clip image encoder
V0.3: 2024/08/15: Completely Reorganized the Code
'''
from .backbone_clip import get_clip_encoders
import numpy as np
from .transformer_clip import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from torch import nn
import torch

__author__ = 'Jian Zhou'


class CVAE(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Paramters from config
        d_action = config['action_dim']
        self.num_queries = num_queries = config['num_queries']

        # Components from other modules
        self.transformer = build_transformer(config)
        self.encoder = build_encoder(config)
        self.backbone_encode_image, self.backbone_encode_text = get_clip_encoders()
        d_model = self.transformer.d_model

        # Components for CVAE's encoder
        self.d_z = 32  # dimension of z
        self.cls_embed = nn.Embedding(1, d_model)  # extra cls token embedding
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, d_model))  # [CLS], qpos, a_seq

        # Components for CVAE's decoder
        self.PE_query = nn.Embedding(num_queries, d_model).weight  # nn.Embedding here is only used for create an autograd [num_queries, d_model] matrix
        self.PE_text_currPos_z = nn.Embedding(3, d_model).weight  # learned position embedding for proprio and latent
        self.register_buffer('PE_table_images', get_sinusoid_encoding_table(7 * 7, d_model))  # 4577:(feature_channel/model_dim)*feature_height*feature_width,

        # Components for input and output
        self.proj_action2hidden = nn.Linear(d_action, d_model)
        self.proj_pos2hidden = self.proj_action2hidden  # same as action2hidden in EXP6.2
        self.proj_latent2z = nn.Linear(d_model, self.d_z * 2)  # project hidden state to latent std, var
        self.proj_z2latent = nn.Linear(self.d_z, d_model)  # project latent sample to embedding
        self.proj_features_images_2_latent = nn.Linear(2048, d_model)
        self.proj_latent_2_a_hat = nn.Linear(d_model, d_action)

    def forward(self, images, text, curr_pos, actions=None, is_pad=None):

        is_training = actions is not None  # train or val
        bs, num_cam, c, h, w = images.shape

        # 1. Obtain latent z from action sequence
        if is_training:  # encoder of CVAE level, nearly unchanged.
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.proj_action2hidden(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.proj_pos2hidden(curr_pos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(curr_pos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)  # ZJA: 就是Z啦。and 这里的pos_embed是Positional Encoding
            encoder_output = encoder_output[0]  # take cls output only # ZJA 这一行操作有点奇怪
            latent_info = self.proj_latent2z(encoder_output)  # ZJA 这可能就是VAE的A所在了，因为本文的encoder生成的是hidden state z的hidden state，而不是直接生成hidden state z本身的std和var
            mu = latent_info[:, :self.d_z]
            logvar = latent_info[:, self.d_z:]

            z = reparametrize(mu, logvar)  # ZJA：VAE的reparametrize trick
            z = self.proj_z2latent(z)
        else:
            mu = logvar = None
            z = torch.zeros([bs, self.d_z], dtype=torch.float32).to(curr_pos.device)
            z = self.proj_z2latent(z)

        # 2. Get all observations
        #       Get images' features
        images = images.reshape(bs * num_cam, c, h, w)  # backbone requirement
        images = self.backbone_encode_image(images)
        images = images.reshape(bs, num_cam, images.shape[1], images.shape[2], images.shape[3])
        images = images.permute(1, 3, 4, 0, 2).flatten(start_dim=0, end_dim=2)  # [B,N,H,W,dim] to [NHW,B,dim] which is [Num_cam*Hf*Wf,bs,dim]
        images = self.proj_features_images_2_latent(images)
        #       Get text's features
        text = self.backbone_encode_text(text).unsqueeze(0)  # [1,B,dim]
        #       Currten position, embedding
        curr_pos = self.proj_pos2hidden(curr_pos).unsqueeze(0)  # [1,B,dim]
        #       Get latent z, embedding
        z = z.unsqueeze(0)  # [1,bs,dim]
        #       Features all
        features_all = torch.cat([images, text, curr_pos, z], axis=0)

        # 3. Get all PE
        #       Handle PE_images, PE_text, PE_z_robotposition, PE_query#  No need PE_text The output of the clip is kinda features_text
        PE_images = self.PE_table_images.repeat(bs, num_cam, 1, 1).flatten(1, 2).permute(1, 0, 2)  # [HW,dim] to [B,N,HW,dim] to [NHW,B,dim]
        PE_text_currPos_z = self.PE_text_currPos_z.unsqueeze(1).repeat(1, bs, 1)  # [3,dim] to [3,1,dim] to [3,B,dim]
        PE_all = torch.cat([PE_text_currPos_z, PE_images], axis=0)  # concate([3,B,dim], [NHW,B,dim]) to [3+NHW,B,dim]=[254+3,16,512] in default
        PE_query = self.PE_query.unsqueeze(1).repeat(1, bs, 1)

        # 4. Forward to transformer
        latent_a_hat = self.transformer(features_all, PE_all, PE_query)[0]

        # 5. Get action and is_pad
        a_hat = self.proj_latent_2_a_hat(latent_a_hat)
        # Transformer直接预测100个action

        return a_hat, [mu, logvar]


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def build_encoder(config):
    d_model = config['hidden_dim']  # 256
    dropout = config['dropout']  # 0.1
    nhead = config['nheads']  # 8
    dim_feedforward = config['dim_feedforward']  # 2048
    num_encoder_layers = config['enc_layers']  # 4 # TODO shared with VAE decoder
    normalize_before = config['pre_norm']  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_transformer(config):
    return Transformer(
        d_model=config['hidden_dim'],
        dropout=config['dropout'],
        nhead=config['nheads'],
        dim_feedforward=config['dim_feedforward'],
        num_encoder_layers=config['enc_layers'],
        num_decoder_layers=config['dec_layers'],
        normalize_before=config['pre_norm'],
        return_intermediate_dec=True,
    )
