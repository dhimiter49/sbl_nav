from typing import Sequence
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
import torch
import torch.nn as nn


def get_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    layer_norm: bool = True,
    use_bias: bool = True
):
    activation = nn.Tanh()
    affine_layers = nn.ModuleList()

    prev = input_dim
    x = nn.Linear(prev, hidden_sizes[0], bias=use_bias)
    for p in x.parameters():
        if len(p.data.shape) >= 2:
            nn.init.orthogonal_(p.data, gain=2**0.5)
        else:
            p.data.zero_()
    affine_layers.append(x)
    prev = hidden_sizes[0]

    if layer_norm:
        x = nn.LayerNorm(prev)
        affine_layers.extend([x, torch.nn.Tanh()])
    else:
        affine_layers.append(activation)

    for i, l in enumerate(hidden_sizes[1:]):
        x = nn.Linear(prev, l, bias=use_bias)
        for p in x.parameters():
            if len(p.data.shape) >= 2:
                nn.init.orthogonal_(p.data, gain=2**0.5)
            else:
                p.data.zero_()
        affine_layers.extend([x, activation])
        prev = l

    return affine_layers


class CNN1dFE(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        channels: int = 4,
        kernel_size: int = 5,
        non_lidar_dim: int = 4,
        one_cnn: bool = True,
    ) -> None:
        self.obs_dim = observation_space.shape[0]
        self.channels = channels
        self.kernel_size = kernel_size
        self.non_lidar_dim = non_lidar_dim
        self.one_cnn = one_cnn
        # goal x2, vel x2, time from ProDMP 1
        self.lidar_dim = (self.obs_dim - self.non_lidar_dim) // 2
        # add kernel size div 2 to have circular 1d cnn
        # the obs looks like |N-1|N|1|2|...|N-1|N|1|2|
        self.lidar_dim_out = (self.obs_dim - self.non_lidar_dim) // 2 +\
            (self.kernel_size // 2) * 2
        if not self.one_cnn:
            feature_dim = 2 * 2 * (self.lidar_dim_out - self.kernel_size + 1) *\
                self.channels
        else:
            feature_dim = 2 * (self.lidar_dim_out - self.kernel_size + 1) * self.channels
        super().__init__(observation_space, feature_dim)

        if not self.one_cnn:
            self.cnn_1d_lidar = nn.ModuleList()
            self.cnn_1d_lidar.append(nn.Conv1d(1, self.channels, self.kernel_size))
            self.cnn_1d_lidar.append(nn.Tanh())
            self.cnn_1d_lidar_vel = nn.ModuleList()
            self.cnn_1d_lidar_vel.append(nn.Conv1d(1, self.channels, self.kernel_size))
            self.cnn_1d_lidar_vel.append(nn.Tanh())
            self.mlp_no_lidar = get_mlp(
                self.non_lidar_dim,
                [32, 2 * (self.lidar_dim_out - self.kernel_size + 1) * self._channels],
            )
        else:
            self.cnn_1d_lidar = nn.ModuleList()
            self.cnn_1d_lidar.append(nn.Conv1d(2, self.channels, self.kernel_size))
            self.cnn_1d_lidar.append(nn.Tanh())
            self.mlp_no_lidar = get_mlp(
                self.non_lidar_dim,
                [32, (self.lidar_dim_out - self.kernel_size + 1) * self.channels],
            )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        if not self.one_cnn:
            single = len(x.shape) == 1
            if single:
                x = x.unsqueeze(0)
            batch = x.shape[0]
            x_no_lidar = x[:, :self.non_lidar_dim]
            x_lidar = x[:, self.non_lidar_dim:self.non_lidar_dim + self.lidar_dim]
            x_lidar = torch.cat([
                x_lidar[:, -(self.kernel_size // 2):],
                x_lidar,
                x_lidar[:, :self.kernel_size // 2]
            ], dim=-1)
            x_lidar_vel = x[
                :,
                self.non_lidar_dim + self.lidar_dim:
                self.non_lidar_dim + 2 * self.lidar_dim
            ]
            x_lidar_vel = torch.cat([
                x_lidar_vel[:, -(self.kernel_size // 2):],
                x_lidar_vel,
                x_lidar_vel[:, :self.kernel_size // 2]
            ], dim=-1)
            x_lidar = x_lidar.unsqueeze(1)
            x_lidar_vel = x_lidar_vel.unsqueeze(1)
            for affine in self.mlp_no_lidar:
                x_no_lidar = affine(x_no_lidar)
            for cnn_lidar, cnn_lidar_vel in zip(
                self.cnn_1d_lidar, self.cnn_1d_lidar_vel
            ):
                x_lidar = cnn_lidar(x_lidar)
                x_lidar_vel = cnn_lidar_vel(x_lidar_vel)

            x = torch.cat([
                x_lidar.view(batch, -1),
                x_lidar_vel.view(batch, -1),
                x_no_lidar
            ], dim=-1)
            if single:
                x = x.squeeze(0)
        else:
            single = len(x.shape) == 1
            if single:
                x = x.unsqueeze(0)
            batch = x.shape[0]
            x_no_lidar = x[:, :self.non_lidar_dim]
            x_lidar = x[:, self.non_lidar_dim:self.non_lidar_dim + self.lidar_dim]
            x_lidar_vel = x[
                :,
                self.non_lidar_dim + self.lidar_dim:
                self.non_lidar_dim + 2 * self.lidar_dim
            ]
            x_lidar_lidar_vel = torch.stack([x_lidar, x_lidar_vel], dim=1)
            x_lidar_lidar_vel = torch.cat([
                x_lidar_lidar_vel[:, :, -(self.kernel_size // 2):],
                x_lidar_lidar_vel,
                x_lidar_lidar_vel[:, :, :self.kernel_size // 2]
            ], dim=-1)
            for affine in self.mlp_no_lidar:
                x_no_lidar = affine(x_no_lidar)
            for cnn_lidar in self.cnn_1d_lidar:
                x_lidar_lidar_vel = cnn_lidar(x_lidar_lidar_vel)
            x = torch.cat([
                x_lidar_lidar_vel.view(batch, -1), x_no_lidar
            ], dim=-1)

            if single:
                x = x.squeeze(0)
        return x
