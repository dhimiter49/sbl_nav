from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
import torch
import torch.nn as nn
from torch_geometric.nn import MLP


class TransformerFE(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.Space, input_dim: int = 2, feature_dim: int = 16
    ) -> None:
        super().__init__(observation_space, feature_dim)
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, feature_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=2,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.fc_out = MLP([
            self._observation_space.shape[0] // input_dim * feature_dim, feature_dim
        ], norm=None, act='tanh')


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        h = self.embedding(observations.view(batch_size, -1, self.input_dim))
        h = self.transformer_encoder(h)
        return self.fc_out(h.view(batch_size, -1))
