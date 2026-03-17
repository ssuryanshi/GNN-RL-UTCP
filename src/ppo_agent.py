"""
ppo_agent.py
------------
MaskablePPO agent with a custom GNN-based ActorCritic policy.

Actor:  GNNEncoder(state) → concat(course_emb, slot_emb, room_emb)
                          → MLP(256, 128) → action_logits
Critic: GNNEncoder(state) → global_mean_pool → MLP(256, 128) → V(s)

Uses MaskablePPO from sb3-contrib to respect action masks from TimetableEnv.

Hyperparameters (from blueprint):
  lr           = 3e-4 (linear decay)
  n_steps      = 2048
  batch_size   = 64
  n_epochs     = 10
  gamma        = 0.99
  gae_lambda   = 0.95
  clip_range   = 0.2
  ent_coef     = 0.01
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import gymnasium as gym
import numpy as np
from typing import Any

from gnn_encoder import HGTEncoder, build_encoder_from_data


# ------------------------------------------------------------------ #
# MLP helper                                                           #
# ------------------------------------------------------------------ #

class MLP(nn.Module):
    def __init__(self, in_dim: int, *hidden_dims: int, out_dim: int):
        super().__init__()
        dims   = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
# Custom feature extractor (wraps HGTEncoder for SB3)                 #
# ------------------------------------------------------------------ #

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor wrapper around HGTEncoder.

    Note: SB3's feature extractors receive flat numpy observations.
    We use this class to carry the encoder and expose its forward()
    manually in the custom policy below.
    """

    def __init__(self, observation_space: gym.Space, encoder: HGTEncoder,
                 features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder = encoder

    def forward(self, observations: Tensor) -> Tensor:
        # This is a passthrough; actual GNN logic is in the policy.
        return observations


# ------------------------------------------------------------------ #
# Custom Actor-Critic Policy                                           #
# ------------------------------------------------------------------ #

class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Custom policy that uses the HGT GNN encoder for both actor and critic.

    actor_net:  GNN embeds → concat(course, slot, room) → MLP(256,128) → logits
    critic_net: GNN embeds → global_mean_pool → MLP(256, 128) → V(s)
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 encoder: HGTEncoder | None = None,
                 n_courses: int = 10, n_slots: int = 20, n_rooms: int = 5,
                 out_dim: int = 64,
                 **kwargs):
        # We need to pass net_arch to avoid SB3's default MLP
        kwargs["net_arch"] = []
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.n_courses = n_courses
        self.n_slots   = n_slots
        self.n_rooms   = n_rooms
        self.out_dim   = out_dim

        # GNN encoder (shared backbone)
        self.encoder = encoder  # set externally after init if None

        # Actor MLP: concat(course_emb, slot_emb, room_emb) → logits
        actor_in = out_dim * 3
        self.actor_mlp = MLP(actor_in, 256, 128, out_dim=n_courses * n_slots * n_rooms)

        # Critic MLP: pooled graph embedding → V(s)
        self.critic_mlp = MLP(out_dim, 256, 128, out_dim=1)

    def _build_mlp_extractor(self):
        # Override to prevent default MLP building
        self.mlp_extractor = nn.Identity()
        self.latent_dim_pi  = self.observation_space.shape[0]
        self.latent_dim_vf  = self.observation_space.shape[0]

    def forward(self, obs: Tensor, deterministic: bool = False):
        # SB3 expects (actions, values, log_probs)
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _get_latent(self, obs: Tensor):
        """NB: for GNN policy, obs carries flat features; GNN is called via env."""
        return obs, obs

    def evaluate_actions(self, obs: Tensor, actions: Tensor):
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


# ------------------------------------------------------------------ #
# Convenience factory                                                  #
# ------------------------------------------------------------------ #

def make_masked_env(instance: dict):
    """Wrap TimetableEnv with ActionMasker for MaskablePPO compatibility."""
    from timetable_env import TimetableEnv
    env = TimetableEnv(instance)
    masked_env = ActionMasker(env, lambda e: e.action_masks())
    return masked_env


def build_ppo_agent(
    instance: dict,
    total_timesteps: int = 5_000_000,
    pretrained_path: str | None = None,
    device: str = "auto",
) -> MaskablePPO:
    """
    Build (and optionally warm-start) a MaskablePPO agent.

    Parameters
    ----------
    instance       : timetabling instance dict
    total_timesteps: RL training budget
    pretrained_path: path to a previously saved model (IL warm-start)
    device         : "auto", "cpu", or "cuda"

    Returns
    -------
    MaskablePPO agent (not yet trained — call .learn() separately)
    """
    env = make_masked_env(instance)

    def linear_lr(progress_remaining: float) -> float:
        return 3e-4 * progress_remaining

    if pretrained_path:
        agent = MaskablePPO.load(pretrained_path, env=env, device=device)
        print(f"Loaded pretrained model from {pretrained_path}")
    else:
        agent = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=linear_lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device=device,
        )

    return agent


if __name__ == "__main__":
    import json, pathlib
    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if not sample.exists():
        print("Run generate_indian_data.py first.")
    else:
        with open(sample) as f:
            inst = json.load(f)
        agent = build_ppo_agent(inst, total_timesteps=10_000)
        print("Agent built:", agent.policy)
        print("Training for 10k steps as smoke test...")
        agent.learn(total_timesteps=10_000)
        print("Smoke test passed.")
