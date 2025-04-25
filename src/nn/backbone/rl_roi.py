import torch
import torch.nn as nn
from ray.rllib.policy import Policy

class RLROIPolicy(Policy):
    """
    RL agent for dynamic ROI prioritization via policy gradients.
    """
    def __init__(self, obs_space, act_space, config=None):
        super().__init__(obs_space, act_space, config)
        self.net = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, act_space.n)
        )

    def compute_actions(self, obs_batch, **kwargs):
        logits = self.net(torch.tensor(obs_batch, dtype=torch.float32))
        return logits.argmax(dim=1).numpy(), [], {}

    def learn_on_batch(self, samples):
        # integrate RLlib training logic here
        return {}
