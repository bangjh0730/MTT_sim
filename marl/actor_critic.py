import torch
import torch.nn as nn

ACT_DIM = 2  # (dvx, dvy) — squashed to [-1, 1] then scaled to +/- V_MAX


class MAPPOActor(nn.Module):
    """
    Decentralized actor: maps local obs -> (dvx, dvy) for one UAV.
    All agents share this network (parameter sharing).

    The action space is the velocity increment only. The ISAC sensing/comm split
    is a fixed system parameter, not a control, and which set member to sense is
    decided by the environment's scheduler -- so flying is the whole of the
    policy's job.
    """
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden),  nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, ACT_DIM)
        self.log_std   = nn.Parameter(torch.zeros(ACT_DIM))  # learnable per-dim

    def forward(self, obs: torch.Tensor):
        x    = self.backbone(obs)
        mean = self.mean_head(x)                      # unbounded; tanh applied at sample time
        std  = self.log_std.exp().clamp(1e-3, 1.0)
        return mean, std

    @torch.jit.export
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and compute log-prob.
        Supports batched input (B, obs_dim).
        Marked @export so torch.jit.script includes it alongside forward().
        """
        mean, std = self.forward(obs)
        if deterministic:
            return mean.tanh(), torch.zeros(obs.shape[0], device=obs.device)
        noise   = torch.randn_like(mean)
        action  = (mean + std * noise).tanh()
        logprob = (-0.5 * noise.pow(2) - std.log() - 0.9189385332046727).sum(-1)
        return action, logprob

    @torch.jit.export
    def evaluate(self, obs: torch.Tensor, action_tanh: torch.Tensor):
        """Re-evaluate log-prob and entropy for stored actions."""
        mean, std = self.forward(obs)
        raw     = torch.atanh(action_tanh.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        noise   = (raw - mean) / std
        logprob = (-0.5 * noise.pow(2) - std.log() - 0.9189385332046727).sum(-1)
        entropy = (std.log() + 1.4189385332046727).sum(-1)
        return logprob, entropy


class CentralizedCritic(nn.Module):
    """
    CTDE critic: maps concatenated global obs (all agents) → V(s).
    Used only during training; not needed at execution time.
    """
    def __init__(self, global_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden),          nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, g_obs: torch.Tensor) -> torch.Tensor:
        return self.net(g_obs).squeeze(-1)
