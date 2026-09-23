import torch
import torch.nn as nn

ACT_DIM = 2  # (dvx, dvy) — squashed to [-1, 1] then scaled to +/- V_MAX

# Bounds on the pre-tanh log std. The clamp is on log_std itself and the
# initial value sits strictly inside it: clamping exp(log_std) at 1.0 with
# log_std initialised at 0 put the parameter on the boundary, the entropy bonus
# nudged it just past it, and the zero gradient there froze std at 1 for the
# whole run, so the policy could never become precise.
LOG_STD_INIT = -0.5
LOG_STD_MIN  = -4.0
LOG_STD_MAX  = 0.5

_LOG_SQRT_2PI = 0.9189385332046727
_NEG_INF      = -1e9


def _mlp(d_in: int, d_out: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, d_out), nn.ReLU())


def _pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean and max over the set dim: [B, n, d] -> [B, 2d].
    An empty set pools to zeros."""
    m     = mask.unsqueeze(-1).float()
    count = m.sum(1).clamp(min=1.0)
    mean  = (h * m).sum(1) / count
    mx    = h.masked_fill(~mask.unsqueeze(-1), _NEG_INF).max(1).values
    mx    = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class MAPPOActor(nn.Module):
    """
    Decentralized actor for one UAV, shared by all (parameter sharing).

    Its action is (dvx, dvy, c): a velocity increment and which member of its
    assignment set to sense this slot. The set is read as a set - a shared
    per-member encoder, pooled - so any |A_i| is accepted and the sensing head
    scores each member in the context of the whole set. Nothing about which
    member to sense is hard-coded: it is learned from the reward.
    """
    def __init__(self, ego_dim: int, mem_dim: int, hidden: int = 128):
        super().__init__()
        self.mem_enc = _mlp(mem_dim + ego_dim, hidden)
        self.ctx     = nn.Sequential(
            nn.Linear(ego_dim + 2 * hidden, 256), nn.ReLU(),
            nn.Linear(256, hidden), nn.ReLU(),
        )
        self.vel_head   = nn.Linear(hidden, ACT_DIM)
        self.sense_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), LOG_STD_INIT))

    def forward(self, ego, mem, mask):
        """ego [B, E], mem [B, n, M], mask [B, n] ->
        (velocity mean [B, 2], log_std [2], sensing logits [B, n])."""
        n     = mem.shape[1]
        h     = self.mem_enc(torch.cat([mem, ego.unsqueeze(1).expand(-1, n, -1)], -1))
        c     = self.ctx(torch.cat([ego, _pool(h, mask)], -1))
        mean  = self.vel_head(c)
        logit = self.sense_head(torch.cat([h, c.unsqueeze(1).expand(-1, n, -1)], -1)).squeeze(-1)
        logit = logit.masked_fill(~mask, _NEG_INF)
        return mean, self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX), logit

    def get_action(self, ego, mem, mask, deterministic: bool = False):
        """Returns (tanh velocity [B, 2], pre-tanh velocity [B, 2],
        sensing index [B] (-1 for an empty set), log-prob [B]).

        The pre-tanh sample is what gets stored and re-evaluated: recovering it
        with atanh from a float32 tanh output is lossy once |raw| > ~5.
        """
        mean, log_std, logit = self.forward(ego, mem, mask)
        has = mask.any(1)
        if deterministic:
            idx = torch.where(has, logit.argmax(1), torch.full_like(has, -1, dtype=torch.long))
            return mean.tanh(), mean, idx, torch.zeros(ego.shape[0], device=ego.device)
        noise = torch.randn_like(mean)
        raw   = mean + log_std.exp() * noise
        lp    = (-0.5 * noise.pow(2) - log_std - _LOG_SQRT_2PI).sum(-1)
        dist  = torch.distributions.Categorical(logits=logit)
        idx   = dist.sample()
        lp    = lp + torch.where(has, dist.log_prob(idx), torch.zeros_like(lp))
        idx   = torch.where(has, idx, torch.full_like(idx, -1))
        return raw.tanh(), raw, idx, lp

    def evaluate(self, ego, mem, mask, raw, idx):
        """Log-prob and entropy of stored (pre-tanh velocity, sensing index)."""
        mean, log_std, logit = self.forward(ego, mem, mask)
        has   = mask.any(1)
        noise = (raw - mean) / log_std.exp()
        lp    = (-0.5 * noise.pow(2) - log_std - _LOG_SQRT_2PI).sum(-1)
        ent   = (log_std + 0.5 + _LOG_SQRT_2PI).sum(-1).expand(ego.shape[0])
        dist  = torch.distributions.Categorical(logits=logit)
        safe  = idx.clamp(min=0)
        lp    = lp + torch.where(has, dist.log_prob(safe), torch.zeros_like(lp))
        ent   = ent + torch.where(has, dist.entropy(), torch.zeros_like(lp))
        return lp, ent


class CentralizedCritic(nn.Module):
    """
    CTDE critic V_i(S^t), from agent i's viewpoint: its own features plus the
    UAV set and the target set, each encoded per element and pooled, so the
    number of live targets is unbounded. Each UAV earns its own reward, so each
    gets its own baseline. Training only.
    """
    def __init__(self, ego_dim: int, u_dim: int, t_dim: int, hidden: int = 128):
        super().__init__()
        self.u_enc = _mlp(u_dim, hidden)
        self.t_enc = _mlp(t_dim, hidden)
        self.head  = nn.Sequential(
            nn.Linear(ego_dim + 4 * hidden, 256), nn.ReLU(),
            nn.Linear(256, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ego, uavs, u_mask, tgts, t_mask) -> torch.Tensor:
        pu = _pool(self.u_enc(uavs), u_mask)
        pt = _pool(self.t_enc(tgts), t_mask)
        return self.head(torch.cat([ego, pu, pt], -1)).squeeze(-1)
