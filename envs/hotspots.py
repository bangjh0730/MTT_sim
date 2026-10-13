"""Where survivors appear: disaster sites as a Thomas cluster process.

Spatial model from Hayajneh et al., "Performance Analysis of UAV Enabled
Disaster Recovery Networks: A Stochastic Geometric Framework Based on Cluster
Processes", IEEE Access 2018. Sites are a thinned Poisson process - structures
at density SITE_DENSITY, each destroyed with probability P_DESTROY - and
survivors scatter about a site as a Gaussian (Thomas) cluster whose spread is
drawn from the Stienen radius distribution, mean 1/(2*tau*sqrt(lambda)).

Extended in time here; the paper is a static snapshot. Sites persist for the
mission, each holds a Poisson number of survivors that are born one at a time
and run out, and a share of births is uniform background unrelated to any
site. Persistence is what makes the future partly forecastable: a site that has
produced survivors is likely to produce more, until it is exhausted.

Nothing here reaches the observation or the allocator. Sites are latent; an
allocator that wants them has to infer them from the births it has seen.
"""
import numpy as np

from config.params import (MAP_SIZE, V_MAX_TARGET, HOTSPOTS, SITE_DENSITY,
                           P_DESTROY, SITE_TAU, SITE_OCCUPANCY, P_BACKGROUND)

_MIN_SPREAD = 50.0     # m; the Stienen draw occasionally gives a near-point site


class Site:
    def __init__(self, center, spread, occupancy):
        self.center = np.asarray(center, dtype=float)
        self.spread = float(spread)
        self.occupancy = int(occupancy)
        self.remaining = int(occupancy)


class HotspotField:
    """Draws the sites for one mission and places each new survivor.

    Births are placed in order from the field's own stream, so for a given seed
    the same survivor appears in the same place whatever the UAVs did.
    """

    def __init__(self, seed, map_size: float = MAP_SIZE, enabled: bool = None):
        self.rng = np.random.default_rng(seed)
        self.map_size = float(map_size)
        self.enabled = HOTSPOTS if enabled is None else enabled
        self.sites: list = []
        if self.enabled:
            self._draw_sites()

    def _draw_sites(self) -> None:
        r, L = self.rng, self.map_size
        mean = P_DESTROY * SITE_DENSITY * L * L
        n = 0
        while n == 0:                       # condition on at least one site
            n = int(r.poisson(mean))
        lo, hi = 0.1 * L, 0.9 * L
        for _ in range(n):
            center = r.uniform(lo, hi, size=2)
            # Stienen radius: f(s) = 2 pi lambda tau s exp(-pi lambda tau^2 s^2)
            u = 1.0 - r.random()
            spread = np.sqrt(-np.log(u) / (np.pi * SITE_DENSITY * SITE_TAU ** 2))
            occupancy = max(1, int(r.poisson(SITE_OCCUPANCY)))
            self.sites.append(Site(center, max(spread, _MIN_SPREAD), occupancy))

    def sample(self):
        """(position, velocity, site index or -1 for background)."""
        r, L = self.rng, self.map_size
        live = [j for j, s in enumerate(self.sites) if s.remaining > 0]

        if not self.enabled or not live or r.random() < P_BACKGROUND:
            pos, site = r.uniform(0.05 * L, 0.95 * L, size=2), -1
        else:
            w = np.array([self.sites[j].remaining for j in live], dtype=float)
            site = live[int(r.choice(len(live), p=w / w.sum()))]
            s = self.sites[site]
            pos = np.clip(s.center + r.normal(0.0, s.spread, size=2),
                          0.05 * L, 0.95 * L)
            s.remaining -= 1

        spd = r.uniform(0.0, V_MAX_TARGET)
        ang = r.uniform(0.0, 2.0 * np.pi)
        return pos, spd * np.array([np.cos(ang), np.sin(ang)]), site

    def summary(self) -> list:
        return [dict(center=s.center.round(0).tolist(), spread=round(s.spread),
                     occupancy=s.occupancy, remaining=s.remaining)
                for s in self.sites]
