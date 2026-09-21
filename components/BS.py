import numpy as np


class BS:
    def __init__(self, pos2d):
        self.pos = np.array(pos2d, dtype=float)

        self.assignments = {}  # {uav_id: target_id}
        self.estimates   = {}  # {target_id: (mu, Sigma)}  -- updated by envs/estimation

    def add_target(self, k, mu0, Sigma0):
        self.estimates[k] = (np.array(mu0, dtype=float), np.array(Sigma0, dtype=float))

    def remove_target(self, k):
        self.estimates.pop(k, None)
