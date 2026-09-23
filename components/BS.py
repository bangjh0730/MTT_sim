import numpy as np


class BS:
    """Base station: holds the EKF estimates and the assignment table.

    The table is one-to-MANY - each UAV maps to a SET, since with |K| > |U| a
    single-id table cannot express the mission.
    """

    def __init__(self, pos2d):
        self.pos = np.array(pos2d, dtype=float)
        self.assignments = {}   # {uav_id: set(target_id)}
        self.estimates   = {}   # {target_id: (mu, Sigma)}

    def add_target(self, k, mu0, Sigma0):
        self.estimates[k] = (np.array(mu0, dtype=float), np.array(Sigma0, dtype=float))

    def remove_target(self, k):
        self.estimates.pop(k, None)
        for s in self.assignments.values():
            s.discard(k)

    def set_assignments(self, table: dict) -> None:
        self.assignments = {i: {int(k) for k in ks} for i, ks in table.items()}
