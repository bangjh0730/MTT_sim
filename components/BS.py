import numpy as np


class BS:
    """
    Base station. Holds the target state estimates it maintains with the EKF and
    the assignment table it issues to the UAVs.

    The assignment table is one-to-MANY: each UAV maps to a SET of target ids.
    With |K| > |U| a single-id table cannot express the mission at all — every
    target beyond the U-th would be unassigned by construction.
    """

    def __init__(self, pos2d):
        self.pos = np.array(pos2d, dtype=float)

        self.assignments = {}  # {uav_id: set(target_id)}
        self.estimates   = {}  # {target_id: (mu, Sigma)}  -- updated by envs/estimation

    def add_target(self, k, mu0, Sigma0):
        self.estimates[k] = (np.array(mu0, dtype=float), np.array(Sigma0, dtype=float))

    def remove_target(self, k):
        self.estimates.pop(k, None)
        for s in self.assignments.values():
            s.discard(k)

    def set_assignments(self, table: dict) -> None:
        """table: {uav_id: iterable of target ids}."""
        self.assignments = {i: {int(k) for k in ks} for i, ks in table.items()}
