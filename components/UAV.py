import numpy as np
from config.params import DT, H, V_MAX, E_MAX, ONE_TO_MANY


class UAV:
    """
    Multi-rotor UAV with velocity-based kinematics.
    - vx, vy: velocity vector in world frame — controlled via dvx, dvy
    """

    def __init__(self, uav_id, init_pos, init_vx=0.0, init_vy=0.0):
        self.id = uav_id
        self.x  = float(init_pos[0])
        self.y  = float(init_pos[1])
        self.vx = float(init_vx)   # world-frame velocity x, m/s
        self.vy = float(init_vy)   # world-frame velocity y, m/s

        # ---- Assignment (one-to-many) ----------------------------------
        # The BS assigns a SET of targets A^t_i, not a single id: with |K| > |U|
        # a UAV is responsible for several targets and cycles its sensing among
        # them. `sensing_target` is the one member it actually measures THIS slot
        # (chosen on the fast timescale by the env, see MTTEnv._pick_sensing_target);
        # the set itself only changes when the BS re-partitions.
        self.assignment_set: set = set()
        self.sensing_target = None
        self.located    = False
        self.tau = 0.0

        # Energy is a tracked cost, not a liveness condition: a UAV never leaves
        # the fleet, so there is no `active` flag. Depletion is discouraged
        # through the reward's energy penalty rather than by removing the UAV.
        self.energy = float(E_MAX)

    # ---- assignment-set helpers ------------------------------------------
    @property
    def load(self) -> int:
        """Number of targets this UAV is responsible for."""
        return len(self.assignment_set)

    def set_assignment(self, targets) -> None:
        """Replace the assignment set. Under the one-to-many ablation
        (ONE_TO_MANY=False) the set is truncated to a single target, which
        reproduces the old single-assignment behaviour."""
        s = {int(k) for k in targets if k is not None}
        if not ONE_TO_MANY and len(s) > 1:
            s = {sorted(s)[0]}
        self.assignment_set = s
        if self.sensing_target not in self.assignment_set:
            self.sensing_target = None

    def drop_target(self, k: int) -> None:
        """Remove a target from the set (rescued, or reassigned elsewhere)."""
        self.assignment_set.discard(int(k))
        if self.sensing_target == k:
            self.sensing_target = None

    @property
    def pos2d(self):
        return np.array([self.x, self.y])

    @property
    def pos3d(self):
        return np.array([self.x, self.y, H])

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.vx**2 + self.vy**2))

    @property
    def state(self):
        """Kinematic state [x, y, vx, vy]."""
        return np.array([self.x, self.y, self.vx, self.vy])

    @property
    def residual_energy(self):
        return self.energy / E_MAX

    def step(self, dvx, dvy, tau, energy_cost):
        """One-slot kinematic update. Position updated with pre-step velocity."""
        dvx = float(dvx)
        dvy = float(dvy)

        # Position update using current velocity
        self.x += self.vx * DT
        self.y += self.vy * DT

        # Velocity update, then clamp to V_MAX
        new_vx = self.vx + dvx
        new_vy = self.vy + dvy
        spd = np.sqrt(new_vx**2 + new_vy**2)
        if spd > V_MAX:
            new_vx *= V_MAX / spd
            new_vy *= V_MAX / spd
        self.vx = float(new_vx)
        self.vy = float(new_vy)

        # UAVs are free to fly off-map; no boundary clamp.

        self.tau = float(tau)

        self.energy = max(self.energy - float(energy_cost), 0.0)
