import numpy as np
from config.params import DT, H, V_MAX, E_MAX


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

        self.assignment = None
        self.located    = False
        self.tau = 0.0

        self.energy = float(E_MAX)
        self.active = True

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

        self.energy -= float(energy_cost)
        if self.energy <= 0.0:
            self.energy = 0.0
            self.active = False
