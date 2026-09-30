import numpy as np
from config.params import DT, H, V_MAX, ONE_TO_MANY


class UAV:
    """Multi-rotor UAV, velocity-controlled.

    Holds a SET of targets and senses one per slot; `sensing_target` is the one
    measured this slot, set by the environment rather than by the policy. No
    energy state and no liveness flag: the fleet is fixed and every UAV flies
    the whole mission.
    """

    def __init__(self, uav_id, init_pos, init_vx=0.0, init_vy=0.0):
        self.id = uav_id
        self.x  = float(init_pos[0])
        self.y  = float(init_pos[1])
        self.vx = float(init_vx)
        self.vy = float(init_vy)

        self.assignment_set: set = set()
        self.sensing_target = None

    @property
    def load(self) -> int:
        return len(self.assignment_set)

    def set_assignment(self, targets) -> None:
        """Replace the set. Truncated to one target when ONE_TO_MANY is False."""
        s = {int(k) for k in targets if k is not None}
        if not ONE_TO_MANY and len(s) > 1:
            s = {sorted(s)[0]}
        self.assignment_set = s
        if self.sensing_target not in self.assignment_set:
            self.sensing_target = None

    def drop_target(self, k: int) -> None:
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
        return np.array([self.x, self.y, self.vx, self.vy])

    def step(self, dvx, dvy, map_size: float = None):
        """Position advances with the pre-step velocity, then velocity updates
        and is clamped to V_MAX.

        With `map_size` given the UAV is confined to the area: position clamped
        to [0, map_size] and the outward velocity component zeroed, so it stops
        against the edge rather than sliding along it. An escaped UAV also breaks
        its own observation, where position enters as x / MAP_SIZE and grows
        without bound once outside.
        """
        self.x += self.vx * DT
        self.y += self.vy * DT

        nvx, nvy = self.vx + float(dvx), self.vy + float(dvy)
        spd = np.sqrt(nvx**2 + nvy**2)
        if spd > V_MAX:
            nvx *= V_MAX / spd
            nvy *= V_MAX / spd
        self.vx, self.vy = float(nvx), float(nvy)

        if map_size is not None:
            if self.x <= 0.0:
                self.x, self.vx = 0.0, max(self.vx, 0.0)
            elif self.x >= map_size:
                self.x, self.vx = map_size, min(self.vx, 0.0)
            if self.y <= 0.0:
                self.y, self.vy = 0.0, max(self.vy, 0.0)
            elif self.y >= map_size:
                self.y, self.vy = map_size, min(self.vy, 0.0)
