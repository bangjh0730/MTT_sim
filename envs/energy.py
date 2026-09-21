import numpy as np
from config.params import P0, P1, U_TIP, V0, D0, RHO_AIR, S0, NU, PTX, DT


def propulsion_power(v):
    t1 = P0 * (1.0 + 3.0 * v**2 / U_TIP**2)
    t2 = P1 * np.sqrt(np.sqrt(1.0 + v**4 / (4.0 * V0**4)) - v**2 / (2.0 * V0**2))
    t3 = 0.5 * D0 * RHO_AIR * S0 * NU * v**3
    return t1 + t2 + t3


def slot_energy(v):
    """Total energy consumed in one slot: propulsion + ISAC transmission."""
    return (propulsion_power(v) + PTX) * DT
