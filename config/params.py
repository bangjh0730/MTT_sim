import numpy as np

# ---- Time ----
DT      = 0.5   # s, time slot duration
T_SLOTS       = 1000   # total simulation slots (eval)
T_SLOTS_TRAIN = 500   # shorter horizon for training

# ---- Scenario ----
MAP_SIZE    = 6000.0  # m, square simulation area
NUM_UAVS    = 7
NUM_TARGETS = 5

# ---- UAV kinematics ----
H     = 100.0              # m, fixed altitude
V_MAX = 40.0    # m/s
D_MIN = 100.0   # m, collision avoidance

# ---- UAV propulsion ----
P0      = 79.9    # W, blade profile power
P1      = 88.6    # W, induced power
U_TIP   = 120.0   # m/s, rotor tip speed
V0      = 4.03    # m/s, mean rotor induced velocity
D0      = 0.6     # fuselage drag ratio
RHO_AIR = 1.225   # kg/m³
S0      = 0.05    # rotor solidity
NU      = 0.503   # m², rotor disk area
E_MAX   = 75 * 3600  # J (~30 min)

# ---- ISAC waveform ----
PTX    = 1.0     # W, transmit power
GT_DBI = 20.0    # dBi
GR_DBI = 30.0    # dBi
GT     = 10 ** (GT_DBI / 10)   # linear
GR     = 10 ** (GR_DBI / 10)   # linear
LAMBDA = 0.15    # m
FC     = 2e9     # Hz
SIGMA0 = 1.0     # m², reference RCS
TAU0   = 1.0     # s, reference dwell time
R0     = 500.0   # m, reference range
N0_DBM = -110.0  # dBm
N0     = 10 ** ((N0_DBM - 30) / 10)  # W

SNR0 = PTX * GT * GR * LAMBDA**2 * SIGMA0 * TAU0 / ((4 * np.pi)**3 * R0**4 * N0)

SNR_MIN_DB = 20.0
SNR_MIN    = 10 ** (SNR_MIN_DB / 10)   # linear

# ---- Communication ----
B        = 1e6     # Hz
R_MIN    = 1e6     # bps
ETA_LOS  = 1.6     # dB
ETA_NLOS = 23.0    # dB
C1       = 11.95
C2       = 0.14

# ---- Target motion ----
SIGMA_W2     = 5.0   # (m/s²)²
V_MAX_TARGET = 10.0  # m/s

# Constant-velocity transition matrix F
F_MAT = np.array([
    [1, 0, DT, 0],
    [0, 1, 0,  DT],
    [0, 0, 1,  0],
    [0, 0, 0,  1],
], dtype=float)

# Process noise covariance Q
Q_MAT = SIGMA_W2 * np.array([
    [DT**4 / 4, 0,          DT**3 / 2, 0         ],
    [0,          DT**4 / 4, 0,          DT**3 / 2],
    [DT**3 / 2, 0,          DT**2,      0         ],
    [0,          DT**3 / 2, 0,          DT**2     ],
], dtype=float)

# ---- Measurement noise at unit SNR ----
# Radar SNR is computed from the 3D slant range (horizontal distance + altitude
# H), so it is naturally bounded above even when a UAV passes directly over its
# target — no artificial measurement-noise floor is needed to keep R = σ0²/SNR
# (and the resulting Fisher information) from diverging.
SIGMA_R2_0     = 10.0    # m²
SIGMA_THETA2_0 = 1e-4    # rad²

# ---- UAV Failure Model (evaluation only) ----
BETA_FAIL      = 0.0008   # per-UAV failure rate per slot
D_MAX_FAIL     = 2       # max UAVs a single shock can eliminate per target group
N_FAIL_DETECT  = 5       # consecutive silent slots before BS declares a UAV lost

# ---- Target Birth/Death Model (evaluation only) ----
# Targets may appear and disappear mid-mission. Each slot a new target is born
# with probability P_BIRTH, and each present target survives with probability
# P_SURVIVE (dies with 1 - P_SURVIVE). Target ids stay within [0, MAX_TARGETS-1]
# (births fill freed slots) so the fixed-size assignment / logging structures are
# unaffected. Not used in training.
P_BIRTH     = 0.005          # prob a new target appears each slot (~1 per 125 slots)
P_SURVIVE   = 0.9995          # prob an existing target survives each slot (~330-slot mean life)
MAX_TARGETS = NUM_UAVS       # |K^t| <= |U^t|: targets may reach one-per-active-UAV
