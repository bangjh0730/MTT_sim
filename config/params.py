import os
import numpy as np

# ---- Time ----
# Section V text. Table I lists 1 s [9]; the paper contradicts itself and the
# table should be corrected. dt rescales Q, the dwell, per-slot travel and the
# mission length, so the two readings are not interchangeable.
DT            = 0.5    # s
T_SLOTS       = 1000   # eval horizon
T_SLOTS_TRAIN = 500

# ---- Scenario ----
MAP_SIZE    = 2000.0   # m, square area
NUM_UAVS    = 3
NUM_TARGETS = 8        # targets at t=0; more are born mid-mission

# False caps every set at one target: the "w/o one-to-many" ablation.
ONE_TO_MANY = True

# ---- UAV kinematics ----
H     = 100.0   # m, altitude
V_MAX = 25.0    # m/s [8]

# ---- Target motion - Eqs. (2)-(4) ----
SIGMA_W2     = 5.0    # (m/s^2)^2, maneuver noise [8]
V_MAX_TARGET = 10.0   # m/s [8]

F_MAT = np.array([
    [1, 0, DT, 0],
    [0, 1, 0,  DT],
    [0, 0, 1,  0],
    [0, 0, 0,  1],
], dtype=float)

Q_MAT = SIGMA_W2 * np.array([
    [DT**4 / 4, 0,         DT**3 / 2, 0        ],
    [0,         DT**4 / 4, 0,         DT**3 / 2],
    [DT**3 / 2, 0,         DT**2,     0        ],
    [0,         DT**3 / 2, 0,         DT**2    ],
], dtype=float)

# ---- ISAC waveform - Eq. (7) ----
PTX    = 1.0     # W [9]
GT_DBI = 20.0    # dBi, UAV Tx [9]
GR_DBI = 30.0    # dBi, UAV Rx [9]
GC_DBI = 0.0     # dBi, BS Rx [9]
GT     = 10 ** (GT_DBI / 10)
GR     = 10 ** (GR_DBI / 10)
GC     = 10 ** (GC_DBI / 10)
LAMBDA = 0.125   # m, wavelength (2.4 GHz)
SIGMA0 = 1.0     # m^2, reference RCS [9]
TAU0   = 1.0     # s, reference dwell [8]
R0     = 500.0   # m, reference range [8]
N0_DBM = -110.0  # dBm [9]
N0     = 10 ** ((N0_DBM - 30) / 10)   # W

SNR0 = (PTX * GT * GR * LAMBDA**2 * SIGMA0 * TAU0
        / ((4 * np.pi)**3 * R0**4 * N0))

# Detection threshold, Table I [9]. Gives a 659 m ground radius of the
# detection footprint (range goes as SNR_min^(-1/4)).
SNR_MIN_DB = 20.0
SNR_MIN    = 10 ** (SNR_MIN_DB / 10)

# Sensing/communication split of each slot. A fixed system parameter, not a
# control: the action space of P2 is (dv_x, dv_y, which member to sense).
TAU       = 0.5   # [9]
TAU_SENSE = TAU * DT
TAU_COMM  = (1.0 - TAU) * DT

# ---- Measurement noise at unit SNR - Eq. (10) ----
# SNR uses the 3D slant range, so it stays bounded when a UAV passes overhead
# and no artificial noise floor is needed.
SIGMA_R2_0     = 10.0    # m^2   [8]
SIGMA_THETA2_0 = 1e-4    # rad^2 [8]

# ---- Communication - Eqs. (11)-(13) ----
B     = 1e6   # Hz
R_MIN = 1e6   # bps

# ---- Target birth - Eq. (5) ----
# Uniform over the area; targets leave only by being rescued. No cap.
# Starting point for tuning the load. With a stand-in policy (fly to the
# nearest member, sense the member with the largest expected p_r gain) fleet
# throughput is ~0.40 / 0.59 / 0.76 rescues per slot at 6 / 12 / 24 live
# targets and falls past ~24, so p_b near 0.5 puts |K|/|U| around 3 with wide
# swings. A better policy or allocator clears faster and lowers the ratio.
P_BIRTH = 0.45

# ---- Rescue - Eq. (6) ----
# p_r = LAMBDA_RESCUE / (LAMBDA_RESCUE + tr(Sigma_pos)).
# Set for a median p_r near 0.2 on the slot a target is sensed. With the 659 m
# footprint targets are often sensed from far away, and SNR goes as r^-4, so tr
# after a measurement varies ~100x across the footprint; a policy that flies
# closer raises the median.
LAMBDA_RESCUE = 0.025   # m^2

# ---- Observation ----
# Per-target uncertainty scalar of Eq. (24). "p_r" is lambda/(lambda+tr Sigma);
# "log" is log10(tr Sigma) rescaled to [0,1]. p_r saturates - tr spans ~10
# orders of magnitude and p_r compresses nearly all of it to 0, using a few
# percent of its own range, so the actor cannot tell a member one slot stale
# from one never sensed.
SIGMA_FEATURE = os.environ.get("SIGMA_FEATURE", "log")

# ---- Agentic AI (detector -> judge -> planner) ----
JUDGE_MODEL   = "gemini-3.1-flash-lite"
PLANNER_MODEL = "gemini-3.1-flash"

DETECTOR_TICK   = 40   # slots between periodic (no-event) alarms
REPLAN_COOLDOWN = 5    # floor on the gap between planner invocations
