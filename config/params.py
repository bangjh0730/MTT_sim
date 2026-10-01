import os
import numpy as np

# ---- Time ----
# Table I says 1 s, the Section V text 0.5 s. dt rescales Q, the dwell, per-slot
# travel and the mission length, so the table should be corrected to match.
DT            = 0.5    # s
T_SLOTS       = 1000   # eval horizon
T_SLOTS_TRAIN = 500

# ---- Scenario ----
# 3 km is the smallest area three UAVs cannot simply blanket. At 2 km the queue
# never forms; at 4 km throughput starves to ~12 rescues per mission.
MAP_SIZE    = 3000.0   # m, square area
NUM_UAVS    = 3
NUM_TARGETS = 10       # targets at t=0; more are born mid-mission

# False caps every set at one target: the "w/o one-to-many" ablation.
ONE_TO_MANY = True

# ---- UAV kinematics ----
H     = 100.0   # m, altitude
V_MAX = 25.0    # m/s [8]

# ---- Target motion ----
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

# ---- ISAC waveform ----
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

# Detection threshold, Table I [9]. Gives a 489 m ground radius: 500 m slant at
# 25 dB, less the 100 m altitude, on the 0.25 s sensing dwell. Three UAVs
# therefore see 2.26 km^2 at once, a quarter of a 3 km map.
SNR_MIN_DB = 25.0
SNR_MIN    = 10 ** (SNR_MIN_DB / 10)

# Sensing/communication split of each slot. Fixed, not a control: the action
# space is (dv_x, dv_y) alone, and the radar's aim is set by the environment.
TAU       = 0.5   # [9]
TAU_SENSE = TAU * DT
TAU_COMM  = (1.0 - TAU) * DT

# ---- Measurement noise at unit SNR ----
# SNR uses the 3D slant range, so it stays bounded when a UAV passes overhead.
SIGMA_R2_0     = 10.0    # m^2   [8]
SIGMA_THETA2_0 = 1e-4    # rad^2 [8]

# ---- Communication ----
B     = 1e6   # Hz
R_MIN = 1e6   # bps

# ---- Target birth ----
# Uniform over the area; targets leave only by being rescued. Capacity is ~0.043
# rescues/slot, and the knee is sharp - below it the queue drains, above it the
# backlog builds:
#
#   p_b     |K|/|U|   cleared    D
#   0.030     1.1       90%     33 s
#   0.065     5.2       58%     34 s
#   0.080    10.4       32%     38 s
#
# 0.065 sits just over the knee, so the load rises through the episode rather
# than holding flat. Measured with a stand-in policy; recheck once trained.
P_BIRTH = 0.065

# ---- Rescue ----
# p_r = LAMBDA_RESCUE / (LAMBDA_RESCUE + tr(Sigma_pos)), so lambda is readable as
# an effective rescue radius R50 where p_r = 0.5 under steady sensing. R50 = 200 m
# against a 489 m detection radius: a UAV sees a target from 2.5x the range at
# which it can save it, which is what makes set composition matter rather than
# raw coverage. p_r is 0.89 at 100 m, 0.50 at 200 m, 0.05 at 400 m.
#
# Both extremes ruin the scenario. At 0.025 one measurement from 200 m already
# gives p_r ~ 0.93, so detection IS rescue. At 1.6e-5 (R50 ~ 30 m) capacity
# collapses and sets above two members stop being serveable.
LAMBDA_RESCUE = 1.762e-3   # m^2, R50 = 200 m

# ---- Reward ----
# See marl.reward: the weight lives with the term it scales.


# ---- Observation ----
# Per-target uncertainty scalar. "log" is log10(tr Sigma) rescaled to [0,1];
# "p_r" is lambda/(lambda+tr Sigma) and saturates, compressing nearly all of
# tr's ~10 orders of magnitude to 0, so the actor cannot tell a member one slot
# stale from one never sensed.
SIGMA_FEATURE = os.environ.get("SIGMA_FEATURE", "log")

# ---- Agentic AI (detector -> judge -> planner) ----
JUDGE_MODEL   = "gemini-3.1-flash-lite"
PLANNER_MODEL = "gemini-3.1-flash"

DETECTOR_TICK   = 40   # slots between periodic (no-event) alarms
REPLAN_COOLDOWN = 5    # floor on the gap between planner invocations
