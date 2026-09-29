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
# 3 km is the smallest area the fleet cannot simply blanket. Measured with the
# k-means allocator and a pursuit stand-in policy: at 2 km the queue never forms
# (|K| settles at 0.4, everything cleared); at 4 km throughput starves to 10-14
# rescues per mission and nothing discriminates between allocators.
MAP_SIZE    = 3000.0   # m, square area
NUM_UAVS    = 3
NUM_TARGETS = 10       # targets at t=0; more are born mid-mission

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

# Detection threshold, Table I [9]. Gives a 489 m GROUND radius of the
# detection footprint: slant range 500 m at 25 dB, less the H = 100 m altitude,
# with the 0.25 s sensing dwell (TAU * DT), not a 1 s one. Range goes as
# SNR_min^(-1/4), so the threshold is a weak lever on coverage.
# Footprint 0.75 km^2, so three UAVs see 2.26 km^2 at once - 56% of a 2 km map
# but only 25% of a 3 km one, which is why 2 km is too small to be interesting.
SNR_MIN_DB = 25.0
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
# Fleet capacity at 3 km is ~0.043 rescues/slot with the argmax-tr radar rule
# and a stand-in policy that commits to its nearest member. The knee is sharp:
# below it the queue drains to |K|/|U| ~ 1, above it the backlog builds.
#
#   p_b     |K|/|U|   cleared    D
#   0.030     1.1       90%     33 s
#   0.050     2.6       79%     30 s
#   0.065     5.2       58%     34 s
#   0.070     6.1       50%     36 s
#   0.080    10.4       32%     38 s
#
# |K|/|U| cannot be BOTH held constant and left near 58% cleared: a flat ratio
# needs p_b = capacity, which clears ~90%. So the load deliberately rises
# through the episode - a regime shift inside one mission, which is the setting
# the allocator has to adapt to.
#
# These figures come from a stand-in policy, so re-check them once the MARL
# policy is trained; a better policy raises capacity and lowers the ratio.
P_BIRTH = 0.065

# ---- Rescue - Eq. (6) ----
# p_r = LAMBDA_RESCUE / (LAMBDA_RESCUE + tr(Sigma_pos)).
#
# Read it as an effective rescue radius R50, the range at which p_r = 0.5 under
# steady sensing: lambda = tr_ss(R50), where tr_ss is the converged trace at
# that range. This is the parameter that sets the problem's whole character,
# because tr_ss climbs steeply with range (radar SNR goes as r^-4, compounded
# through the EKF):
#
#   R50    lambda      p_r at R50/2   p_r at 2*R50
#   100 m  2.23e-04       0.63           0.11
#   200 m  1.76e-03       0.89           0.05
#   300 m  9.37e-03       0.94           0.13
#   400 m  3.54e-02       0.95           0.36
#
# 1.762e-3 puts R50 at 200 m against a 489 m detection radius, so a UAV sees a
# target from about 2.5x the range at which it can save it: p_r is 0.89 at
# 100 m, 0.50 at 200 m, 0.16 at 300 m and 0.05 at 400 m. Detecting a target and
# rescuing it stay separate events, which is what makes WHICH targets share a
# set matter rather than only how much area is covered.
#
# Both extremes ruin the scenario. At 0.025 one measurement from 200 m gives
# sigma ~ 4 cm and p_r ~ 0.93, so detection IS rescue and set composition stops
# mattering. At 1.6e-5 (R50 ~ 30 m) rescue needs sigma ~ 1.2 cm, capacity
# collapses and sets above two members stop being serveable at all.
LAMBDA_RESCUE = 1.762e-3   # m^2, R50 = 200 m

# ---- Reward - Eqs. (19), (25) ----
# r_u = -|A_u| + W_SHAPE * sum_{k in A_u} [p_r,k(t) - p_r,k(t-1)], a rescued
# member scoring p_r = 1. The first term is the objective; the second is
# potential-based shaping, so W_SHAPE trades gradient density against variance
# without moving the optimum. Set so a typical per-slot shaping term is
# comparable to the per-slot delay cost |A_u| ~ 3-5.
#
# Measured at R50 = 200 m, the shaping term is zero on 90% of slots and fires
# almost only when a rescue lands (mean +0.84 there): p_r is 0.05 at 400 m and
# 0.02 at 300 m, so Phi = sum p_r is flat across the whole approach and shapes
# nothing until the last ~150 m. At W_SHAPE = 10 a rescue is worth about two
# slots of delay cost for a 4-5 member set. If the approach itself needs
# shaping, Phi = -sum sqrt(tr Sigma_k) is linear in metres and does not
# saturate; it is equally policy-invariant.
W_SHAPE = 10.0

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
