import numpy as np

# ---- Time ----
DT      = 0.5   # s, time slot duration
T_SLOTS       = 1000   # total simulation slots (eval)
T_SLOTS_TRAIN = 500   # shorter horizon for training

# ---- Scenario ----
# Overloaded regime: fewer UAVs than targets (|K| > |U| is the NORMAL state, not a
# failure). A UAV therefore holds a SET of targets and cycles its sensing among
# them; full simultaneous coverage is impossible by construction.
MAP_SIZE    = 2000.0  # m, square simulation area (2 x 2 km post-disaster area)
NUM_UAVS    = 3
NUM_TARGETS = 8       # initial targets; more are born mid-mission

# One-to-many assignment switch. True: a UAV holds an arbitrary set of targets.
# False: sets are capped at one member — the "w/o one-to-many" ablation, which
# reproduces the old single-target assignment path.
ONE_TO_MANY = True

# ---- UAV kinematics ----
H     = 100.0   # m, fixed altitude
V_MAX = 25.0    # m/s
# Collision-avoidance separation. 100 m was sized for the old 6 km map; on a 2 km
# map with 3 UAVs cycling large sets it would forbid most useful geometries, so it
# is retuned to the paper's 20 m.
D_MIN = 20.0    # m

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

# ---- Rescue Model ----
# A target is rescued (and removed) with probability p_r = LAMBDA_RESCUE /
# (LAMBDA_RESCUE + tr(Sigma_pos)), evaluated every slot on the BS's own position
# covariance. The model is CONTINUOUS — no "is it tracked" threshold is needed:
# a well-sensed target sits at small tr(Sigma) and is rescued quickly, while an
# unsensed target's tr(Sigma) grows without bound and drives p_r toward 0.
#
# LAMBDA_RESCUE sets the LOAD REGIME and is the main sweep axis of the experiment:
# small lambda keeps the system in persistent heavy overload; large lambda lets it
# repeatedly drain to a manageable state.
#
# CALIBRATION. Both numbers below were measured on this simulator, not guessed:
# the radar is accurate enough at these ranges that one delivered measurement
# drops tr(Sigma_pos) to a median of ~0.075 m^2 (5th-95th pct: 7e-5 .. 0.70,
# driven by the r^-4 SNR law), while a target nobody senses climbs past 1e4 m^2
# within a few slots. p_r is therefore effectively lambda/(lambda + tr) for the
# ONE target each UAV senses per slot, and ~0 for every other — so the fleet's
# rescue throughput is capped by set-cycling rate, not by geometry. That is the
# rate-vs-breadth tradeoff the allocator exists to reason about.
#
# lambda = 0.005 puts a sensed target's median dwell near 10 slots. Measured over
# a 1000-slot run against a naive least-loaded placement, that yields ~50 rescues
# (one every ~20 slots), a mean backlog of ~4.5 with |K| > |U| in ~70% of slots,
# and occasional drains to a manageable state — i.e. the system spends most of its
# time overloaded but visits BOTH regimes, which is what makes the judge's
# regime-adaptive reasoning observable rather than hypothetical.
#
# The knife edges on either side are sharp, which is why this is calibrated rather
# than assumed. lambda >= 0.03 makes rescue near-certain the moment a target is
# sensed: the backlog drains to ~2 and the problem collapses to "fly at the
# nearest target". lambda <= 0.002 tips the system into runaway saturation — sets
# grow, each member is sensed more rarely, tr(Sigma) climbs, p_r falls further,
# and the backlog pins at the id ceiling. Sweep lambda across that span to
# demonstrate regime-adaptivity; do not wander outside it without re-measuring.
LAMBDA_RESCUE = 0.005  # m^2

# Rescue is policy-dependent BY DESIGN (better tracking -> faster rescue), so
# unlike births it can never be pre-scheduled. It is drawn from a per-target-id
# RNG stream so a target's own draws depend only on its own lifetime.

# ---- UAV liveness ----
# There is no UAV failure model. The fleet is fixed at NUM_UAVS and every UAV is
# active for the whole mission, so there is no shock-failure rate, no BS
# silence-detection delay, and no path by which a UAV leaves the fleet. Battery
# energy is still tracked and penalised in the reward (see marl/reward.py), but
# depleting it no longer removes a UAV -- with only 3 UAVs carrying the entire
# backlog, losing one would not test the allocator, it would just end the run.

# ---- Target Birth/Death Model (evaluation only) ----
# Targets appear mid-mission with probability P_BIRTH per slot and leave ONLY by
# being rescued (see the rescue model above) — there is no scheduled random death
# any more. Births stay pre-scheduled so they are identical across modes for the
# same seed; rescue cannot be, since it is the metric under test. Target ids stay
# within [0, MAX_TARGETS-1] (births fill freed slots) so the fixed-size logging
# structures are unaffected. Not used in training.
# Birth rate. The paper's table lists 0.01, but that value predates the rescue
# model, and it was calibrated here against the greedy baseline rather than
# assumed. What the sweep shows (mean over 4 seeds, 1000 slots):
#
#   p_b    mean |K|   frac slots |K|>|U|   rescued/born   D-bar
#   0.05     2.1            0.21              52/56        19 s
#   0.09     3.2            0.37              88/95        17 s
#   0.12     9.0            0.57             100/115       48 s
#   0.15    21.6            0.93              68/100      119 s
#
# 0.01 would leave the map empty for most of the episode. 0.12 and above tips
# the queue past capacity: the backlog runs away to the id ceiling and the run
# stops measuring allocation quality and starts measuring saturation. 0.09 puts
# arrivals just under the fleet's rescue throughput, so the system sits near
# capacity — overloaded in ~37% of slots, draining to manageable in between.
# That is what makes BOTH load regimes appear in one episode, which is precisely
# what the judge's regime-adaptive reasoning has to be exercised against.
#
# P_BIRTH and LAMBDA_RESCUE jointly set the load and must be swept together.
P_BIRTH     = 0.09           # prob a new target appears each slot
# Ceiling on concurrently-live target ids. This is NOT a coverage invariant any
# more: the old MAX_TARGETS = NUM_UAVS encoded "every target has its own UAV",
# which is exactly the premise the overloaded regime discards. It is now only a
# bound on the fixed-size id space used by the logging/plot arrays, set well above
# any backlog a 1000-slot run can accumulate.
MAX_TARGETS = 32


# ---- Agentic AI (three-tier: detector -> judge -> planner) ----
# Model cascade mirrors the tiers' cost asymmetry: the JUDGE runs often, on a
# digest, and gets the light model; the PLANNER runs rarely, on full state, and
# gets the stronger one.
JUDGE_MODEL   = "gemini-3.1-flash-lite"
PLANNER_MODEL = "gemini-3.1-flash"

# The deterministic detector raises an alarm on every birth and every rescue.
# Between those it also ticks periodically, so the slow load-imbalance that builds
# up as targets wander (sets spread out, round-robin travel grows) still reaches
# the judge. The periodic tick is a BACKSTOP, not the workhorse — birth and rescue
# are the events that actually change the load regime.
DETECTOR_TICK = 40    # slots between periodic (no-event) alarms

# Minimum slots between two planner invocations. Prevents a burst of alarms from
# thrashing the partition; the judge is what decides whether to spend a replan,
# this is only the floor.
REPLAN_COOLDOWN = 5
