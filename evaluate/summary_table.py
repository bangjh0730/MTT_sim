"""
The merged scalar summary — one table replacing three former bar charts
(slots-until-recovery, LLM latency, LLM call count).

Those three were scalar summary statistics with no distributional shape worth a
float slot: a bar of a mean and an error whisker carries exactly two numbers per
group, which is a table row. The shape that DOES matter — the spread and the
tail of recovery time — is now carried properly by the CDF and strip figures
(evaluate/recovery_time.py), which frees this to be what it should be: the cost
line of the argument, read rather than eyeballed.

Rows are the three disturbance modes; columns pair each policy's median and p90
recovery. p90 sits beside the median deliberately: the medians alone can look
merely better, while the tail is where the rule-based policy actually fails, and
a p90 rendered ">232" is the honest way to say a tenth of its compound events
never recovered at all rather than quietly dropping them from the quantile.

Latency and call count are agentic-only (the rule-based policy runs no LLM) and
are reported in slots as well as seconds, because "the planner costs ~4 slots"
is the number that can be compared against a 22-slot median recovery.

Run:  python -m evaluate.summary_table
      python -m evaluate.summary_table --latex     # booktabs body for the paper
"""
import argparse
import glob
import os

import numpy as np

from config.params import DT
from evaluate.recovery_events import (
    MODES, POLICIES, SENSED_THRESHOLD, load_all, outcomes, partition, fmt_quantile,
)

AGENTS = [("MA", "ma"), ("PA", "pa")]


def _llm_stats(mode_dir: str) -> dict:
    """{agent_key: dict(calls_per_run, lat_mean_s, lat_sd_s, n_runs)} for one mode.

    Latency is per CALL (pooled over runs); call count is per RUN. Mixing those
    two denominators in one row is intentional — "how long does one reasoning
    step take" and "how many steps does an episode need" are the two separate
    costs a reviewer asks about.
    """
    runs = []
    for f in sorted(glob.glob(os.path.join("plots", mode_dir, "seed*", "agentic",
                                           "llm_latency.npz"))):
        d = np.load(f)
        runs.append({a: d[f"{a}_latencies"] for _, a in AGENTS})

    out = {}
    for _, akey in AGENTS:
        pooled = np.concatenate([r[akey] for r in runs]) if runs else np.array([])
        per_run = [r[akey].size for r in runs]
        out[akey] = dict(
            n_runs=len(runs),
            calls_per_run=float(np.mean(per_run)) if per_run else 0.0,
            lat_mean_s=float(np.mean(pooled)) if pooled.size else 0.0,
            lat_sd_s=float(np.std(pooled)) if pooled.size else 0.0,
        )
    return out


def _rows(events: dict, clean_only: bool) -> list:
    rows = []
    for mlabel, mkey in MODES:
        cells = {}
        for plabel, pkey, _, _ in POLICIES:
            ev = events[(mkey, pkey)]
            vals = outcomes(ev, clean_only)
            p = partition(ev, clean_only)
            cells[pkey] = dict(
                n=p["n_total"],
                median=fmt_quantile(vals, 0.5),
                p90=fmt_quantile(vals, 0.9),
                never=f"{p['n_censored']}/{p['n_total']}",
                clean=p["n_total"] - p["n_contaminated"],
            )
        rows.append((mlabel, mkey, cells, _llm_stats(mkey)))
    return rows


def _print_text(rows: list):
    head = (f"{'Mode':14s} | {'med':>5s} {'p90':>5s} {'never':>6s} | "
            f"{'med':>5s} {'p90':>5s} {'never':>6s} | {'MA':>5s} {'PA':>5s} | "
            f"{'PA latency':>16s}")
    sub = (f"{'':14s} | {'--- rule-based ---':^20s} | "
           f"{'---- agentic ----':^20s} | {'calls/run':^11s} | {'per call':>16s}")
    print(sub)
    print(head)
    print("-" * len(head))
    for mlabel, _, c, llm in rows:
        r, a = c["marl"], c["agentic"]
        pa = llm["pa"]
        lat = (f"{pa['lat_mean_s']:.2f}+-{pa['lat_sd_s']:.2f}s "
               f"({pa['lat_mean_s'] / DT:.1f} sl)")
        print(f"{mlabel:14s} | {r['median']:>5s} {r['p90']:>5s} "
              f"{r['never']:>6s} | {a['median']:>5s} {a['p90']:>5s} {a['never']:>6s} | "
              f"{llm['ma']['calls_per_run']:>5.1f} {pa['calls_per_run']:>5.1f} | {lat:>16s}")
    print()
    print("never        = events whose target never recovered / (target, event) pairs")
    print("               pooled over that mode's seeds. The two policies can differ by")
    print("               one event: a failure's victim target depends on the assignment")
    print("               that policy happened to be holding when the UAV was lost.")
    print("med / p90    = slots from the disturbance until that target's own PCRLB is")
    print(f"               sustained below {SENSED_THRESHOLD:g} m^2; '>' = the quantile falls")
    print("               among the never-recovered events, so it is a lower bound")
    print("MA / PA      = Monitor / Planner agent; rule-based makes no LLM calls")
    print(f"latency      = wall-clock per PA reasoning call (1 slot = {DT:g} s)")


def _print_latex(rows: list):
    print(r"% requires \usepackage{booktabs}")
    print(r"\begin{tabular}{lrrrrrrrrr}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Rule-based} & \multicolumn{3}{c}{Agentic}"
          r" & \multicolumn{2}{c}{Calls/run} & PA latency \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}")
    print(r"Mode & med & p90 & never & med & p90 & never & MA & PA & (s) \\")
    print(r"\midrule")
    for mlabel, _, c, llm in rows:
        r, a = c["marl"], c["agentic"]
        pa = llm["pa"]
        fmt = lambda s: s.replace(">", r"$>$")
        print(f"{mlabel} & {fmt(r['median'])} & {fmt(r['p90'])} & {r['never'].replace('/', ' / ')} & "
              f"{fmt(a['median'])} & {fmt(a['p90'])} & {a['never']} & "
              f"{llm['ma']['calls_per_run']:.1f} & {pa['calls_per_run']:.1f} & "
              f"${pa['lat_mean_s']:.2f} \\pm {pa['lat_sd_s']:.2f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(
        description="Merged scalar summary: recovery median/p90, LLM latency, call count")
    parser.add_argument("--threshold", type=float, default=SENSED_THRESHOLD)
    parser.add_argument("--clean-only", action="store_true",
                        help="Restrict recovery columns to non-overlapping events")
    parser.add_argument("--latex", action="store_true",
                        help="Emit a booktabs table body instead of the text table")
    args = parser.parse_args()

    rows = _rows(load_all(args.threshold), args.clean_only)
    if args.latex:
        _print_latex(rows)
    else:
        _print_text(rows)


if __name__ == "__main__":
    main()
