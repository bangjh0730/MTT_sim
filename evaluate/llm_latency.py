"""
LLM reasoning cost, per agent: call count and wall-clock latency for the Monitor
Agent (MA, cheap yes/no coverage poll) and Planner Agent (PA, reassignment
reasoning), across disturbance modes.

This is the detailed breakdown behind the cost columns of
evaluate/summary_table.py, which is where these numbers belong in the paper.
The two bar charts this script used to render (mean latency, mean calls/run)
are gone on purpose: each carried one mean and one error whisker per group,
which is two numbers — a table row, not a figure. Latency and call count have no
distributional shape worth a float slot, unlike recovery time, whose spread and
tail now get proper distributional figures in evaluate/recovery_time.py.

Reads the per-run llm_latency.npz files saved by evaluate.agentic, laid out as
plots/<mode>/seed<N>/agentic/llm_latency.npz. Latency is logged in wall-clock
seconds but also reported in sim slots (latency_s / DT), since slots are what's
comparable to recovery time and disturbance cadence elsewhere in evaluate/.

Run:  python -m evaluate.llm_latency
"""
import os
import glob

import numpy as np

from config.params import DT

MODES  = [("Birth", "birth"), ("Failure", "failure"), ("Birth + Failure", "birth+fail")]
AGENTS = [("MA", "ma"), ("PA", "pa")]


def _load_runs(mode_dir: str) -> list:
    """Per-seed (ma_lat, pa_lat) arrays for one mode."""
    out = []
    for f in sorted(glob.glob(os.path.join("plots", mode_dir, "seed*", "agentic", "llm_latency.npz"))):
        d = np.load(f)
        out.append(dict(ma_lat=d["ma_latencies"] / DT, pa_lat=d["pa_latencies"] / DT))
    return out


def _agent_stat(runs: list, akey: str) -> dict:
    """calls/run and per-call latency (mean +- sd, in slots) for one agent."""
    all_lat = np.concatenate([r[f"{akey}_lat"] for r in runs]) if runs else np.array([])
    return dict(
        calls_per_run=float(np.mean([r[f"{akey}_lat"].size for r in runs])) if runs else 0.0,
        mean=float(np.mean(all_lat)) if all_lat.size else 0.0,
        sd=float(np.std(all_lat)) if all_lat.size else 0.0,
    )


def main():
    data = {mkey: _load_runs(mkey) for _, mkey in MODES}

    lat_w = 16   # width of one "mean +- sd" latency cell
    print(f"{'Mode':16s} | {'calls/run':^11s} | {'MA latency':^{lat_w}s} | {'PA latency':^{lat_w}s}")
    print(f"{'':16s} | {'MA':>4s} / {'PA':<4s} | {'(slots)':^{lat_w}s} | {'(slots)':^{lat_w}s}")
    print("-" * (16 + 3 + 11 + 3 + lat_w + 3 + lat_w))
    for lbl, mkey in MODES:
        runs = data[mkey]
        stats = {akey: _agent_stat(runs, akey) for _, akey in AGENTS}
        calls = f"{stats['ma']['calls_per_run']:>4.1f} / {stats['pa']['calls_per_run']:<4.1f}"
        ma_lat = f"{stats['ma']['mean']:.2f} +- {stats['ma']['sd']:.2f}"
        pa_lat = f"{stats['pa']['mean']:.2f} +- {stats['pa']['sd']:.2f}"
        print(f"{lbl:16s} | {calls:^11s} | {ma_lat:^{lat_w}s} | {pa_lat:^{lat_w}s}")
    print()
    print("n runs per mode: " + ", ".join(f"{lbl}={len(data[mkey])}" for lbl, mkey in MODES))
    print("Latency is per call, pooled over runs, reported in sim slots "
          f"(1 slot = {DT:g} s); calls/run is per episode.")
    print("For the paper-facing summary see: python -m evaluate.summary_table")


if __name__ == "__main__":
    main()
