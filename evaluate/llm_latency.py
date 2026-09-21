"""
LLM reasoning cost, per tier: call count and wall-clock latency for the Judge
Agent (JA, the light "is a re-partition worth it" call) and the Planner Agent
(PA, the heavy set-allocation call), across disturbance modes.

The old Monitor Agent is gone: the first tier is now a DETERMINISTIC detector
that costs nothing and raises alarms every slot, so it has no latency to report.
Its alarm count is reported instead, because the ratio that justifies the whole
tiering is alarms -> judge calls -> planner calls: a cheap detector firing often,
a cheap judge filtering, and the expensive planner running rarely. A judge that
approved every alarm would show equal JA and PA counts and would not be earning
its place in the stack.

Reads the per-run llm_latency.npz files saved by evaluate.agentic, laid out as
plots/<mode>/seed<N>/agentic/llm_latency.npz. Latency is logged in wall-clock
seconds but also reported in sim slots (latency_s / DT), since slots are what is
comparable to the rescue cadence elsewhere in evaluate/.

Run:  python -m evaluate.llm_latency
"""
import os
import glob

import numpy as np

from config.params import DT

MODES  = [("Birth", "birth"), ("Failure", "failure"), ("Birth + Failure", "birth+fail")]
AGENTS = [("JA", "ja"), ("PA", "pa")]


def _load_runs(mode_dir: str) -> list:
    """Per-seed (ja_lat, pa_lat) arrays for one mode."""
    out = []
    for f in sorted(glob.glob(os.path.join("plots", mode_dir, "seed*", "agentic", "llm_latency.npz"))):
        d = np.load(f)
        out.append(dict(ja_lat=d["ja_latencies"] / DT, pa_lat=d["pa_latencies"] / DT))
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
    print(f"{'':16s} | {'JA':>4s} / {'PA':<4s} | {'(slots)':^{lat_w}s} | {'(slots)':^{lat_w}s}")
    print("-" * (16 + 3 + 11 + 3 + lat_w + 3 + lat_w))
    for lbl, mkey in MODES:
        runs = data[mkey]
        stats = {akey: _agent_stat(runs, akey) for _, akey in AGENTS}
        calls = f"{stats['ma']['calls_per_run']:>4.1f} / {stats['pa']['calls_per_run']:<4.1f}"
        ja_lat = f"{stats['ma']['mean']:.2f} +- {stats['ma']['sd']:.2f}"
        pa_lat = f"{stats['pa']['mean']:.2f} +- {stats['pa']['sd']:.2f}"
        print(f"{lbl:16s} | {calls:^11s} | {ja_lat:^{lat_w}s} | {pa_lat:^{lat_w}s}")
    print()
    print("n runs per mode: " + ", ".join(f"{lbl}={len(data[mkey])}" for lbl, mkey in MODES))
    print("Latency is per call, pooled over runs, reported in sim slots "
          f"(1 slot = {DT:g} s); calls/run is per episode.")
    print("For the paper-facing summary see: python -m evaluate.summary_table")


if __name__ == "__main__":
    main()
