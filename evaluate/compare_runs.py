"""
Overlay the mission curves from separate eval runs that were each launched as
their own `python main.py --mode ...` invocation — different seeds, different
lambda values, or the one-to-one ablation.

This is the tool for the LAMBDA SWEEP. lambda sets the load regime (small values
hold the system in persistent heavy overload, large ones let it repeatedly drain
to a manageable state), so overlaying runs across lambda is how the allocator's
regime-adaptivity is demonstrated rather than asserted.

main.py's marl/agentic/compare modes already save each run's series to
plots/seed{N}/{mode}/raw_eval.npz. Point this script at any set of those files
with a label per run.

Example
-------
python -m evaluate.compare_runs \
    --run "lambda=0.002 (heavy overload)" plots/seed1/agentic/raw_eval.npz \
    --run "lambda=0.005 (nominal)"        plots/seed2/agentic/raw_eval.npz \
    --run "lambda=0.03  (drains)"         plots/seed3/agentic/raw_eval.npz \
    -o plots/lambda_sweep
"""
import argparse

from evaluate.utils import load_raw_eval
from evaluate.plot  import plot_multi_run


def main():
    parser = argparse.ArgumentParser(
        description="Overlay backlog / rescue / delay curves from separate eval runs")
    parser.add_argument(
        "--run", nargs=2, action="append", required=True,
        metavar=("LABEL", "NPZ_PATH"),
        help="Repeatable. A legend label and the raw_eval.npz path for one run.",
    )
    parser.add_argument(
        "-o", "--out", default="plots/compare",
        help="Output directory for the comparison figures",
    )
    args = parser.parse_args()

    runs = [(label, load_raw_eval(path)) for label, path in args.run]
    plot_multi_run(runs, args.out)

    # The scalar table alongside the curves: the curves show the shape, these are
    # the numbers that go in the paper.
    print("=" * 78)
    print(f"{'run':<34}{'D-bar (s)':>11}{'rescued':>9}{'born':>7}{'mean |K|':>10}")
    for label, d in runs:
        D    = float(d["avg_rescue_delay"]) if "avg_rescue_delay" in d else float("nan")
        nres = int(d["n_rescued"]) if "n_rescued" in d else 0
        nborn = int(d["n_born"]) if "n_born" in d else 0
        km   = float(d["backlog"].mean())
        print(f"{label[:34]:<34}{D:11.2f}{nres:9d}{nborn:7d}{km:10.2f}")
    print("=" * 78)
    print(f"Saved figures to {args.out}/")


if __name__ == "__main__":
    main()
