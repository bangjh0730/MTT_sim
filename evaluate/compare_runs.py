"""
Overlay PCRLB / energy curves from separate eval runs (different seeds and/or
disturbance flags, e.g. births-only vs failures-only vs both) that were each
run as their own `python main.py --mode ...` invocation.

main.py's marl/agentic/compare modes already save each run's raw series to
plots/seed{N}/{mode}/raw_eval.npz. Point this script at any set of those
files with a label per run to plot them together.

Example
-------
python -m evaluate.compare_runs \
    --run "Births (seed1)"          plots/seed1/agentic/raw_eval.npz \
    --run "Failures (seed2)"        plots/seed2/agentic/raw_eval.npz \
    --run "Births+Failures (seed3)" plots/seed3/agentic/raw_eval.npz \
    -o plots/scenario_compare
"""
import argparse

from evaluate.utils import load_raw_eval
from evaluate.plot   import plot_multi_run


def main():
    parser = argparse.ArgumentParser(
        description="Overlay PCRLB/energy curves from separate eval runs")
    parser.add_argument(
        "--run", nargs=2, action="append", required=True,
        metavar=("LABEL", "NPZ_PATH"),
        help="Repeatable. A legend label and the raw_eval.npz path for one run.",
    )
    parser.add_argument(
        "-o", "--out", default="plots/compare",
        help="Output directory for pcrlb_compare.png / energy_compare.png",
    )
    args = parser.parse_args()

    runs = []
    for label, path in args.run:
        pcrlb, energy, failure_slots = load_raw_eval(path)
        runs.append((label, pcrlb, energy, failure_slots))

    plot_multi_run(runs, args.out)
    print(f"Saved {args.out}/pcrlb_compare.png and {args.out}/energy_compare.png")


if __name__ == "__main__":
    main()
