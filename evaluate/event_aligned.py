"""
Event-aligned ("epoch") traces: the AFFECTED TARGET's own PCRLB re-indexed to
t=0 at the disturbance that hit it, median + IQR band across events, split by
policy.

Two things changed from the fleet-average version this replaces, both to stop
the figure smearing:

  1. Per-target, not per-fleet. The old curve averaged PCRLB over every live
     target, so a disturbance that wrecked one target was diluted by the four
     that were fine, and the recovery shape was mostly the dilution. Each event
     now contributes only the series of the target it actually hit — see
     evaluate/recovery_events.py for how a disturbance is attributed to a target.

  2. Clean events only. An event with a second disturbance inside the plotted
     window is contaminated: whatever the curve does after that point is a
     response to both, and averaging it in is what let a slow tail get blamed on
     the policy when it was really event pile-up. Cleanliness is judged over a
     FIXED horizon, never the event's own realised recovery window, so the
     filter cannot preferentially delete slow events — see CLEAN_HORIZON.

This figure is the temporal-curve intuition and the mechanism check; it is NOT
the headline. The distributional evidence lives in evaluate/recovery_time.py
(per-event CDF + strip), which needs no clean filter at all because each event
contributes exactly one number and overlap cannot smear across events.

--metric rmse plots tracking RMSE instead — the appendix confirmation that the
estimator actually tracks the bound. RMSE and PCRLB are tightly coupled (the
PCRLB lower-bounds the MSE), so RMSE earns a secondary panel rather than a
top-level figure of its own. Note the RMSE series in raw_eval.npz is a
fleet-wide mean over live targets, not per-target, so its panel is the coupling
check only and cannot resolve a single target's recovery.

Run:  python -m evaluate.event_aligned
      python -m evaluate.event_aligned --metric rmse
"""
import os
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate.recovery_events import (
    MODES, POLICIES, CLEAN_HORIZON, MUTED, SENSED_THRESHOLD, load_events,
)

# (facet key, title, mode folder, event kind). Laid out 2x2 as (event type)
# columns x (isolated / compounded) rows, so the same event type sits in one
# column and the isolated-vs-compounded comparison is a straight up/down read at
# the same x position.
FACETS = [
    ("birth_only",     "Target birth (isolated)",  "birth",      "birth"),
    ("fail_only",      "UAV failure (isolated)",   "failure",    "failure"),
    ("birth_compound", "Target birth (compound)",  "birth+fail", "birth"),
    ("fail_compound",  "UAV failure (compound)",   "birth+fail", "failure"),
]

METRICS = {
    "pcrlb": dict(log=True,  per_target=True,
                  ylabel=r"Affected target's $\log_{10}$ PCRLB (m$^2$)"),
    "rmse":  dict(log=False, per_target=False,
                  ylabel="Fleet tracking RMSE (m)"),
}


def _series_for(event, metric: str) -> np.ndarray:
    """The metric series this event should contribute, or None."""
    d = np.load(event.path)
    if METRICS[metric]["per_target"]:
        return d["pcrlb_per_target"][event.target]
    return d["rmse"] if "rmse" in d.files else None


def _window(series, onset: int, pre: int, post: int, log: bool):
    """series over [onset-pre, onset+post], NaN-padded where it runs off either
    end of the episode. Padding rather than dropping keeps a late-episode event
    in the sample instead of biasing toward events with room around them."""
    idx = onset - 1                     # onsets are 1-indexed
    out = np.full(pre + post + 1, np.nan)
    lo, hi = idx - pre, idx + post + 1
    src = series[max(lo, 0):min(hi, len(series))]
    out[max(0, -lo):max(0, -lo) + len(src)] = src
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log10(out)
    return out


def _facet_events(mode_dir: str, kind: str, pre: int, post: int, metric: str,
                  clean_only: bool) -> dict:
    """{pkey: [(window, absorbed), ...]}. absorbed = the disturbance never pushed
    the affected target above threshold (e.recovery == 0) — for a failure, the
    UAV lost was a co-tracker so its loss was marginal. Always False for births
    (a newborn starts at the spawn prior, well above threshold)."""
    out = {pkey: [] for _, pkey, _, _ in POLICIES}
    for _, pkey, _, _ in POLICIES:
        for e in load_events(mode_dir, pkey):
            if e.kind != kind or (clean_only and not e.clean):
                continue
            s = _series_for(e, metric)
            if s is None:
                continue
            w = _window(s, e.onset, pre, post, METRICS[metric]["log"])
            out[pkey].append((w, e.recovery == 0))
    return out


def _windows(events: dict, pkey: str) -> np.ndarray:
    """Just the window arrays for one policy (drop the absorbed flag)."""
    return np.array([w for w, _ in events[pkey]])


MIN_N_BAND = 3   # below this, an IQR band is a decoration, not a spread


def _band(ax, t, arr, color, label, zbase=2):
    """Median line + IQR fill for one array of windows. nan-aware: a window is
    NaN before its target was born, after it died, and past the episode edge,
    so the event count varies along the curve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices
        med = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
    ax.plot(t, med, color=color, lw=1.8, label=label, zorder=zbase + 1)
    ax.fill_between(t, q1, q3, color=color, alpha=0.18, linewidth=0, zorder=zbase)


def _traces(ax, t, arr, color, label, alpha=0.45, lw=1.0):
    """Every individual window as a light line — for the n<3 case and for the
    bimodal failure facets, where a median+band implies a continuum that the
    two-cluster (absorbed / degraded) data does not have."""
    for j, w in enumerate(arr):
        ax.plot(t, w, color=color, lw=lw, alpha=alpha, zorder=3,
                label=label if j == 0 else None)


def _decorate(ax, title, thresh_y):
    ax.axvline(0, color=MUTED, ls="--", lw=1, zorder=1)
    if thresh_y is not None:
        ax.axhline(thresh_y, color=MUTED, ls=":", lw=1, zorder=1)
    ax.set_title(title, fontsize=11)
    ax.grid(ls=":", alpha=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def _plot_band(ax, events, title, pre, post, thresh_y):
    """Median + IQR band per policy (falls back to traces below MIN_N_BAND)."""
    t = np.arange(-pre, post + 1)
    for plabel, pkey, color, _ in POLICIES:
        arr = _windows(events, pkey)
        if arr.size == 0:
            continue
        n = arr.shape[0]
        if n < MIN_N_BAND:
            _traces(ax, t, arr, color, f"{plabel} (n={n}, traces)", alpha=0.85, lw=1.3)
        else:
            _band(ax, t, arr, color, f"{plabel} (n={n})")
    _decorate(ax, title, thresh_y)


def _plot_traces(ax, events, title, pre, post, thresh_y):
    """Option 1: every event as its own light line, no median, no band — so the
    two clusters show directly instead of being averaged into a false middle."""
    t = np.arange(-pre, post + 1)
    for plabel, pkey, color, _ in POLICIES:
        arr = _windows(events, pkey)
        if arr.size:
            _traces(ax, t, arr, color, f"{plabel} (n={arr.shape[0]})")
    _decorate(ax, title, thresh_y)


def _plot_split(ax, events, title, pre, post, thresh_y):
    """Option 2: split each policy into absorbed vs degraded and summarise each
    separately, with counts. Degraded (target actually blew up) is unimodal, so
    a median+band is honest there; absorbed (never left threshold) is a flat
    cluster near the floor, drawn as a dashed median only."""
    t = np.arange(-pre, post + 1)
    for plabel, pkey, color, _ in POLICIES:
        arr = _windows(events, pkey)
        if arr.size == 0:
            continue
        absorbed = np.array([w for (w, a) in events[pkey] if a])
        degraded = np.array([w for (w, a) in events[pkey] if not a])
        nD, nA = len(degraded), len(absorbed)
        if nD >= MIN_N_BAND:
            _band(ax, t, degraded, color, f"{plabel} degraded (n={nD})")
        elif nD:
            _traces(ax, t, degraded, color, f"{plabel} degraded (n={nD})", alpha=0.85, lw=1.3)
        if nA:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                medA = np.nanmedian(absorbed, axis=0)
            ax.plot(t, medA, color=color, lw=1.6, ls=(0, (4, 2)), alpha=0.9,
                    zorder=4, label=f"{plabel} absorbed (n={nA})")
    _decorate(ax, title, thresh_y)


# Failure facets are bimodal; the birth facets are not, so they always use band.
FAILURE_STYLES = {"band": _plot_band, "traces": _plot_traces, "split": _plot_split}


def _shared_ylim(facets: list, full_range: bool, headroom: float = 0.40):
    """Common y range across facets, with headroom above the highest mark so the
    upper-right legend never sits on top of a curve. full_range uses the actual
    data min/max (traces/split modes draw individual events, which can sit
    outside the IQR the band mode summarised)."""
    lo, hi = [], []
    for _, _, events in facets:
        for _, pkey, _, _ in POLICIES:
            arr = _windows(events, pkey)
            if arr.size == 0 or np.isnan(arr).all():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if full_range:
                    lo.append(np.nanmin(arr))
                    hi.append(np.nanmax(arr))
                else:
                    lo.append(np.nanmin(np.nanpercentile(arr, 25, axis=0)))
                    hi.append(np.nanmax(np.nanpercentile(arr, 75, axis=0)))
    if not lo:
        return None
    lo, hi = min(lo), max(hi)
    span = hi - lo
    return lo - 0.08 * span, hi + headroom * span


def main():
    parser = argparse.ArgumentParser(
        description="Event-aligned per-target recovery traces: rule-based vs agentic")
    parser.add_argument("--pre", type=int, default=10,
                        help="Slots kept before each disturbance onset")
    parser.add_argument("--post", type=int, default=CLEAN_HORIZON,
                        help="Slots kept after onset; defaults to the horizon over "
                             "which events are screened for interference")
    parser.add_argument("--all-events", action="store_true",
                        help="Include contaminated events too (shows what the clean "
                             "filter removes; not the figure to publish)")
    parser.add_argument("--metric", choices=list(METRICS), default="pcrlb")
    parser.add_argument("--failure-style", choices=list(FAILURE_STYLES), default="band",
                        help="How to summarise the (bimodal) UAV-failure facets: "
                             "'band' median+IQR (default; misleading on bimodal data), "
                             "'traces' every event as a light line (option 1), or "
                             "'split' absorbed-vs-degraded curves with counts (option 2). "
                             "Birth facets always use band (they are unimodal).")
    parser.add_argument("-o", "--out", default="plots/paper")
    args = parser.parse_args()
    pre, post, metric = args.pre, args.post, args.metric
    clean_only = not args.all_events

    facets = [(title, kind, _facet_events(mode, kind, pre, post, metric, clean_only))
              for _, title, mode, kind in FACETS]
    if all(not ev[p[1]] for _, _, ev in facets for p in POLICIES):
        raise SystemExit(
            f"No '{metric}' data found in any raw_eval.npz under plots/<mode>/seed*/. "
            f"Run the evals first (main.py --mode marl|agentic --births --failures).")

    plt.rcParams.update({
        "font.size": 11, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
    })
    # The recovery threshold, drawn as a reference line: the curve crossing it
    # (and staying) is exactly the event the recovery figures count.
    thresh_y = (np.log10(SENSED_THRESHOLD) if METRICS[metric]["log"]
                else None) if metric == "pcrlb" else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), sharex=True, sharey=True)
    for (title, kind, events), ax in zip(facets, axes.flat):
        # Birth facets are unimodal -> always band. Failure facets follow --failure-style.
        plot_fn = FAILURE_STYLES[args.failure_style] if kind == "failure" else _plot_band
        plot_fn(ax, events, title, pre, post, thresh_y)
    # traces/split draw individual events that can exceed the IQR the band mode used.
    ylim = _shared_ylim(facets, full_range=(args.failure_style != "band"))
    if ylim:
        axes.flat[0].set_ylim(*ylim)     # shared via sharey
    for ax in axes[:, 0]:
        ax.set_ylabel(METRICS[metric]["ylabel"])
    for ax in axes.flat:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("Slots since disturbance")
    fig.tight_layout()

    os.makedirs(args.out, exist_ok=True)
    style_tag = "" if args.failure_style == "band" else f"_{args.failure_style}"
    tag = ("" if clean_only else "_allevents") + style_tag
    out = os.path.join(args.out, f"event_aligned_{metric}{tag}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    print()
    print(f"{'Facet':28s} " + " ".join(f"{p[0]:>14s}" for p in POLICIES))
    for title, kind, events in facets:
        cells = []
        for _, pkey, _, _ in POLICIES:
            n = len(events[pkey])
            if kind == "failure":
                nA = sum(1 for _, a in events[pkey] if a)
                cells.append(f"{n} ({n - nA}deg/{nA}abs)")
            else:
                cells.append(str(n))
        print(f"  {title:26s} " + " ".join(f"{c:>14s}" for c in cells))
    print()
    print("n = events per curve; failure facets also show degraded/absorbed split"
          + ("" if clean_only else "  (contaminated events INCLUDED)"))


if __name__ == "__main__":
    main()
