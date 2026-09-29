"""Follow-up patch: pandas 2.x + matplotlib 3.4 incompatibility.

matplotlib 3.4's cbook._check_1d probes `x[:, None]`, which pandas >= 2.0 turns
into a hard ValueError (matplotlib only catches AssertionError/IndexError/TypeError).
Result: every `ax.plot(Series, ...)`, `ax.scatter(Series, ...)`, `ax.barh(Series, ...)`
raises. Coerce all pandas columns to numpy arrays before handing them to matplotlib.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
PLOT_UTILS = ROOT / "common" / "plot_utils.py"

# --- 1) module-level helper -------------------------------------------------
OLD_HELPER = '''def save_csv(df: pd.DataFrame, csv_path: Path) -> None:
'''
NEW_HELPER = '''def _np(s: "pd.Series | np.ndarray") -> np.ndarray:
    """Coerce to ndarray before handing data to matplotlib.

    matplotlib 3.4's ``cbook._check_1d`` probes ``x[:, None]``; pandas >= 2.0
    raises ValueError instead of warning, and matplotlib only catches
    AssertionError/IndexError/TypeError, so any ``ax.plot(Series, ...)`` blows up.
    """
    return np.asarray(s)


def save_csv(df: pd.DataFrame, csv_path: Path) -> None:
'''

# --- 2) plot_trace: use numpy arrays ---------------------------------------
OLD_GEN = '''    save_csv(trace_df, out_prefix.with_suffix(".csv"))
    gen = trace_df["gen"]
    has_q = {"p05", "p25", "p75", "p95"}.issubset(trace_df.columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gen, trace_df["best"], color="C3", lw=2.2, label="best (cumulative)")
'''
NEW_GEN = '''    save_csv(trace_df, out_prefix.with_suffix(".csv"))
    gen = _np(trace_df["gen"])
    best = _np(trace_df["best"])
    mean = _np(trace_df["mean"])
    has_q = {"p05", "p25", "p75", "p95"}.issubset(trace_df.columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gen, best, color="C3", lw=2.2, label="best (cumulative)")
'''

OLD_BRANCH = '''    if has_q:
        ax.fill_between(
            gen, trace_df["p05"], trace_df["p95"],
            color="C0", alpha=0.15, lw=0, label="population 5-95%",
        )
        ax.fill_between(
            gen, trace_df["p25"], trace_df["p75"],
            color="C0", alpha=0.30, lw=0, label="population 25-75%",
        )
        ax.plot(gen, trace_df["median"], color="C0", lw=1.4, label="median (per gen)")
        if "gen_best" in trace_df.columns:
            ax.plot(
                gen, trace_df["gen_best"], color="C1", ls="--", lw=1.4,
                label="best in generation",
            )
    else:
        # Legacy 4-column trace (gen/best/mean/std): quantiles were never stored,
        # so clip the band at the hard upper bound `best`. No evaluated
        # individual can exceed `best`, hence a band reaching above it is
        # provably outside the achievable range.
        upper = np.minimum(trace_df["mean"] + trace_df["std"], trace_df["best"])
        ax.fill_between(
            gen, trace_df["mean"] - trace_df["std"], upper,
            color="C0", alpha=0.25, lw=0,
            label="mean +/- std (upper clipped at best)",
        )
    ax.plot(gen, trace_df["mean"], color="C2", lw=1.2, alpha=0.9, label="mean (per gen)")
'''
NEW_BRANCH = '''    if has_q:
        ax.fill_between(
            gen, _np(trace_df["p05"]), _np(trace_df["p95"]),
            color="C0", alpha=0.15, lw=0, label="population 5-95%",
        )
        ax.fill_between(
            gen, _np(trace_df["p25"]), _np(trace_df["p75"]),
            color="C0", alpha=0.30, lw=0, label="population 25-75%",
        )
        ax.plot(
            gen, _np(trace_df["median"]), color="C0", lw=1.4,
            label="median (per gen)",
        )
        if "gen_best" in trace_df.columns:
            ax.plot(
                gen, _np(trace_df["gen_best"]), color="C1", ls="--", lw=1.4,
                label="best in generation",
            )
    else:
        # Legacy 4-column trace (gen/best/mean/std): quantiles were never stored,
        # so clip the band at the hard upper bound `best`. No evaluated
        # individual can exceed `best`, hence a band reaching above it is
        # provably outside the achievable range.
        std = _np(trace_df["std"])
        upper = np.minimum(mean + std, best)
        ax.fill_between(
            gen, mean - std, upper,
            color="C0", alpha=0.25, lw=0,
            label="mean +/- std (upper clipped at best)",
        )
    ax.plot(gen, mean, color="C2", lw=1.2, alpha=0.9, label="mean (per gen)")
'''

OLD_STD_AXIS = '''    if "std" in trace_df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            gen, trace_df["std"], color="C7", ls=":", lw=1.6,
            label="std (population diversity)",
        )
'''
NEW_STD_AXIS = '''    if "std" in trace_df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            gen, _np(trace_df["std"]), color="C7", ls=":", lw=1.6,
            label="std (population diversity)",
        )
'''

# --- 3) same latent bug in the other plotters ------------------------------
OLD_BARH = '''    ax.barh(plot_df["feature"], plot_df["importance"])'''
NEW_BARH = '''    ax.barh(_np(plot_df["feature"]), _np(plot_df["importance"]))'''

OLD_SCATTER = '''    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=20)'''
NEW_SCATTER = '''    ax.scatter(_np(df[x_col]), _np(df[y_col]), alpha=0.6, s=20)'''

PAIRS = [
    (OLD_HELPER, NEW_HELPER),
    (OLD_GEN, NEW_GEN),
    (OLD_BRANCH, NEW_BRANCH),
    (OLD_STD_AXIS, NEW_STD_AXIS),
    (OLD_BARH, NEW_BARH),
    (OLD_SCATTER, NEW_SCATTER),
]


def main() -> None:
    text = PLOT_UTILS.read_text(encoding="utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for old, new in PAIRS:
        if new in text:
            print(f"[skip] already applied: {old.splitlines()[0][:60]}")
            continue
        if old not in text:
            print(f"[FAIL] anchor not found: {old.splitlines()[0][:60]}", file=sys.stderr)
            sys.exit(1)
        text = text.replace(old, new, 1)
        print(f"[ok] applied: {old.splitlines()[0][:60]}")
    shutil.copy2(PLOT_UTILS, PLOT_UTILS.with_suffix(PLOT_UTILS.suffix + f".bak_{stamp}"))
    PLOT_UTILS.write_text(text, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
