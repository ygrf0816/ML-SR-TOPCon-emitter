"""One-shot patch: fix DE trace std/percentile semantics.

Targets (relative to the topcon_experiments package root):
  1. exp2_inverse/de_single.py   -> record per-generation distribution stats
  2. common/plot_utils.py        -> plot quantile band instead of mean +/- std

Backups must already exist. Refuses to run if the exact anchors are not found.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
DE_SINGLE = ROOT / "exp2_inverse" / "de_single.py"
PLOT_UTILS = ROOT / "common" / "plot_utils.py"


# --------------------------------------------------------------------------
# 1) de_single.py : _flush_generation
# --------------------------------------------------------------------------
OLD_FLUSH = '''    def _flush_generation(gen: int) -> None:
        nonlocal last_flush
        chunk = gen_vals[last_flush:]
        if not chunk:
            return
        trace_rows.append({
            "gen": gen,
            "best": float(best_so_far),
            "mean": float(np.mean(chunk)),
            "std": float(np.std(chunk)),
        })
        last_flush = len(gen_vals)
'''

NEW_FLUSH = '''    def _flush_generation(gen: int) -> None:
        # NOTE on sampling口径 (sample definition), three facts that matter:
        #   (a) `best_so_far` is the CUMULATIVE running max over EVERY objective
        #       evaluation so far -> monotone non-decreasing across generations.
        #       It is NOT a statistic of `chunk`.
        #   (b) `chunk` holds only the individuals evaluated *within the current
        #       generation*. SciPy evaluates NP = popsize * n_params trial vectors
        #       per generation; the very first chunk additionally contains the NP
        #       initial-population evaluations, so gen 0 has 2*NP samples. The
        #       `n_evals` column makes this explicit.
        #   (c) `mean +/- 1 std` is a DISPERSION measure, NOT a range. For the
        #       strongly left-skewed populations of a converged DE (most members
        #       near the optimum, a few stragglers far below) it is mathematically
        #       possible and in fact common that mean + std > best. Do not use it
        #       as an interval: use the quantiles / gen_best / gen_worst below.
        nonlocal last_flush
        chunk = gen_vals[last_flush:]
        if not chunk:
            return
        arr = np.asarray(chunk, dtype=float)
        q05, q25, q50, q75, q95 = (
            float(v) for v in np.percentile(arr, [5, 25, 50, 75, 95])
        )
        trace_rows.append({
            "gen": gen,
            "best": float(best_so_far),      # cumulative running max (all evals)
            "n_evals": int(arr.size),        # samples behind the stats below
            "gen_best": float(np.max(arr)),  # best within THIS generation
            "gen_worst": float(np.min(arr)),  # worst within THIS generation
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),  # dispersion, NOT a range bound
            "median": q50,
            "p05": q05,
            "p25": q25,
            "p75": q75,
            "p95": q95,
        })
        last_flush = len(gen_vals)
'''


# --------------------------------------------------------------------------
# 2) plot_utils.py : plot_trace
# --------------------------------------------------------------------------
OLD_PLOT = '''def plot_trace(trace_df: pd.DataFrame, metric: str, out_prefix: Path) -> None:
    save_csv(trace_df, out_prefix.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trace_df["gen"], trace_df["best"], label="best", lw=2)
    ax.plot(trace_df["gen"], trace_df["mean"], label="mean", alpha=0.8)
    ax.fill_between(
        trace_df["gen"],
        trace_df["mean"] - trace_df["std"],
        trace_df["mean"] + trace_df["std"],
        alpha=0.2,
        label="mean +/- std",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric)
    ax.set_title(f"DE convergence: maximize {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)
'''

NEW_PLOT = '''def plot_trace(trace_df: pd.DataFrame, metric: str, out_prefix: Path) -> None:
    """Plot a DE convergence trace.

    Two different sampling units live in this frame and must not be mixed:

      * ``best``     - cumulative running max over ALL evaluations so far, hence
                       monotone non-decreasing. Hard upper bound on *every*
                       individual ever evaluated (including the current one).
      * ``mean``, ``std``, ``median``, ``p05..p95``, ``gen_best``, ``gen_worst``
                     - statistics of the individuals evaluated *within that
                       generation only*.

    ``mean +/- 1 std`` is a dispersion measure, not an interval. Because a
    converged DE population is strongly left-skewed (most members hug the
    optimum, a few stragglers sit far below), ``mean + std`` can legitimately
    exceed ``best``: by Popoviciu's inequality std <= (max - min) / 2, so
    ``mean + std`` overshoots the max as soon as the mean sits close to it.
    Observed on the archived runs: 25/30 generations for iv_Eff and 29/30 for
    iv_Jsc. Drawing that as a shaded band is therefore misleading.

    The band is consequently drawn from population QUANTILES (hard-bounded by
    the achievable range, so it can never overshoot), while ``std`` is plotted
    separately on the right axis as a population-diversity indicator.
    """
    save_csv(trace_df, out_prefix.with_suffix(".csv"))
    gen = trace_df["gen"]
    has_q = {"p05", "p25", "p75", "p95"}.issubset(trace_df.columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gen, trace_df["best"], color="C3", lw=2.2, label="best (cumulative)")

    if has_q:
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

    ax.set_xlabel("Generation")
    ax.set_ylabel(metric)
    ax.set_title(f"DE convergence: maximize {metric}")

    if "std" in trace_df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            gen, trace_df["std"], color="C7", ls=":", lw=1.6,
            label="std (population diversity)",
        )
        ax2.set_ylabel(f"std of {metric}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)
'''


def apply(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new.strip() in text:
        print(f"[skip] already patched: {path}")
        return
    if old not in text:
        print(f"[FAIL] anchor not found in {path}", file=sys.stderr)
        sys.exit(1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, path.with_suffix(path.suffix + f".bak_{stamp}"))
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[ok] patched {path} (backup {path.name}.bak_{stamp})")


if __name__ == "__main__":
    apply(DE_SINGLE, OLD_FLUSH, NEW_FLUSH)
    apply(PLOT_UTILS, OLD_PLOT, NEW_PLOT)
    print("done")
