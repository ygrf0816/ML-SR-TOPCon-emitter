"""Generate the active-learning candidate batch for the R1-4 revision.

Design rationale
----------------
The surrogate is an ensemble of tree-based models, so its output cannot leave the
range of its training labels (max 26.02 %).  Consequently the predicted mean mu(x)
is flat above ~26.2 % and a purely model-driven acquisition function degenerates:
every plausible recipe receives the same mu.  The first batch is therefore a
*design of experiments* batch centred on the only known point above the ceiling
(the differential-evolution optimum, 27.04 % by TCAD), and only the second batch
is model-driven (Gaussian process + expected improvement) once the augmented
labels extend the range.

Batch composition (50 points)
    A1  10 points : Gaussian perturbation of the DE optimum, sigma_u = 0.010
    A2  20 points : same, sigma_u = 0.020                (u = fraction of the box)
    B   20 points : Latin hypercube inside the high-efficiency slab, i.e. the
                    envelope of the 20 best samples widened by 20 % of its own
                    width, applied to all eight parameters (clipped to the box)

Everything is reproducible from the constants below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paths are placeholders on purpose. Fill in the three locations for your own
# installation; nothing else in this file depends on them.
ROOT = Path("<PATH>")          # project root
DATA = Path("<PATH>")          # the 10,159-row dataset CSV
DE = Path("<PATH>")            # outputs/exp2_inverse/de_best_iv_Eff.csv
OUT = Path(__file__).resolve().parent          # this directory

PAR = ["athena_thick", "athena_c_boron", "athena_temp1", "athena_time1",
       "athena_temp2", "athena_time2", "athena_F_N2", "athena_F_O2"]

SEED = 20261021
N_A1, N_A2, N_B = 10, 20, 20
SIG_A1, SIG_A2 = 0.010, 0.020
WIDEN = 1.20          # widening factor on the top-20 envelope


def main() -> None:
    df = pd.read_csv(DATA)
    de = pd.read_csv(DE).iloc[0]

    lo = df[PAR].min().values.astype(float)
    hi = df[PAR].max().values.astype(float)
    span = hi - lo
    mu = df[PAR].mean().values.astype(float)
    sd = df[PAR].std().values.astype(float)

    # high-efficiency slab from the 20 best samples, widened by WIDEN (clipped to the box)
    top = df.nlargest(20, "iv_Eff")
    slab_lo = np.clip(top[PAR].min().values - (WIDEN - 1) * (top[PAR].max().values - top[PAR].min().values) / 2, lo, hi)
    slab_hi = np.clip(top[PAR].max().values + (WIDEN - 1) * (top[PAR].max().values - top[PAR].min().values) / 2, lo, hi)

    rng = np.random.default_rng(SEED)
    x_de = de[PAR].values.astype(float)
    u_de = (x_de - lo) / span

    rows: list[dict] = []

    # A1 / A2: Gaussian perturbation of the optimum, in box-normalised coordinates
    for tag, n, sig in (("A1", N_A1, SIG_A1), ("A2", N_A2, SIG_A2)):
        u = np.clip(u_de + rng.normal(0.0, sig, size=(n, len(PAR))), 0.0, 1.0)
        u[:, 0] = np.clip(u[:, 0], 0.0, (0.030 - lo[0]) / span[0])   # keep the BSG thin
        for k in range(n):
            rows.append(dict(group=tag, **{p: float(lo[i] + u[k, i] * span[i]) for i, p in enumerate(PAR)}))

    # B: Latin hypercube inside the slab
    def lhs(n: int, d: int, rng) -> np.ndarray:
        cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
        for j in range(d):
            cut[:, j] = rng.permutation(cut[:, j])
        return cut

    u_slab = lhs(N_B, len(PAR), rng)
    for k in range(N_B):
        x = slab_lo + u_slab[k] * (slab_hi - slab_lo)
        rows.append(dict(group="B", **{p: float(v) for p, v in zip(PAR, x)}))

    cand = pd.DataFrame(rows)
    cand.insert(0, "cand_id", [f"AL{i:03d}" for i in range(1, len(cand) + 1)])
    for p, m, s in zip(PAR, mu, sd):
        cand[f"z_{p}"] = (cand[p] - m) / s
    cand["max_abs_z"] = cand[[f"z_{p}" for p in PAR]].abs().max(axis=1)
    cand["in_box"] = ((cand[PAR] >= lo) & (cand[PAR] <= hi)).all(axis=1)
    cand.to_csv(OUT / "candidates_50.csv", index=False, float_format="%.6g")

    spec = dict(seed=SEED, n_A1=N_A1, n_A2=N_A2, n_B=N_B, sig_A1=SIG_A1, sig_A2=SIG_A2,
                widen=WIDEN, box_lo={p: float(v) for p, v in zip(PAR, lo)},
                box_hi={p: float(v) for p, v in zip(PAR, hi)},
                slab_lo={p: float(v) for p, v in zip(PAR, slab_lo)},
                slab_hi={p: float(v) for p, v in zip(PAR, slab_hi)},
                de_optimum={p: float(v) for p, v in zip(PAR, de[PAR].values)},
                note="u = (x - box_lo) / (box_hi - box_lo); A groups perturb u, B is LHS in u_slab")
    (OUT / "design_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print(cand[["cand_id", "group"] + PAR[:2] + ["max_abs_z", "in_box"]].head(12).to_string(index=False))
    print("\nn=%d  in_box=%s  max|z| median %.2f max %.2f" % (
        len(cand), cand.in_box.all(), cand.max_abs_z.median(), cand.max_abs_z.max()))


if __name__ == "__main__":
    main()
