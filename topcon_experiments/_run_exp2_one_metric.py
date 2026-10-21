"""Run ONE IV metric of exp2_inverse DE into the re-run output directory.

Usage: python _run_exp2_one_metric.py iv_Voc

Why one metric per process
--------------------------
The 4 metrics are independent. Running them as 4 concurrent processes keeps each
process at workers=1 -> scipy stays on updating='immediate' (using workers != 1
would force 'deferred' and change the search), while cutting wall-clock time
from ~7.5 h sequential to ~2 h.

Per-metric search semantics are identical to the sequential run:
DE_SEED=42, DE_MAXITER=30, DE_POPSIZE=15, mutation=(0.5,1.0), recombination=0.7,
polish=False, bounds = 5-95% quantile of the dataset.

Thread counts are pinned via env (OMP/MKL/OPENBLAS) by the launcher so that the
torch models in the AutoGluon ensembles use a fixed number of threads -> stable
floating-point reduction order.

Run with the `autogluon` env (AutoGluon 1.1.0; the `pytorch` env has 1.5.0 and
cannot load the predictors).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

metric = sys.argv[1]

import topcon_experiments.config as cfg  # noqa: E402

cfg.OUTPUT_ROOT = cfg.EXP_ROOT / "outputs" / "exp2_inverse_rerun_20260927"
cfg.IV_TARGETS = [metric]

# Oracle report needs ALL de_best files; run it once at the end instead.
import topcon_experiments.exp2_inverse.oracle_report as oracle_report  # noqa: E402

oracle_report.EXP2_OUT = cfg.OUTPUT_ROOT
oracle_report.main = lambda: None

from topcon_experiments.exp2_inverse import de_single  # noqa: E402

NEW = cfg.OUTPUT_ROOT
NEW.mkdir(parents=True, exist_ok=True)
de_single.EXP2_OUT = NEW

assert cfg.META_JSON.exists(), f"model meta missing: {cfg.META_JSON}"

print(f"[{metric}] OUTPUT_ROOT -> {NEW}", flush=True)
print(f"[{metric}] IV_TARGETS  -> {cfg.IV_TARGETS}", flush=True)
print(f"[{metric}] DE cfg: maxiter={cfg.DE_MAXITER} popsize={cfg.DE_POPSIZE} "
      f"seed={cfg.DE_SEED} mutation={cfg.DE_MUTATION} recomb={cfg.DE_RECOMBINATION}",
      flush=True)

de_single.main()

(NEW / f"_DONE_{metric}").write_text("ok\n", encoding="utf-8")
print(f"[{metric}] DONE", flush=True)
