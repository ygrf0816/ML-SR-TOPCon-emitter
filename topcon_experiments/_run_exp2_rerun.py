"""Faithful re-run of exp2_inverse DE into a NEW output directory.

Faithful = identical search semantics to the June archive:
  workers=1 (default) -> scipy keeps updating='immediate'
  DE_SEED=42, DE_MAXITER=30, DE_POPSIZE=15, polish=False
  (NOTE: workers != 1 would force updating='deferred' and change the trajectory.)

Only the *recording* differs from the original run: `_flush_generation` now also
stores per-generation n_evals / gen_best / gen_worst / median / p05 / p25 / p75 / p95,
which is what makes a correct (quantile-based) convergence band possible.

Run with the `autogluon` env (AutoGluon 1.1.0 matches the trained predictors;
the `pytorch` env has 1.5.0 and refuses to load them).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import topcon_experiments.config as cfg  # noqa: E402

# Redirect outputs BEFORE the sub-modules capture OUTPUT_ROOT at import time.
cfg.OUTPUT_ROOT = cfg.EXP_ROOT / "outputs" / "exp2_inverse_rerun_20260927"

from topcon_experiments.exp2_inverse import de_single, oracle_report  # noqa: E402

NEW = cfg.OUTPUT_ROOT
NEW.mkdir(parents=True, exist_ok=True)
de_single.EXP2_OUT = NEW
oracle_report.EXP2_OUT = NEW

# Sanity: model paths must still point at the ORIGINAL exp1_forward models.
assert cfg.META_JSON.exists(), f"model meta missing: {cfg.META_JSON}"

print("OUTPUT_ROOT ->", NEW, flush=True)
print("META_JSON   ->", cfg.META_JSON, flush=True)
print("DE cfg: maxiter=%d popsize=%d seed=%s mutation=%s recomb=%s"
      % (cfg.DE_MAXITER, cfg.DE_POPSIZE, cfg.DE_SEED, cfg.DE_MUTATION,
         cfg.DE_RECOMBINATION), flush=True)
print("IV scale enabled:", cfg.DE_IV_SCALE_ENABLED, flush=True)

de_single.main()

(NEW / "_RUN_COMPLETE").write_text("done\n", encoding="utf-8")
print("RUN COMPLETE ->", NEW, flush=True)
