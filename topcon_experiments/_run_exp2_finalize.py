"""Run the oracle/envelope report once, after all 4 metrics finished."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import topcon_experiments.config as cfg  # noqa: E402

cfg.OUTPUT_ROOT = cfg.EXP_ROOT / "outputs" / "exp2_inverse_rerun_20260927"

from topcon_experiments.exp2_inverse import oracle_report  # noqa: E402

NEW = cfg.OUTPUT_ROOT
oracle_report.EXP2_OUT = NEW

missing = [m for m in cfg.IV_TARGETS if not (NEW / f"de_best_{m}.csv").exists()]
if missing:
    print("missing de_best for:", missing, file=sys.stderr)
    sys.exit(1)

oracle_report.main()
(NEW / "_DONE_oracle").write_text("ok\n", encoding="utf-8")
print("ORACLE DONE ->", NEW, flush=True)
