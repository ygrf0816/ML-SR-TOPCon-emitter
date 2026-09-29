"""Reorder existing tier_shap beeswarm CSV columns by mean |SHAP| (beeswarm y-axis order)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import OUTPUT_ROOT
from topcon_experiments.exp3_classifier.shap_analysis import (
    beeswarm_columns_ordered,
    feature_order_by_shap,
)

TIER_SHAP = OUTPUT_ROOT / "exp3_classifier" / "tier_shap"


def reorder_one(path: Path) -> None:
    df = pd.read_csv(path)
    feature_cols = [
        c
        for c in df.columns
        if not c.startswith("shap_") and not c.startswith("label_") and c != "eff_tier"
    ]
    if not feature_cols:
        log(f"Skip {path.name}: no feature columns")
        return
    shap_mat = np.column_stack([df[f"shap_{c}"].values for c in feature_cols])
    order = feature_order_by_shap(shap_mat, feature_cols)
    out = df[beeswarm_columns_ordered(order)]
    save_csv(out, path)
    meta = pd.DataFrame({
        "rank": range(1, len(order) + 1),
        "feature": order,
        "label": [label_for(c) for c in order],
        "mean_abs_shap": [float(df[f"shap_{c}"].abs().mean()) for c in order],
        "value_column": order,
        "shap_column": [f"shap_{c}" for c in order],
    })
    save_csv(meta, path.with_name(path.stem + "_column_order.csv"))
    log(f"Reordered {path.name} -> {', '.join(order[:3])} ...")


def main() -> None:
    setup_runtime()
    for path in sorted(TIER_SHAP.glob("shap_beeswarm_*.csv")):
        if path.name.endswith("_column_order.csv"):
            continue
        reorder_one(path)


if __name__ == "__main__":
    main()
