"""Report dataset oracle vs surrogate inverse design envelope."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.forward import load_forward_chain
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, IV_TARGETS, META_JSON, OUTPUT_ROOT

EXP2_OUT = OUTPUT_ROOT / "exp2_inverse"
TARGET_EFF_PCT = 28.0


def _segment_stats(df: pd.DataFrame, name: str) -> dict:
    anchor = int(df["iv_Eff"].max() // 1)
    band = df[(df["iv_Eff"] >= anchor - 1) & (df["iv_Eff"] < anchor)]
    return {
        "segment": name,
        "count": len(df),
        "iv_Eff_min": float(df["iv_Eff"].min()),
        "iv_Eff_mean": float(df["iv_Eff"].mean()),
        "iv_Eff_max": float(df["iv_Eff"].max()),
        "iv_Eff_p99": float(df["iv_Eff"].quantile(0.99)),
    }


def main() -> None:
    EXP2_OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataframe()
    chain = load_forward_chain(META_JSON)

    rows = []
    top10 = df.nlargest(10, "iv_Eff")
    top50 = df.nlargest(50, "iv_Eff")
    anchor = int(df["iv_Eff"].max() // 1)
    band_25_26 = df[(df["iv_Eff"] >= anchor - 1) & (df["iv_Eff"] < anchor)]

    for _, row in top10.iterrows():
        r = {"source": "dataset_oracle", "file_base": row["file_base"]}
        for c in ATHENA_FEATURES + IV_TARGETS:
            r[c] = row[c]
        rows.append(r)

    for _, row in top10.iterrows():
        x = row[ATHENA_FEATURES].values.astype(float)
        pred = chain.predict_all(x)
        r = {"source": "surrogate_replay", "file_base": row["file_base"]}
        for c in ATHENA_FEATURES:
            r[c] = row[c]
        for k, v in pred.items():
            r[k] = v
        rows.append(r)

    de_best_eff = None
    for metric in IV_TARGETS:
        p = EXP2_OUT / f"de_best_{metric}.csv"
        if p.exists():
            d = pd.read_csv(p)
            r = d.iloc[0].to_dict()
            r["source"] = f"de_optimize_{metric}"
            rows.append(r)
            if metric == "iv_Eff":
                de_best_eff = float(r["iv_Eff"])

    out = pd.DataFrame(rows)
    save_csv(out, EXP2_OUT / "inverse_design_report.csv")

    segments = [
        _segment_stats(df, "full_dataset"),
        _segment_stats(top10, "top_10"),
        _segment_stats(top50, "top_50"),
        _segment_stats(band_25_26, "band_25_26_pct"),
    ]
    save_csv(pd.DataFrame(segments), EXP2_OUT / "inverse_design_envelope.csv")

    dataset_max = float(df["iv_Eff"].max())
    feasibility = pd.DataFrame([{
        "iv_Eff_unit": "percent_PCE",
        "dataset_iv_Eff_max": dataset_max,
        "dataset_iv_Eff_p99": float(df["iv_Eff"].quantile(0.99)),
        "dataset_iv_Eff_p95": float(df["iv_Eff"].quantile(0.95)),
        "de_iv_Eff_best": de_best_eff,
        "target_iv_Eff_pct": TARGET_EFF_PCT,
        "gap_to_target_pct": TARGET_EFF_PCT - (de_best_eff if de_best_eff is not None else dataset_max),
        "feasible_in_dataset": dataset_max >= TARGET_EFF_PCT,
        "note": (
            f"iv_Eff is PCE in %. TCAD dataset max={dataset_max:.2f}%; "
            f"{TARGET_EFF_PCT}% is outside the current simulation envelope. "
            "Surrogate optimization cannot reliably extrapolate beyond training data."
        ),
    }])
    save_csv(feasibility, EXP2_OUT / "inverse_design_feasibility.csv")

    print(feasibility.to_string(index=False))
    print(f"Report -> {EXP2_OUT / 'inverse_design_report.csv'}")


if __name__ == "__main__":
    main()
