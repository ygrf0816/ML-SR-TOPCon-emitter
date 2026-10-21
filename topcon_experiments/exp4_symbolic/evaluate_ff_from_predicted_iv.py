"""FF from SR-predicted Voc/Jsc/Eff (full features) vs true FF; composite formula substitution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import preprocess_model2
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import MODEL2_FEATURES, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    best_row_by_test_r2,
    predict_from_sr_csv,
    substitute_ff_composite,
)
from topcon_experiments.exp4_symbolic.sr_eval_utils import PREPROCESS_NOTES
from topcon_experiments.exp4_symbolic.sr_split_utils import split_train_test
from topcon_experiments.exp4_symbolic.sr_variable_map import (
    LOG_TRANSFORMED_FEATURES,
    substitute_equation,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
REPORT_PATH = EXP4_OUT / "sr_reports" / "ff_from_predicted_iv.md"
FF_COMPARISON_PATH = EXP4_OUT / "sr_ff_comparison.csv"

IV_PATHS = {
    "iv_Voc": EXP4_OUT / "sr_tabular_full_to_iv_Voc.csv",
    "iv_Jsc": EXP4_OUT / "sr_tabular_full_to_iv_Jsc.csv",
    "iv_Eff": EXP4_OUT / "sr_tabular_full_to_iv_Eff.csv",
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            "R2": float("nan"),
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "pearson_r": float("nan"),
            "n": 0,
        }
    yt, yp = y_true[mask], y_pred[mask]
    r = float(pearsonr(yt, yp)[0]) if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0 else float("nan")
    return {
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "pearson_r": r,
        "n": int(mask.sum()),
    }


def _append_ff_comparison(test_m: dict) -> None:
    row = {
        "method": "FF_from_SR_Voc_Jsc_Eff",
        "train_R2": test_m.get("train_R2"),
        "test_R2": test_m.get("test_R2"),
        "train_RMSE": test_m.get("train_RMSE"),
        "test_RMSE": test_m.get("test_RMSE"),
        "test_MAE": test_m.get("test_MAE"),
        "test_pearson_r": test_m.get("test_pearson_r"),
        "equation": "FF_calc = 100 * Eff_pred / (Voc_pred * Jsc_pred)",
        "preprocess_note": PREPROCESS_NOTES["model2_iv_full"]
        + "；Voc/Jsc/Eff 为各自 test_R² 最优 SR 公式预测值。",
    }
    if FF_COMPARISON_PATH.exists():
        df = pd.read_csv(FF_COMPARISON_PATH)
        df = df[df["method"] != "FF_from_SR_Voc_Jsc_Eff"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    save_csv(df, FF_COMPARISON_PATH)


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)

    for k, p in IV_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} SR results: {p}")

    X, y_map, _ = preprocess_model2()
    Xv = X.values.astype(float)
    ff = y_map["iv_FF"].values.astype(float)
    voc_t = y_map["iv_Voc"].values.astype(float)
    jsc_t = y_map["iv_Jsc"].values.astype(float)
    eff_t = y_map["iv_Eff"].values.astype(float)
    train_idx, test_idx = split_train_test(len(X))

    voc_pred = predict_from_sr_csv(IV_PATHS["iv_Voc"], Xv)
    jsc_pred = predict_from_sr_csv(IV_PATHS["iv_Jsc"], Xv)
    eff_pred = predict_from_sr_csv(IV_PATHS["iv_Eff"], Xv)
    ff_calc = 100.0 * eff_pred / (voc_pred * jsc_pred)

    iv_metrics = []
    for name, pred, true in [
        ("iv_Voc", voc_pred, voc_t),
        ("iv_Jsc", jsc_pred, jsc_t),
        ("iv_Eff", eff_pred, eff_t),
    ]:
        for split, idx in [("train", train_idx), ("test", test_idx)]:
            m = _metrics(true[idx], pred[idx])
            iv_metrics.append({"target": name, "split": split, **m})
            log(f"{name} pred ({split}): R2={m['R2']:.4f}, MAE={m['MAE']:.4f}")

    ablations = {
        "all_SR_pred": ff_calc,
        "true_Voc_Jsc_pred_Eff": 100.0 * eff_pred / (voc_t * jsc_t),
        "pred_Voc_true_Jsc_Eff": 100.0 * eff_t / (voc_pred * jsc_t),
        "pred_Jsc_true_Voc_Eff": 100.0 * eff_t / (voc_t * jsc_pred),
        "pred_Voc_Jsc_true_Eff": 100.0 * eff_t / (voc_pred * jsc_pred),
    }
    ablation_rows = []
    for method, calc in ablations.items():
        for split, idx in [("train", train_idx), ("test", test_idx)]:
            m = _metrics(ff[idx], calc[idx])
            ablation_rows.append({"method": method, "split": split, **m})

    rows = []
    robust_mask = (ff_calc >= 75.0) & (ff_calc <= 95.0)
    for subset, mask in [
        ("all", np.ones(len(X), dtype=bool)),
        ("robust_ff_calc_75_95", robust_mask),
    ]:
        for split, idx in [("train", train_idx), ("test", test_idx)]:
            sel = idx[mask[idx]]
            m = _metrics(ff[sel], ff_calc[sel])
            rows.append({
                "method": "FF_from_SR_Voc_Jsc_Eff",
                "subset": subset,
                "split": split,
                "n_outliers_ff_calc": int((~robust_mask[idx]).sum()) if subset == "all" else 0,
                **m,
            })
            log(
                f"FF from predicted IV ({subset}/{split}): R2={m['R2']:.4f}, "
                f"MAE={m['MAE']:.4f}, r={m['pearson_r']:.4f}, n={m['n']}"
            )

    metrics_df = pd.DataFrame(rows)
    save_csv(metrics_df, EXP4_OUT / "sr_ff_from_predicted_iv_metrics.csv")
    save_csv(pd.DataFrame(iv_metrics), EXP4_OUT / "sr_ff_from_predicted_iv_iv_metrics.csv")
    save_csv(pd.DataFrame(ablation_rows), EXP4_OUT / "sr_ff_from_predicted_iv_ablation.csv")

    detail = pd.DataFrame({
        "iv_Voc_true": voc_t,
        "iv_Jsc_true": jsc_t,
        "iv_Eff_true": eff_t,
        "iv_FF_true": ff,
        "iv_Voc_pred": voc_pred,
        "iv_Jsc_pred": jsc_pred,
        "iv_Eff_pred": eff_pred,
        "iv_FF_calc": ff_calc,
        "ff_calc_in_75_95": robust_mask,
        "split": np.where(np.isin(np.arange(len(X)), test_idx), "test", "train"),
    })
    save_csv(detail, EXP4_OUT / "sr_ff_from_predicted_iv_detail.csv")

    voc_row = best_row_by_test_r2(IV_PATHS["iv_Voc"])
    jsc_row = best_row_by_test_r2(IV_PATHS["iv_Jsc"])
    eff_row = best_row_by_test_r2(IV_PATHS["iv_Eff"])
    voc_s = str(voc_row.get("sympy_format") or voc_row["equation"])
    jsc_s = str(jsc_row.get("sympy_format") or jsc_row["equation"])
    eff_s = str(eff_row.get("sympy_format") or eff_row["equation"])

    composite_raw, composite_simp = substitute_ff_composite(
        voc_s, jsc_s, eff_s, len(MODEL2_FEATURES)
    )
    log_feats = LOG_TRANSFORMED_FEATURES
    phys_voc = substitute_equation(str(voc_row["equation"]), list(MODEL2_FEATURES), log_features=log_feats)
    phys_jsc = substitute_equation(str(jsc_row["equation"]), list(MODEL2_FEATURES), log_features=log_feats)
    phys_eff = substitute_equation(str(eff_row["equation"]), list(MODEL2_FEATURES), log_features=log_feats)

    test_all = metrics_df[(metrics_df["subset"] == "all") & (metrics_df["split"] == "test")].iloc[0]
    test_robust = metrics_df[
        (metrics_df["subset"] == "robust_ff_calc_75_95") & (metrics_df["split"] == "test")
    ].iloc[0]
    train_all = metrics_df[(metrics_df["subset"] == "all") & (metrics_df["split"] == "train")].iloc[0]

    _append_ff_comparison({
        "train_R2": train_all["R2"],
        "test_R2": test_all["R2"],
        "train_RMSE": train_all["RMSE"],
        "test_RMSE": test_all["RMSE"],
        "test_MAE": test_all["MAE"],
        "test_pearson_r": test_all["pearson_r"],
    })

    lines = [
        "# FF 由 SR 预测的 Voc/Jsc/Eff 计算",
        "",
        "流程：用 **全特征 SR 公式**（各自 test_R² 最优）分别预测 Voc、Jsc、Eff，",
        "再代入 `FF_calc = 100 × Eff_pred / (Voc_pred × Jsc_pred)`，与实测 FF 比较。",
        "最后将三个 SR 公式依次代入 FF 恒等式并做 sympy 简化。",
        "",
        "## 各 IV 采用的 SR 公式（test_R² 最优）",
        "",
        f"### Voc (test_R²={voc_row.get('test_R2', 'N/A')})",
        f"```text\n{voc_row['equation']}\n```",
        "",
        f"### Jsc (test_R²={jsc_row.get('test_R2', 'N/A')})",
        f"```text\n{jsc_row['equation']}\n```",
        "",
        f"### Eff (test_R²={eff_row.get('test_R2', 'N/A')})",
        f"```text\n{eff_row['equation']}\n```",
        "",
        "## 各 IV 预测误差（sanity check）",
        "",
        "| target | split | R² | MAE | RMSE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in pd.DataFrame(iv_metrics).iterrows():
        lines.append(
            f"| {r['target']} | {r['split']} | {r['R2']:.4f} | {r['MAE']:.4f} | {r['RMSE']:.4f} |"
        )

    lines += [
        "",
        "## FF 计算误差",
        "",
        "> **说明**：iv_FF 在数据集中分布极窄（约 81–86%），R² 对少量离群点非常敏感；",
        "> 建议同时看 **MAE** 与 **robust 子集**（FF_calc ∈ [75, 95]%）指标。",
        "",
        "| subset | split | R² | Pearson r | RMSE | MAE | n | 离群点数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in metrics_df.iterrows():
        out_n = int(r["n_outliers_ff_calc"]) if r["subset"] == "all" else "—"
        lines.append(
            f"| {r['subset']} | {r['split']} | {r['R2']:.4f} | {r['pearson_r']:.4f} | "
            f"{r['RMSE']:.4f} | {r['MAE']:.4f} | {int(r['n'])} | {out_n} |"
        )

    lines += [
        "",
        f"- **测试集（全样本）**：MAE = {test_all['MAE']:.3f}%，Pearson r = {test_all['pearson_r']:.4f}",
        f"- **测试集（robust 子集，n={int(test_robust['n'])}）**："
        f"MAE = {test_robust['MAE']:.3f}%，Pearson r = {test_robust['pearson_r']:.4f}",
        "",
        "## 误差传播消融（测试集）",
        "",
        "| 方案 | R² | MAE | RMSE | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    abl_labels = {
        "all_SR_pred": "Voc_pred + Jsc_pred + Eff_pred",
        "true_Voc_Jsc_pred_Eff": "Voc_true + Jsc_true + Eff_pred",
        "pred_Voc_true_Jsc_Eff": "Voc_pred + Jsc_true + Eff_true",
        "pred_Jsc_true_Voc_Eff": "Voc_true + Jsc_pred + Eff_true",
        "pred_Voc_Jsc_true_Eff": "Voc_pred + Jsc_pred + Eff_true",
    }
    for _, r in pd.DataFrame(ablation_rows)[pd.DataFrame(ablation_rows)["split"] == "test"].iterrows():
        lines.append(
            f"| {abl_labels.get(r['method'], r['method'])} | {r['R2']:.4f} | "
            f"{r['MAE']:.4f} | {r['RMSE']:.4f} | — |"
        )

    lines += [
        "",
        "Jsc 公式在少数样本上预测偏低，经除法放大后产生 FF_calc 离群点（见 detail CSV 中 `ff_calc_in_75_95=False`）。",
        "",
        "## 代入合并后的 FF 公式",
        "",
        "### 符号编号形式",
        f"```text\nFF = 100 * ({eff_s}) / (({voc_s}) * ({jsc_s}))\n```",
        "",
        "### sympy 简化",
        f"```text\n{composite_simp}\n```",
        "",
        "### 物理变量形式（代入后）",
        f"```text\nFF = 100 * ({phys_eff}) / (({phys_voc}) * ({phys_jsc}))\n```",
        "",
        "> 注：Voc/Jsc/Eff 公式中的 xᵢ 为 model2 特征（见各 IV 报告 §1.1）。",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    save_csv(
        pd.DataFrame([{
            "composite_equation": composite_raw,
            "composite_simplified": composite_simp,
            "voc_equation": voc_row["equation"],
            "jsc_equation": jsc_row["equation"],
            "eff_equation": eff_row["equation"],
        }]),
        EXP4_OUT / "sr_ff_composite_formula.csv",
    )
    log(f"FF-from-predicted-IV report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
