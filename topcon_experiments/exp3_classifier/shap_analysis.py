"""SHAP beeswarm per efficiency tier + dependence for highest-efficiency tier only."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_meta, load_raw_dataframe, preprocess_model2
from topcon_experiments.common.inverse_utils import get_efficiency_tiers
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import OUTPUT_ROOT, SHAP_TOP_FEATURES

EXP3_OUT = OUTPUT_ROOT / "exp3_classifier"
TIER_META = EXP3_OUT / "tier_classifier_meta.json"


def get_tree_model(predictor: TabularPredictor):
    lb = predictor.leaderboard(silent=True)
    skip = ("WeightedEnsemble", "Stacker")
    for model_name in lb["model"]:
        if any(model_name.startswith(s) for s in skip):
            continue
        wrapper = predictor._trainer.load_model(model_name)
        inner = getattr(wrapper, "model", None)
        if inner is not None:
            return inner
    raise RuntimeError("No tree model with native .model found for SHAP")


def safe_tier_name(label: str) -> str:
    return label.replace("%", "pct").replace("~", "_").replace(">=", "ge").replace("<", "lt").replace("/", "_")


def stack_multiclass_shap(shap_values) -> np.ndarray:
    """Return SHAP array with shape (n_samples, n_features, n_classes)."""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=-1)
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def class_index_for_tier(predictor: TabularPredictor, tier_label: str) -> int | None:
    labels = list(predictor.class_labels)
    if tier_label in labels:
        return labels.index(tier_label)
    return None


def shap_for_tier(
    predictor: TabularPredictor,
    shap_all: np.ndarray,
    tier: dict,
    X: pd.DataFrame,
    X_all: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    idx = tier["mask"].loc[X.index].values
    pos = np.where(idx)[0]
    X_sub = X_all.iloc[pos].reset_index(drop=True)

    class_idx = class_index_for_tier(predictor, tier["label"])
    if class_idx is not None and class_idx < shap_all.shape[-1]:
        shap_sub = shap_all[pos, :, class_idx]
    else:
        log(f"  tier {tier['label']}: class not in model, using mean SHAP across classes")
        shap_sub = shap_all[pos].mean(axis=-1)
    return X_sub, shap_sub


def feature_order_by_shap(shap_sub: np.ndarray, columns: list[str]) -> list[str]:
    """Order features by mean |SHAP| descending (matches beeswarm y-axis: top = most important)."""
    mean_abs = np.abs(shap_sub).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1]
    return [columns[i] for i in idx]


def beeswarm_columns_ordered(feature_order: list[str]) -> list[str]:
    """Interleave value + SHAP columns left-to-right by beeswarm importance."""
    cols: list[str] = []
    for feat in feature_order:
        cols.extend([feat, f"shap_{feat}"])
    cols.append("eff_tier")
    return cols


def build_beeswarm_dataframe(
    tier_label: str,
    X_sub: pd.DataFrame,
    shap_sub: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    order = feature_order_by_shap(shap_sub, list(X_sub.columns))
    beeswarm_df = X_sub.copy()
    for i, col in enumerate(X_sub.columns):
        beeswarm_df[f"shap_{col}"] = shap_sub[:, i]
    beeswarm_df["eff_tier"] = tier_label
    ordered = beeswarm_df[beeswarm_columns_ordered(order)]

    meta = pd.DataFrame({
        "rank": range(1, len(order) + 1),
        "feature": order,
        "label": [label_for(c) for c in order],
        "mean_abs_shap": [float(np.abs(shap_sub[:, list(X_sub.columns).index(c)]).mean()) for c in order],
        "mean_shap": [float(shap_sub[:, list(X_sub.columns).index(c)].mean()) for c in order],
        "min_shap": [float(shap_sub[:, list(X_sub.columns).index(c)].min()) for c in order],
        "max_shap": [float(shap_sub[:, list(X_sub.columns).index(c)].max()) for c in order],
        "value_column": order,
        "shap_column": [f"shap_{c}" for c in order],
    })
    return ordered, meta, order


def plot_beeswarm_for_tier(
    tier_label: str,
    X_sub: pd.DataFrame,
    shap_sub: np.ndarray,
    out_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_tier_name(tier_label)

    beeswarm_df, meta_df, order = build_beeswarm_dataframe(tier_label, X_sub, shap_sub)
    save_csv(beeswarm_df, out_dir / f"shap_beeswarm_{safe}.csv")
    save_csv(meta_df, out_dir / f"shap_beeswarm_{safe}_column_order.csv")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_sub, X_sub, show=False)
    plt.title(f"SHAP beeswarm — efficiency tier {tier_label}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_beeswarm_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return order[:SHAP_TOP_FEATURES]


def plot_pairwise_dependence(
    tier_label: str,
    X_sub: pd.DataFrame,
    shap_sub: np.ndarray,
    top_features: list[str],
    out_dir: Path,
) -> None:
    """SHAP interaction dependence: x=feature_i, y=SHAP_i, color=feature_j value."""
    safe = safe_tier_name(tier_label)
    dep_dir = out_dir / f"dependence_{safe}"
    dep_dir.mkdir(parents=True, exist_ok=True)
    n_plots = len(top_features) ** 2

    for n, (feat_i, feat_j) in enumerate(
        ((fi, fj) for fi in top_features for fj in top_features),
        start=1,
    ):
        idx_i = list(X_sub.columns).index(feat_i)
        idx_j = list(X_sub.columns).index(feat_j)

        dep_df = pd.DataFrame({
            "eff_tier": tier_label,
            "primary_feature_column": feat_i,
            "primary_feature_label": label_for(feat_i),
            "interaction_feature_column": feat_j,
            "interaction_feature_label": label_for(feat_j),
            "primary_value": X_sub[feat_i].values,
            "interaction_value": X_sub[feat_j].values,
            "shap_primary": shap_sub[:, idx_i],
            "shap_interaction": shap_sub[:, idx_j],
            # legacy aliases for downstream scripts
            "feature_x_column": feat_i,
            "feature_y_column": feat_j,
            "x_value": X_sub[feat_i].values,
            "y_value": X_sub[feat_j].values,
            "shap_x": shap_sub[:, idx_i],
            "shap_y": shap_sub[:, idx_j],
        })
        fname = f"shap_dep_{safe}_{feat_i}_vs_{feat_j}"
        save_csv(dep_df, dep_dir / f"{fname}.csv")

        fig, ax = plt.subplots(figsize=(6.5, 5))
        shap_i = dep_df["shap_primary"].values
        vmin = float(np.percentile(shap_i, 2))
        vmax = float(np.percentile(shap_i, 98))
        if vmin >= 0:
            vmin = float(shap_i.min())
        if vmax <= 0:
            vmax = float(shap_i.max())
        sc = ax.scatter(
            dep_df["primary_value"],
            dep_df["shap_primary"],
            c=dep_df["interaction_value"],
            cmap="coolwarm",
            alpha=0.55,
            s=16,
            edgecolors="none",
        )
        ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        plt.colorbar(sc, ax=ax, label=label_for(feat_j))
        ax.set_xlabel(label_for(feat_i))
        ax.set_ylabel(f"SHAP ({label_for(feat_i)})")
        ax.set_title(f"Tier {tier_label}: SHAP({feat_i}) colored by {feat_j}")
        fig.tight_layout()
        fig.savefig(dep_dir / f"{fname}.png", dpi=120)
        plt.close(fig)
        log_step(n, n_plots, f"dependence {tier_label}: SHAP({feat_i}) | color={feat_j}")


def cleanup_legacy_outputs() -> None:
    for pattern in ("shap_dependence_*.csv", "shap_dependence_*.png", "shap_beeswarm.csv"):
        for path in EXP3_OUT.glob(pattern):
            path.unlink(missing_ok=True)
            log(f"Removed legacy output: {path.name}")


def main() -> None:
    setup_runtime()
    EXP3_OUT.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_outputs()

    log("Loading tier classifier and data...")
    meta = load_meta(TIER_META)
    X, _, _ = preprocess_model2()
    raw = load_raw_dataframe().loc[X.index]

    predictor = TabularPredictor.load(meta["model_dir"])
    log("Extracting tree model for SHAP...")
    model = get_tree_model(predictor)

    X_all = X.reset_index(drop=True)
    log(f"Computing SHAP values for {len(X_all)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_all = stack_multiclass_shap(explainer.shap_values(X_all))
    log("SHAP computation done.")

    tiers = get_efficiency_tiers(raw)
    highest_tier = tiers[0]
    top5_for_dep: list[str] | None = None

    for i, t in enumerate(tiers, start=1):
        if t["count"] < 10:
            log(f"Skip beeswarm tier {t['label']}: only {t['count']} samples")
            continue
        log_step(i, len(tiers), f"beeswarm tier {t['label']} ({t['count']} samples)")
        X_sub, shap_sub = shap_for_tier(predictor, shap_all, t, X, X_all)
        top5 = plot_beeswarm_for_tier(t["label"], X_sub, shap_sub, EXP3_OUT / "tier_shap")
        if t["tier_id"] == 0:
            top5_for_dep = top5

    if highest_tier["count"] < 2:
        log(f"Skip dependence: highest tier {highest_tier['label']} has only {highest_tier['count']} samples")
    else:
        log(f"Dependence plots: highest-efficiency tier only ({highest_tier['label']}, n={highest_tier['count']})")
        X_hi, shap_hi = shap_for_tier(predictor, shap_all, highest_tier, X, X_all)
        if top5_for_dep is None:
            mean_abs = np.abs(shap_hi).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:SHAP_TOP_FEATURES]
            top5_for_dep = [X_hi.columns[j] for j in top_idx]
        plot_pairwise_dependence(
            highest_tier["label"], X_hi, shap_hi, top5_for_dep, EXP3_OUT / "tier_shap",
        )

    log(f"Tier SHAP saved to {EXP3_OUT / 'tier_shap'}")


if __name__ == "__main__":
    main()
