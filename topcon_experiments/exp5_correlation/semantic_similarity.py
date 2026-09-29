"""Semantic cosine-similarity heatmaps from Ollama embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from topcon_experiments.common.log_utils import log
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import EXP5_OUT
from topcon_experiments.exp5_correlation.llm_texts import TextItem, items_to_dataframe
from topcon_experiments.exp5_correlation.ollama_embedding import embed_texts, load_cache


def _sim_heatmap(
    labels: list[str],
    sim: np.ndarray,
    title: str,
    out_png: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    apply_plot_style()
    n = len(labels)
    figsize = (max(6, n * 0.65), max(5, n * 0.6))
    fig, ax = plt.subplots(figsize=figsize)
    off_diag = sim[~np.eye(n, dtype=bool)] if n > 1 else sim.ravel()
    if vmin is None and off_diag.size:
        vmin = max(0.0, float(np.percentile(off_diag, 5)))
    if vmax is None:
        vmax = 1.0
    sns.heatmap(
        sim,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "cosine similarity"},
    )
    ax.set_title(title, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_semantic_panels(
    items: list[TextItem],
    out_dir: Path | None = None,
) -> list[Path]:
    out_dir = out_dir or EXP5_OUT
    plot_dir = out_dir / "plots"
    csv_dir = out_dir / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    df = items_to_dataframe(items)
    if df.empty:
        log("No LLM texts found for semantic similarity; run exp4 llm_analysis first.")
        return []

    save_csv(df.drop(columns=["text"]), csv_dir / "llm_text_index.csv")
    save_csv(df[["panel", "label", "source", "text_len"]], csv_dir / "llm_text_summary.csv")

    cache = load_cache()
    written: list[Path] = []

    for panel, sub in df.groupby("panel"):
        sub = sub.reset_index(drop=True)
        if len(sub) < 2:
            log(f"  skip panel {panel}: only {len(sub)} text(s)")
            continue
        labels = sub["label"].astype(str).tolist()
        texts = sub["text"].tolist()
        log(f"  semantic panel `{panel}`: {len(labels)} texts")
        embs = embed_texts(texts, cache=cache)
        valid = [(lab, e) for lab, e in zip(labels, embs) if e is not None]
        if len(valid) < 2:
            log(f"  skip panel {panel}: insufficient embeddings")
            continue
        vlabels = [v[0] for v in valid]
        vectors = np.vstack([v[1] for v in valid])
        sim = cosine_similarity(vectors)

        sim_df = pd.DataFrame(sim, index=vlabels, columns=vlabels)
        csv_path = csv_dir / f"semantic_{panel}.csv"
        save_csv(sim_df.reset_index().rename(columns={"index": "label"}), csv_path)
        png_path = plot_dir / f"semantic_{panel}.png"
        atype = str(sub["analysis_type"].iloc[0])
        _sim_heatmap(
            vlabels,
            sim,
            f"LLM semantic similarity ({panel}, {atype})",
            png_path,
        )
        written.extend([csv_path, png_path])

    return written
