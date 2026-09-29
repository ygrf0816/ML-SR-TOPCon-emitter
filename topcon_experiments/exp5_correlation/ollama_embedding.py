"""Ollama local embedding client with JSON cache."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from topcon_experiments.common.log_utils import log
from topcon_experiments.config import (
    EXP5_OUT,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_TIMEOUT,
    OLLAMA_EMBEDDING_MODEL,
)

CACHE_PATH = EXP5_OUT / "embedding_cache.json"
MAX_TEXT_CHARS = 12_000


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log(f"Warning: could not read embedding cache: {exc}")
        return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def check_ollama_embedding(model: str | None = None) -> None:
    """Fail fast if Ollama is unreachable or model is missing."""
    model = model or OLLAMA_EMBEDDING_MODEL
    url = OLLAMA_BASE_URL.rstrip("/")
    try:
        with httpx.Client(timeout=20.0) as client:
            tags = client.get(f"{url}/api/tags")
            tags.raise_for_status()
            names = {m.get("name", "") for m in tags.json().get("models", [])}
            if model not in names and f"{model}:latest" not in names:
                raise RuntimeError(
                    f"Ollama 未找到模型 `{model}`。请先执行: ollama pull {model}"
                )
            resp = client.post(
                f"{url}/api/embed",
                json={"model": model, "input": "ping"},
                timeout=OLLAMA_EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embeddings") or data.get("embedding")
            if emb is None:
                raise RuntimeError(f"Ollama /api/embed 响应无 embedding 字段: {list(data.keys())}")
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"无法连接 Ollama（{url}）。请确认 ollama serve 已启动。"
        ) from exc


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    cache: dict[str, list[float]] | None = None,
) -> list[np.ndarray | None]:
    """Embed texts; returns None for empty inputs. Uses cache keyed by text hash."""
    model = model or OLLAMA_EMBEDDING_MODEL
    url = OLLAMA_BASE_URL.rstrip("/")
    cache = cache if cache is not None else load_cache()
    dirty = False

    results: list[np.ndarray | None] = [None] * len(texts)
    pending_idx: list[int] = []
    pending_texts: list[str] = []

    for i, raw in enumerate(texts):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        text = str(raw).strip()[:MAX_TEXT_CHARS]
        if not text:
            continue
        key = _text_key(text)
        if key in cache:
            results[i] = np.asarray(cache[key], dtype=float)
        else:
            pending_idx.append(i)
            pending_texts.append(text)

    if pending_texts:
        log(f"  Ollama embed: {len(pending_texts)} new texts (model={model})")
        with httpx.Client(timeout=OLLAMA_EMBED_TIMEOUT) as client:
            for batch_start in range(0, len(pending_texts), 8):
                batch_texts = pending_texts[batch_start : batch_start + 8]
                batch_idx = pending_idx[batch_start : batch_start + 8]
                for attempt in range(3):
                    try:
                        resp = client.post(
                            f"{url}/api/embed",
                            json={"model": model, "input": batch_texts},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        embs = data.get("embeddings")
                        if embs is None and "embedding" in data:
                            embs = [data["embedding"]]
                        if not embs or len(embs) != len(batch_texts):
                            raise ValueError(f"unexpected embed count: {len(embs or [])}")
                        break
                    except (httpx.HTTPError, ValueError) as exc:
                        if attempt >= 2:
                            raise
                        delay = 2.0 * (attempt + 1)
                        log(f"  Ollama embed retry in {delay:.0f}s: {exc}")
                        time.sleep(delay)
                else:
                    embs = []

                for j, vec in enumerate(embs):
                    arr = np.asarray(vec, dtype=float)
                    results[batch_idx[j]] = arr
                    cache[_text_key(batch_texts[j])] = arr.tolist()
                    dirty = True

    if dirty:
        save_cache(cache)
    return results
