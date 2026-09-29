"""LLM analysis for curve SR and tabular SR formulas (doping/defect/tabular)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_TAIL_DEPTH_MIN,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEPTH_MAX,
    DOPING_DESCRIPTORS,
    IV_TARGETS,
    LLM_INTER_CALL_DELAY,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT,
    LLM_RETRY_BASE_DELAY,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
    SR_DEFECT_CURVE_FORMULA_PATH,
    SR_DOPING_CURVE_FORMULA_PATH,
    SR_TOP_FORMULAS_FOR_LLM,
)

from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
ANALYSIS_TYPES = ("blind_math", "simplify", "physics")
RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError)


def check_api_connectivity() -> None:
    """Fail fast with a clear message when outbound HTTPS to DeepSeek is blocked."""
    try:
        with httpx.Client(timeout=20.0) as http:
            resp = http.get(DEEPSEEK_BASE_URL)
        log(f"API reachable: {DEEPSEEK_BASE_URL} (HTTP {resp.status_code})")
    except httpx.ConnectTimeout as exc:
        raise RuntimeError(
            f"无法连接 DeepSeek API（{DEEPSEEK_BASE_URL}）：TCP 连接超时。"
            "这是服务器出站网络/防火墙问题，与 API 余额、公式数量无关。"
            "请在本机执行: curl -v --connect-timeout 15 https://api.deepseek.com "
            "若同样超时，需联系网管或配置 HTTP_PROXY/HTTPS_PROXY。"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"无法访问 DeepSeek API（{DEEPSEEK_BASE_URL}）: {type(exc).__name__}: {exc}"
        ) from exc


def create_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise EnvironmentError("Set API key in topcon_experiments/ds_api or DEEPSEEK_API_KEY env")
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=0,
    )


def chat(client: OpenAI, messages: list[dict], label: str = "") -> str:
    """Single chat completion with exponential-backoff retries."""
    last_err: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content or ""
            if not content.strip() and resp.choices[0].finish_reason == "length":
                log(
                    f"Warning: empty LLM content ({label or 'chat'}, "
                    f"model={DEEPSEEK_MODEL}, finish_reason=length)"
                )
            if LLM_INTER_CALL_DELAY > 0:
                time.sleep(LLM_INTER_CALL_DELAY)
            return content
        except RETRYABLE_ERRORS as exc:
            last_err = exc
            if attempt >= LLM_MAX_RETRIES:
                break
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            cause = getattr(exc, "__cause__", None)
            cause_msg = f", cause={cause}" if cause else ""
            log(
                f"LLM transient error ({label or 'chat'}): {type(exc).__name__}: {exc}{cause_msg}; "
                f"retry {attempt}/{LLM_MAX_RETRIES} in {delay:.1f}s"
            )
            time.sleep(delay)
        except APIStatusError as exc:
            # 5xx / rate-limit style server errors
            if exc.status_code and exc.status_code >= 500 and attempt < LLM_MAX_RETRIES:
                last_err = exc
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log(
                    f"LLM server error ({label or 'chat'}): HTTP {exc.status_code}; "
                    f"retry {attempt}/{LLM_MAX_RETRIES} in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise
    assert last_err is not None
    raise last_err


def load_top_formulas(path: Path, n: int = SR_TOP_FORMULAS_FOR_LLM) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return sort_formulas_by_accuracy(df).head(n)


def tabular_task_type(target: str) -> str:
    if target.startswith("full_to_"):
        return "tabular_full_to_iv"
    raw = target.replace("athena_to_", "")
    if raw.endswith("_log"):
        raw = raw[:-4]
    if raw in IV_TARGETS:
        return "tabular_athena_to_iv"
    if raw in DOPING_DESCRIPTORS:
        return "tabular_athena_to_descriptor"
    return "tabular"


def var_context_for_task(task_type: str, curve_type: str = "", target: str = "") -> str:
    athena_desc = ", ".join(f"{c} ({label_for(c)})" for c in ATHENA_FEATURES)
    if task_type == "doping_curve_tail_full":
        feat_desc = ", ".join(f"x{i}={c} ({label_for(c)})" for i, c in enumerate(MODEL2_FEATURES))
        return (
            f"Doping concentration curve in tail window [{CURVE_TAIL_DEPTH_MIN}, {DEPTH_MAX}] μm "
            f"(BSG spike below {CURVE_TAIL_DEPTH_MIN} μm excluded). "
            f"Target y = ln(Na). Full model2 inputs (Athena + doping + defect descriptors; "
            f"log applied to concentration-like features in training): {feat_desc}, "
            f"x{len(MODEL2_FEATURES)}=depth_um (μm)."
        )
    if task_type == "curve":
        return f"Curve type: {curve_type}. x0-x7 are athena process params: {athena_desc}. x8/depth is depth (um)."
    if task_type == "tabular_athena_to_iv":
        return f"Predict IV target {target} from athena process params: {athena_desc}."
    if task_type == "tabular_athena_to_descriptor":
        return f"Predict doping curve descriptor {target} from athena process params: {athena_desc}."
    if task_type == "tabular_full_to_iv":
        feat_desc = ", ".join(f"x{i}={c} ({label_for(c)})" for i, c in enumerate(MODEL2_FEATURES))
        return (
            f"Predict IV target {target.replace('full_to_', '')} from full model2 features "
            f"(athena + doping + defect descriptors; log applied to concentrations in training): {feat_desc}."
        )
    return f"Inputs are athena process params: {athena_desc}."


def blind_math(client, equation: str, metrics: dict, task_type: str, source: str, target: str = "") -> dict:
    ctx = var_context_for_task(
        task_type,
        source if task_type in ("curve", "doping_curve_tail_full") else "",
        target,
    )
    prompt = f"""Blind mathematical analysis — do NOT use domain knowledge beyond variable names below.

{ctx}
Formula: {equation}
Metrics: {json.dumps(metrics)}

Infer relationship type and variable roles. Compare inferred roles to provided names.
JSON keys: relationship_type, inferred_variable_roles, matches_provided_names, confidence, notes"""
    label = f"{source}/blind_math"
    return {
        "source": source, "task_type": task_type, "equation": equation,
        "analysis_type": "blind_math", "response": chat(client, [{"role": "user", "content": prompt}], label),
    }


def simplify(client, equation: str, metrics: dict, task_type: str, source: str) -> dict:
    prompt = f"""Simplify this formula. Use symbolic parameters a,b,c for constants.
Original: {equation}
Metrics: {json.dumps(metrics)}
JSON keys: simplified_formula, sympy_form, simplification_notes, expected_accuracy_change"""
    label = f"{source}/simplify"
    return {
        "source": source, "task_type": task_type, "equation": equation,
        "analysis_type": "simplify", "response": chat(client, [{"role": "user", "content": prompt}], label),
    }


def physics(client, equation: str, metrics: dict, task_type: str, source: str, target: str = "") -> dict:
    ctx = var_context_for_task(
        task_type,
        source if task_type in ("curve", "doping_curve_tail_full") else "",
        target,
    )
    prompt = f"""TOPCon boron diffusion / TOPCon cell physics context.
{ctx}
Target/phenomenon: {target or source}
Formula: {equation}
Metrics: {json.dumps(metrics)}
JSON keys: overall_physics, term_interpretations, plausibility_assessment"""
    label = f"{source}/physics"
    return {
        "source": source, "task_type": task_type, "target": target, "equation": equation,
        "analysis_type": "physics", "response": chat(client, [{"role": "user", "content": prompt}], label),
    }


def _output_paths(source: str, tabular: bool) -> dict[str, Path]:
    if tabular:
        return {
            "blind_math": EXP4_OUT / f"llm_blind_math_tabular_{source}.csv",
            "simplify": EXP4_OUT / f"llm_simplified_tabular_{source}.csv",
            "physics": EXP4_OUT / f"llm_physics_tabular_{source}.csv",
        }
    return {
        "blind_math": EXP4_OUT / f"llm_blind_math_{source}.csv",
        "simplify": EXP4_OUT / f"llm_simplified_{source}.csv",
        "physics": EXP4_OUT / f"llm_physics_{source}.csv",
    }


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _equations_with_analysis(df: pd.DataFrame) -> set[str]:
    if df.empty or "equation" not in df.columns:
        return set()
    return set(df["equation"].astype(str))


def analyze_formula_rows(
    client,
    df: pd.DataFrame,
    source: str,
    task_type: str,
    target: str = "",
    *,
    tabular: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _output_paths(source, tabular)
    existing = {k: _load_existing(p) for k, p in paths.items()}
    done_by_type = {k: _equations_with_analysis(df) for k, df in existing.items()}

    blind_frames = [existing["blind_math"]] if not existing["blind_math"].empty else []
    simp_frames = [existing["simplify"]] if not existing["simplify"].empty else []
    phys_frames = [existing["physics"]] if not existing["physics"].empty else []

    n_total = len(df)
    pending = sum(
        1
        for _, row in df.iterrows()
        if any(str(row.get("equation", "")) not in done_by_type[t] for t in ANALYSIS_TYPES)
    )
    log(
        f"  task {source}: {n_total} formulas, ~{pending * len(ANALYSIS_TYPES)} pending API calls "
        f"(3 per formula, one call at a time)"
    )

    call_i = 0
    for fi, (_, row) in enumerate(df.iterrows(), start=1):
        eq = str(row.get("equation", ""))
        pending_types = [t for t in ANALYSIS_TYPES if eq not in done_by_type[t]]
        if not pending_types:
            log(f"  [{fi}/{n_total}] skip cached: {eq[:60]}...")
            continue

        metrics = {
            k: row[k]
            for k in row.index
            if k in [
                "loss", "complexity", "score", "dataset_R2",
                "train_MSE", "test_MSE", "train_MAE", "test_MAE",
                "train_R2", "test_R2",
            ]
            and pd.notna(row[k])
        }
        log(f"  [{fi}/{n_total}] analyzing ({', '.join(pending_types)}), loss={metrics.get('loss', 'n/a')}")

        if "blind_math" in pending_types:
            call_i += 1
            log(f"    API call {call_i}: blind_math")
            blind_frames.append(pd.DataFrame([blind_math(client, eq, metrics, task_type, source, target)]))
            save_csv(pd.concat(blind_frames, ignore_index=True), paths["blind_math"])
            done_by_type["blind_math"].add(eq)

        if "simplify" in pending_types:
            call_i += 1
            log(f"    API call {call_i}: simplify")
            simp_frames.append(pd.DataFrame([simplify(client, eq, metrics, task_type, source)]))
            save_csv(pd.concat(simp_frames, ignore_index=True), paths["simplify"])
            done_by_type["simplify"].add(eq)

        if "physics" in pending_types:
            call_i += 1
            log(f"    API call {call_i}: physics")
            phys_frames.append(pd.DataFrame([physics(client, eq, metrics, task_type, source, target)]))
            save_csv(pd.concat(phys_frames, ignore_index=True), paths["physics"])
            done_by_type["physics"].add(eq)

    blind = pd.concat(blind_frames, ignore_index=True) if blind_frames else pd.DataFrame()
    simp = pd.concat(simp_frames, ignore_index=True) if simp_frames else pd.DataFrame()
    phys = pd.concat(phys_frames, ignore_index=True) if phys_frames else pd.DataFrame()
    return blind, simp, phys


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    check_api_connectivity()
    client = create_client()
    log(
        f"Starting LLM formula analysis (model={DEEPSEEK_MODEL}, "
        f"top_n={SR_TOP_FORMULAS_FOR_LLM}, timeout={LLM_REQUEST_TIMEOUT}s, "
        f"retries={LLM_MAX_RETRIES})..."
    )

    jobs = [
        ("doping", SR_DOPING_CURVE_FORMULA_PATH, "doping_curve_tail_full", "doping_curve", False),
        ("defect", SR_DEFECT_CURVE_FORMULA_PATH, "curve", "defect_vac_curve", False),
    ]
    for i, (src, path, ttype, tgt, tabular) in enumerate(jobs, start=1):
        df = load_top_formulas(path)
        if df.empty:
            log(f"Skip LLM {src}: no formulas at {path}")
            continue
        log_step(i, len(jobs), f"LLM curve {src} ({path.name})")
        analyze_formula_rows(client, df, src, ttype, tgt, tabular=tabular)

    skip_tabular = {
        "sr_tabular_formulas_summary.csv",
        "sr_tabular_full_iv_summary.csv",
    }
    tabular_paths = sorted(
        p for p in EXP4_OUT.glob("sr_tabular_*.csv") if p.name not in skip_tabular
    )
    for j, path in enumerate(tabular_paths, start=1):
        target = path.stem.replace("sr_tabular_", "")
        df = load_top_formulas(path)
        if df.empty:
            continue
        ttype = tabular_task_type(target)
        log_step(j, len(tabular_paths), f"LLM tabular {target} ({ttype})")
        analyze_formula_rows(client, df, target, ttype, target, tabular=True)

    all_blind = sorted(EXP4_OUT.glob("llm_blind_math*.csv"))
    all_blind = [f for f in all_blind if f.name != "llm_blind_math_all.csv"]
    if all_blind:
        combined = pd.concat([pd.read_csv(f) for f in all_blind], ignore_index=True)
        save_csv(combined, EXP4_OUT / "llm_blind_math_all.csv")

    log(f"LLM outputs -> {EXP4_OUT}")

    from topcon_experiments.exp4_symbolic.generate_sr_report import main as report_main
    report_main()


if __name__ == "__main__":
    main()
