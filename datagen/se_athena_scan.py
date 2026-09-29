"""
se_athena_scan.py

基于 se.in 模板批量运行 SE（选择性发射极）Athena 工艺仿真。

功能：
- 对 8 个核心工艺参数（P1~P8）随机采样
- 读取 se.in，覆盖变量后逐条发送给 athena.exe
- 输出 .str 结构文件到指定目录
- 记录工艺参数与提取指标（Rs_light / Xj_light 等）到 JSONL
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import string
import time
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional
import concurrent.futures as futures

import numpy as np

from python_tcad import pythontcad

SCRIPT_DIR = Path(__file__).resolve().parent
SE_TEMPLATE = SCRIPT_DIR / "se.in"

SAMPLING_MODE = os.environ.get("SE_SAMPLING_MODE", "continuous").strip().lower()
RANDOM_SEED = os.environ.get("SE_RANDOM_SEED", "").strip()
if RANDOM_SEED:
    random.seed(int(RANDOM_SEED))

# 工艺指标名（与 se.in 中 extract name 一致）
METRIC_NAMES = (
    "Rs_light",
    "Rs_heavy",
    "Xj_light",
    "Xj_heavy",
    "Cs_light",
    "Cs_heavy",
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rand_name(prefix: str = "SE_Scan_") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    suf = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}{ts}_{suf}"


def loguniform(start: float, stop: float) -> float:
    log_low = math.log10(start)
    log_high = math.log10(stop)
    return 10 ** random.uniform(log_low, log_high)


def sample_se_params() -> Dict[str, float]:
    """采样 SE 核心工艺参数（14 个）。"""
    return {
        "sub_resis": round(random.uniform(0.5, 5.0), 4),
        "light_dose": loguniform(3e14, 2e15),
        "light_energy": round(random.uniform(10, 60), 4),
        "light_drive_time": round(random.uniform(5, 25), 4),
        "light_drive_temp": round(random.uniform(850, 950), 4),
        "light_oxide_thick": round(random.uniform(0.005, 0.02), 5),
        "heavy_dose": loguniform(5e14, 1e16),
        "heavy_energy": round(random.uniform(10, 60), 4),
        "mask_oxide_thick": round(random.uniform(0.1, 0.3), 4),
        "select_boundary_x": round(random.uniform(14.0, 16.0), 4),
        "anneal_temp": round(random.uniform(850, 980), 4),
        "anneal_time": round(random.uniform(5, 30), 4),
        "repair_temp": round(random.uniform(800, 900), 4),
        "repair_time": round(random.uniform(1, 10), 4),
    }


def _format_var_value(key: str, value: float) -> str:
    if key in ("light_dose", "heavy_dose"):
        return f"{value:.6e}"
    if isinstance(value, float) and value == int(value) and abs(value) >= 100:
        return str(int(value))
    return str(value)


def _substitute_commands(commands: List[str], variables: Dict[str, str]) -> List[str]:
    out: List[str] = []
    sorted_vars = sorted(variables.items(), key=lambda kv: len(kv[0]), reverse=True)
    for raw in commands:
        cmd = raw
        for name, val in sorted_vars:
            for pat in (f"${name}", f"$'{name}'", f'$"{name}"'):
                cmd = re.sub(re.escape(pat), val, cmd, flags=re.IGNORECASE)
        out.append(cmd)
    return out


def _filter_commands(commands: List[str], outfile_basename: str) -> List[str]:
    """跳过 go 指令与中间步骤结构保存，仅保留最终 outfile。"""
    filtered: List[str] = []
    for cmd in commands:
        low = cmd.lower().strip()
        if low.startswith("go athena") or low.startswith("go atlas"):
            continue
        if low.startswith("structure outfile="):
            # 替换前命令里是 $outfile_name；替换后才是具体文件名
            if "$outfile_name" in low or outfile_basename.lower() in low:
                filtered.append(cmd)
            continue
        filtered.append(cmd)
    return filtered


def _parse_metrics(output_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name in METRIC_NAMES:
        # 匹配: Rs_light= 123.4  或  extract ... Rs_light 123.4
        patterns = [
            rf"{name}\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
            rf"{name}\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, output_text, flags=re.IGNORECASE)
            if m:
                try:
                    metrics[name] = float(m.group(1))
                    break
                except ValueError:
                    pass
    return metrics


def run_one(
    exe_dir: str,
    out_dir: str,
    template_path: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    base = rand_name()
    outfile_basename = f"{base}.str"
    outstr = os.path.join(out_dir, outfile_basename)

    params = sample_se_params()

    tpl = template_path or str(SE_TEMPLATE)
    if not os.path.isfile(tpl):
        raise FileNotFoundError(f"找不到 SE 模板: {tpl}")

    a = pythontcad()
    if exe_dir:
        a.set_tcad(exe_dir)
    a.read_in(tpl)

    # 覆盖采样参数 + 输出文件名（使用绝对路径，避免 cwd 影响）
    a.variables["outfile_name"] = outstr.replace("\\", "/")
    for k, v in params.items():
        a.variables[k] = _format_var_value(k, float(v))

    commands = _filter_commands(a.commands, outfile_basename)
    commands = _substitute_commands(commands, a.variables)

    a.updateprocess()
    a.run_athena_start(output_state=False)
    all_output = ""
    try:
        for cmd in commands:
            chunk = a.run_athena_command(cmd, output_state=False)
            if chunk:
                all_output += chunk
        a.run_athena_command("quit", output_state=False)
    finally:
        for proc_name in ("athena_process", "atlas_process"):
            proc = getattr(a, proc_name, None)
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    metrics = _parse_metrics(all_output)
    status = "ok" if os.path.isfile(outstr) else "error: str not found"

    # 用 cutline 从 .str 补全 O1~O6（Athena 独立模式 extract 无效）
    if status == "ok":
        try:
            from cutline_extract import extract_composite_from_str

            cut = extract_composite_from_str(Path(outstr))
            for k, v in cut["metrics"].items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    metrics[k] = float(v)
        except Exception as e:
            status = f"ok (cutline metrics failed: {e})"

    return {
        "file_base": base,
        "outstr": outstr,
        "params": params,
        "metrics": metrics,
        "ts": int(time.time()),
        "status": status,
    }


def _worker_entry(exe_dir: str, out_dir: str) -> Dict[str, Any]:
    try:
        return run_one(exe_dir, out_dir)
    except Exception as e:
        base = rand_name()
        return {
            "file_base": base,
            "outstr": os.path.join(out_dir, f"{base}.str"),
            "params": {},
            "metrics": {},
            "ts": int(time.time()),
            "status": f"error: {e}",
        }


def run_scan(
    runs: int = 100,
    out_dir: str | None = None,
    exe_dir: str = r"C:\sedatools\exe",
    jsonl: str | None = None,
    parallel: bool = False,
    workers: int = max(1, (os.cpu_count() or 1) // 2),
) -> None:
    """
    批量运行 SE Athena 扫描。

    参数:
        runs: 仿真次数
        out_dir: .str 输出目录，默认 <项目>/saomiao3
        exe_dir: TCAD 可执行文件目录（含 athena.exe）
        jsonl: 记录文件路径，默认 <out_dir>/se_scan.jsonl
        parallel: 是否并行（每个 worker 独立启动 athena 进程）
        workers: 并行进程数
    """
    if out_dir is None:
        out_dir = str(SCRIPT_DIR / "saomiao3")
    jsonl_path = jsonl or os.path.join(out_dir, "se_scan.jsonl")
    ensure_dir(out_dir)

    print(f"[INFO] SE 扫描: runs={runs}, out_dir={out_dir}, exe_dir={exe_dir}")
    print(f"[INFO] 模板: {SE_TEMPLATE}")

    if parallel and workers > 1:
        with futures.ProcessPoolExecutor(max_workers=workers) as ex, open(
            jsonl_path, "a", encoding="utf-8"
        ) as f:
            for rec in ex.map(
                _worker_entry,
                repeat(exe_dir, runs),
                repeat(out_dir, runs),
                chunksize=1,
            ):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    else:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for i in range(max(1, int(runs))):
                print(f"[{i + 1}/{runs}] 运行 SE Athena ...")
                rec = _worker_entry(exe_dir, out_dir)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  -> {rec.get('status')}  {rec.get('file_base')}")

    print(f"[OK] SE 扫描完成，记录: {jsonl_path}")


def main() -> None:
    exe = os.environ.get("TCAD_EXE_DIR", r"C:\sedatools\exe")
    out = os.environ.get("SE_OUT_DIR", str(SCRIPT_DIR / "saomiao3"))
    runs = int(os.environ.get("SE_SCAN_RUNS", "5"))
    parallel = os.environ.get("SE_SCAN_PARALLEL", "0").strip() in ("1", "true", "yes")
    workers = int(os.environ.get("SE_SCAN_WORKERS", str(max(1, (os.cpu_count() or 1) // 2))))
    run_scan(
        runs=runs,
        out_dir=out,
        exe_dir=exe,
        parallel=parallel,
        workers=workers,
    )


if __name__ == "__main__":
    main()
