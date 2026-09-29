"""
cutline_extract.py

从 Athena 2D .str 结构文件中用 cutline 方法提取 1D 剖面：
1) 在指定 x 位置做竖直 cutline（轻掺 / 重掺）
2) 合并为一条同时覆盖轻掺+重掺的复合掺杂曲线
3) 从 cutline 计算工艺指标（Cs / Xj / Rs 近似）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 默认与 se.in 中 x_light_extract / x_heavy_extract 一致
DEFAULT_X_LIGHT = 50.0
DEFAULT_X_HEAVY = 7.5
DEFAULT_Y_MIN = 0.0
DEFAULT_Y_MAX = 5.0
JUNCTION_N_THRESHOLD = 1e16


def _pick_qid(qids: List[int], candidates: List[int]) -> Optional[int]:
    for q in candidates:
        if q in qids:
            return q
    return None


def parse_str_mesh(path: Path) -> Dict:
    """解析 2D .str 网格与解变量。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    coord_by_node: Dict[int, Tuple[float, float]] = {}
    for ln in lines:
        t = ln.split()
        if t and t[0] == "c" and len(t) >= 4:
            coord_by_node[int(t[1])] = (float(t[2]), float(t[3]))

    s_line = next((ln for ln in lines if ln.startswith("s ")), None)
    if s_line is None:
        raise ValueError(f"未找到 s 行: {path}")
    qids = list(map(int, s_line.split()[2:]))
    if qids and qids[0] == 513:
        qids = qids[1:]

    phos_q = _pick_qid(qids, [3, 6, 50, 51])
    bor_q = _pick_qid(qids, [5])
    acc_q = _pick_qid(qids, [72, 71])
    vac_q = _pick_qid(qids, [0])
    inter_q = _pick_qid(qids, [1])
    trap_q = _pick_qid(qids, [12, 14])

    nodes: List[Dict] = []
    for ln in lines:
        t = ln.split()
        if not t or t[0] != "n" or len(t) < 5:
            continue
        try:
            node = int(t[1])
            region = int(t[2])
        except ValueError:
            continue
        if region != 3:
            continue
        coord = coord_by_node.get(node)
        if coord is None:
            continue
        vals = list(map(float, t[4:]))
        if len(vals) != len(qids):
            continue
        row = {qid: v for qid, v in zip(qids, vals)}
        x_node, y_node = coord
        nodes.append(
            {
                "x": x_node,
                "y": y_node,
                "phosphorus": row.get(phos_q, float("nan")) if phos_q else float("nan"),
                "boron": row.get(bor_q, float("nan")) if bor_q else float("nan"),
                "acceptor": row.get(acc_q, float("nan")) if acc_q else float("nan"),
                "vacancies": row.get(vac_q, float("nan")) if vac_q else float("nan"),
                "interstitials": row.get(inter_q, float("nan")) if inter_q else float("nan"),
                "traps": row.get(trap_q, float("nan")) if trap_q else float("nan"),
            }
        )

    return {"nodes": nodes, "qids": qids}


def _net_doping(node: Dict) -> float:
    phos = abs(float(node.get("phosphorus", 0.0) or 0.0))
    acc = abs(float(node.get("acceptor", 0.0) or 0.0))
    bor = abs(float(node.get("boron", 0.0) or 0.0))
    return max(phos, acc, bor)


def extract_cutline_at_x(
    mesh: Dict,
    x_target: float,
    y_min: float = DEFAULT_Y_MIN,
    y_max: float = DEFAULT_Y_MAX,
    x_band: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    在 x ≈ x_target 处做竖直 cutline：
    - 先取 x 落在带宽内的节点
    - 按 y 分组，每组取最接近 x_target 的点
    """
    nodes = mesh["nodes"]
    if not nodes:
        return {"y_um": [], "net_doping": [], "phosphorus": [], "vacancies": [], "interstitials": [], "traps": []}

    xs = np.asarray([n["x"] for n in nodes], dtype=float)
    if x_band is None:
        dx = np.diff(np.unique(np.round(xs, 6)))
        x_band = float(np.min(dx[dx > 0])) * 1.5 if dx.size else 0.5
        x_band = max(x_band, 0.2)

    band_nodes = [n for n in nodes if abs(n["x"] - x_target) <= x_band]
    if not band_nodes:
        # 兜底：取离 x_target 最近的一批节点
        band_nodes = sorted(nodes, key=lambda n: abs(n["x"] - x_target))[: max(50, len(nodes) // 20)]

    y_bins: Dict[float, Dict] = {}
    for n in band_nodes:
        y = float(n["y"])
        if not np.isfinite(y) or y < y_min or y > y_max:
            continue
        y_key = round(y, 8)
        if y_key not in y_bins or abs(n["x"] - x_target) < abs(y_bins[y_key]["x"] - x_target):
            y_bins[y_key] = n

    if not y_bins:
        return {"y_um": [], "net_doping": [], "phosphorus": [], "vacancies": [], "interstitials": [], "traps": []}

    ys = sorted(y_bins.keys())
    out = {
        "y_um": ys,
        "net_doping": [_net_doping(y_bins[y]) for y in ys],
        "phosphorus": [abs(float(y_bins[y].get("phosphorus", 0.0))) for y in ys],
        "vacancies": [abs(float(y_bins[y].get("vacancies", 0.0))) for y in ys],
        "interstitials": [abs(float(y_bins[y].get("interstitials", 0.0))) for y in ys],
        "traps": [abs(float(y_bins[y].get("traps", 0.0))) for y in ys],
        "x_um": [float(y_bins[y]["x"]) for y in ys],
    }
    return out


def normalize_depth_y(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    valid = y[np.isfinite(y)]
    if valid.size == 0:
        return y
    return y - float(np.min(valid))


def merge_light_heavy_profiles(
    light: Dict[str, List[float]],
    heavy: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """
    合并轻掺 / 重掺 cutline 为一条复合曲线：
    在每个深度 y 上取 net_doping 的最大值，使曲线同时体现两个区域的高掺特征。
    """
    keys = sorted(set(light.get("y_um", [])) | set(heavy.get("y_um", [])))
    if not keys:
        return {"y_um": [], "net_doping": [], "phosphorus": [], "vacancies": [], "interstitials": [], "traps": []}

    def _interp_dict(profile: Dict[str, List[float]], yq: float, field: str) -> float:
        ys = profile.get("y_um", [])
        vs = profile.get(field, [])
        if not ys:
            return 0.0
        arr_y = np.asarray(ys, dtype=float)
        arr_v = np.asarray(vs, dtype=float)
        if yq <= arr_y[0]:
            return float(arr_v[0])
        if yq >= arr_y[-1]:
            return float(arr_v[-1])
        return float(np.interp(yq, arr_y, arr_v))

    merged = {
        "y_um": keys,
        "net_doping": [],
        "phosphorus": [],
        "vacancies": [],
        "interstitials": [],
        "traps": [],
    }
    for yq in keys:
        n_l = _interp_dict(light, yq, "net_doping")
        n_h = _interp_dict(heavy, yq, "net_doping")
        merged["net_doping"].append(max(n_l, n_h))
        merged["phosphorus"].append(max(_interp_dict(light, yq, "phosphorus"), _interp_dict(heavy, yq, "phosphorus")))
        merged["vacancies"].append(max(_interp_dict(light, yq, "vacancies"), _interp_dict(heavy, yq, "vacancies")))
        merged["interstitials"].append(max(_interp_dict(light, yq, "interstitials"), _interp_dict(heavy, yq, "interstitials")))
        merged["traps"].append(max(_interp_dict(light, yq, "traps"), _interp_dict(heavy, yq, "traps")))
    return merged


def _safe_trapz(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def _surface_conc(y: np.ndarray, n: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    idx = int(np.argmin(y))
    return float(n[idx])


def _junction_depth(y: np.ndarray, n: np.ndarray, threshold: float = JUNCTION_N_THRESHOLD) -> float:
    mask = np.isfinite(y) & np.isfinite(n) & (n >= threshold)
    if not np.any(mask):
        return float("nan")
    return float(np.max(y[mask]))


def _estimate_sheet_resistance(y_um: np.ndarray, n_cm3: np.ndarray, xj_um: float) -> float:
    """由 cutline 浓度积分近似方阻（Ω/sq）。"""
    if not np.isfinite(xj_um) or xj_um <= 0 or y_um.size == 0:
        return float("nan")
    mask = (y_um <= xj_um) & np.isfinite(n_cm3) & (n_cm3 > 0)
    if not np.any(mask):
        return float("nan")
    y_cm = y_um[mask] * 1e-4
    integral = _safe_trapz(y_cm, n_cm3[mask])  # cm^-2
    if integral <= 0:
        return float("nan")
    q = 1.602176634e-19
    mu_eff = 200.0  # cm^2/V/s，高掺杂区有效迁移率近似
    return float(1.0 / (q * mu_eff * integral))


def compute_process_metrics(
    light: Dict[str, List[float]],
    heavy: Dict[str, List[float]],
) -> Dict[str, float]:
    """从轻/重掺 cutline 计算 O1~O6 工艺指标。"""
    y_l = normalize_depth_y(np.asarray(light.get("y_um", []), dtype=float))
    n_l = np.asarray(light.get("net_doping", []), dtype=float)
    y_h = normalize_depth_y(np.asarray(heavy.get("y_um", []), dtype=float))
    n_h = np.asarray(heavy.get("net_doping", []), dtype=float)

    xj_l = _junction_depth(y_l, n_l)
    xj_h = _junction_depth(y_h, n_h)
    cs_l = _surface_conc(y_l, n_l)
    cs_h = _surface_conc(y_h, n_h)

    return {
        "Rs_light": _estimate_sheet_resistance(y_l, n_l, xj_l),
        "Rs_heavy": _estimate_sheet_resistance(y_h, n_h, xj_h),
        "Xj_light": xj_l,
        "Xj_heavy": xj_h,
        "Cs_light": cs_l,
        "Cs_heavy": cs_h,
    }


def extract_composite_from_str(
    path: Path,
    x_light: float = DEFAULT_X_LIGHT,
    x_heavy: float = DEFAULT_X_HEAVY,
    y_min: float = DEFAULT_Y_MIN,
    y_max: float = DEFAULT_Y_MAX,
) -> Dict:
    """完整 cutline 流程：解析 .str → 轻/重 cutline → 复合曲线 + 工艺指标。"""
    mesh = parse_str_mesh(path)
    light = extract_cutline_at_x(mesh, x_light, y_min=y_min, y_max=y_max)
    heavy = extract_cutline_at_x(mesh, x_heavy, y_min=y_min, y_max=y_max)
    composite = merge_light_heavy_profiles(light, heavy)
    metrics = compute_process_metrics(light, heavy)
    return {
        "mesh": mesh,
        "light": light,
        "heavy": heavy,
        "composite": composite,
        "metrics": metrics,
    }
