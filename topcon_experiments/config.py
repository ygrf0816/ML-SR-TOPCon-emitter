"""Central configuration for TOPCon experiments."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXP_ROOT / "outputs"

DATA_CSV = PROJECT_ROOT / "new_data" / "topcon_dataset_v1.csv"
DOPING_CURVE_DIR = PROJECT_ROOT / "new_data" / "saomiao3csv"
DEFECT_CURVE_DIR = PROJECT_ROOT / "new_data" / "trap_curves"

ATHENA_FEATURES = [
    "athena_thick",
    "athena_c_boron",
    "athena_temp1",
    "athena_time1",
    "athena_temp2",
    "athena_time2",
    "athena_F_N2",
    "athena_F_O2",
]

DOPING_DESCRIPTORS = [
    "doping_N_peak",
    "doping_x_peak",
    "doping_junction_depth",
    "doping_FWHM",
    "doping_gradient_max",
    "doping_dose",
    "doping_R_sheet",
]

DEFECT_DESCRIPTORS = [
    "defect_vac_N_peak",
    "defect_vac_gradient_max",
    "defect_vac_dose",
]

CURVE_DESCRIPTORS = DOPING_DESCRIPTORS + DEFECT_DESCRIPTORS

IV_TARGETS = ["iv_Voc", "iv_Jsc", "iv_FF", "iv_Eff"]
IV_TARGET_LABELS = {"iv_Voc": "Voc", "iv_Jsc": "Jsc", "iv_FF": "FF", "iv_Eff": "PCE"}

LOG_FEATURES = [
    "athena_c_boron",
    "doping_N_peak",
    "doping_dose",
    "defect_vac_N_peak",
    "defect_vac_dose",
]

LOG_TARGETS_MODEL1 = [
    "doping_N_peak",
    "doping_dose",
    "defect_vac_N_peak",
    "defect_vac_dose",
]

MODEL2_FEATURES = ATHENA_FEATURES + CURVE_DESCRIPTORS

RANDOM_STATE = 42
TEST_SIZE = 0.2
AUTOGLUON_PRESET = "medium_quality_faster_train"

MODEL1_DIR = OUTPUT_ROOT / "exp1_forward" / "models" / "model1"
MODEL2_DIR = OUTPUT_ROOT / "exp1_forward" / "models" / "model2"
META_JSON = OUTPUT_ROOT / "exp1_forward" / "models_meta.json"
CLASSIFIER_DIR = OUTPUT_ROOT / "exp3_classifier" / "models"

# Efficiency tier SHAP: 4 classes, 1% width; anchor shifted down from floor(max)
EFF_TIER_STEP = 1.0
EFF_TIER_COUNT = 4
EFF_TIER_ANCHOR_SHIFT = 1  # tiers: >=25%, 24~25%, 23~24%, 22~23% when max~26%
SHAP_TOP_FEATURES = 5
CLASSIFIER_LABEL = "high_efficiency"

# Curve preprocessing
CURVE_NUM_POINTS = 256
DEPTH_MAX = 2.0
CURVE_SMOOTH_METHOD = "spline"

# Inverse design (differential evolution)
DE_LOCAL_REFINE = True
DE_MAXITER = 30
DE_POPSIZE = 15
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_SEED = 42
DE_TOL = 0.0 #控制早停，设置为0不启用
DE_ATOL = 0.0
DE_LOG_EVERY_EVALS = 10
# Surrogate IV calibration for inverse design (multiplicative: scaled = raw * scale).
# Set reference = old surrogate at a fixed design point; target = updated simulation at same point.
DE_IV_SCALE_ENABLED = True
DE_IV_CALIB_REFERENCE = {
    "iv_Voc": 0.716534,
    "iv_Jsc": 42.4565,
    "iv_Eff": 26.2222,
    "iv_FF": 86.0884,
}
DE_IV_CALIB_TARGET = {
    "iv_Voc": 0.78,
    "iv_Jsc": 41.79,
    "iv_Eff": 27.02,
    "iv_FF": 86.0884,
}
# Legacy: tight bounds from top-efficiency samples (no longer used for iv_Eff seeding)
DE_TOP_FRAC = 0.05
BOUNDS_QUANTILE_LOW = 0.05
BOUNDS_QUANTILE_HIGH = 0.95

# NSGA-II
NSGA_POP_SIZE = 40
NSGA_N_GEN = 20

# Symbolic regression — fast validation defaults (increase for production runs)
SR_CURVE_UNIFORM_FRAC = 0.15
SR_CURVE_UNIFORM_FRAC_DOPING = 0.05
SR_CURVE_GRAD_POWER_DOPING = 1.5
CURVE_TAIL_DEPTH_MIN = 0.25
CURVE_TRUNCATION_DEPTH_MARGIN = 0.05
CURVE_TRUNCATION_DROP_RATIO = 0.5
SR_MAX_FILES = 80
SR_MAX_POINTS_PER_FILE = 200
SR_MAX_TOTAL_POINTS = 16000
# Tail-window curve SR: use every depth point (no adaptive subsampling per sample)
SR_TAIL_USE_ALL_POINTS = True
SR_CURVE_MAX_FIT_ROWS = 0  # 0 = use all train rows for curve PySR fit
SR_TABULAR_MAX_ROWS = 10000
SR_NITERATIONS = 500
SR_POPULATIONS = 40
SR_POPULATION_SIZE = 300
SR_MAXSIZE = 20
SR_BATCHING = True
# Accuracy-first: LLM simplifies formulas later; avoid parsimony during PySR search.
SR_MODEL_SELECTION = "accuracy"
SR_PARSIMONY = 0.0
SR_TOP_FORMULAS_FOR_LLM = 10
SR_TOP_FORMULAS_IN_REPORT = 20
# LLM / report: doping curve uses tail-window full-feature SR (not legacy full-curve athena-only)
SR_DOPING_CURVE_FORMULA_PATH = (
    OUTPUT_ROOT / "exp4_symbolic" / "curve_tail_experiment" / "sr_formulas_doping_tail_full.csv"
)
SR_DEFECT_CURVE_FORMULA_PATH = OUTPUT_ROOT / "exp4_symbolic" / "sr_formulas_defect.csv"
# Quick validation: only 2 tabular targets; set False for full tabular SR
SR_QUICK_VALIDATE = False
SR_QUICK_TABULAR_TARGETS = ["doping_N_peak", "iv_Eff"]

# LLM
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DS_API_FILE = EXP_ROOT / "ds_api"
LLM_REQUEST_TIMEOUT = 180  # seconds per API call (connect + read)
LLM_MAX_RETRIES = 5
LLM_RETRY_BASE_DELAY = 5.0  # exponential backoff base (seconds)
LLM_INTER_CALL_DELAY = 0.3  # pause between calls to ease rate limits


def _load_deepseek_api_key() -> str | None:
    env_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    if DS_API_FILE.exists():
        return DS_API_FILE.read_text(encoding="utf-8").strip()
    return None


DEEPSEEK_API_KEY = _load_deepseek_api_key()

# Local Ollama embeddings (exp5 correlation analysis)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")
OLLAMA_EMBED_TIMEOUT = 300  # seconds; first load can be slow
EXP5_OUT = OUTPUT_ROOT / "exp5_correlation"
