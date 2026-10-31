"""COPY of `topcon_experiments/config.py`, patched for the R1-4 rerun.

One edit, deliberate.  Everything else is byte-identical to the original, so
numbers stay comparable with the published results.

--- Edit: `doping_R_sheet` removed from DOPING_DESCRIPTORS --------------------

The deployed Model 2 was trained on **17** inputs
(`models_meta.json::feature_columns_model2`):
    8 process + 6 doping (no R_sheet) + 3 defect.
The dataset used for this study carries 28 columns and has **no** sheet-resistance
column; `doping_R_sheet` was added later in a side analysis.  With R_sheet kept in
DOPING_DESCRIPTORS, `MODEL2_FEATURES` resolved to 18 and the predictor was handed a
column it had never seen.  Dropping it makes CURVE_DESCRIPTORS = 9 and
MODEL2_FEATURES = 17, which matches both the weights and the dataset.

Paths are re-anchored because this copy sits one directory deeper than the
original (`topcon_experiments/response_inverse_experiment/`).
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../topcon_experiments/response_inverse_experiment
_EXP = _HERE.parent                              # .../topcon_experiments
PROJECT_ROOT = _EXP.parent                       # .../tcadproject
EXP_ROOT = _EXP
OUTPUT_ROOT = _EXP / "outputs"

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

# EDIT 1: "doping_R_sheet" removed - see module docstring.
DOPING_DESCRIPTORS = [
    "doping_N_peak",
    "doping_x_peak",
    "doping_junction_depth",
    "doping_FWHM",
    "doping_gradient_max",
    "doping_dose",
]

DEFECT_DESCRIPTORS = [
    "defect_vac_N_peak",
    "defect_vac_gradient_max",
    "defect_vac_dose",
]

CURVE_DESCRIPTORS = DOPING_DESCRIPTORS + DEFECT_DESCRIPTORS       # 9

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

MODEL2_FEATURES = ATHENA_FEATURES + CURVE_DESCRIPTORS              # 17

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

# Inverse design (differential evolution) - raw surrogate output only
DE_LOCAL_REFINE = True
DE_MAXITER = 30
DE_POPSIZE = 15
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_SEED = 42
DE_TOL = 0.0
DE_ATOL = 0.0
DE_LOG_EVERY_EVALS = 10
# EDIT 2: DE_IV_SCALE_ENABLED / DE_IV_CALIB_REFERENCE / DE_IV_CALIB_TARGET deleted.

BOUNDS_QUANTILE_LOW = 0.05
BOUNDS_QUANTILE_HIGH = 0.95

# NSGA-II
NSGA_POP_SIZE = 40
NSGA_N_GEN = 20

# LLM (kept for completeness; unused by the R1-4 pipeline)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DS_API_FILE = EXP_ROOT / "ds_api"
LLM_REQUEST_TIMEOUT = 180
LLM_MAX_RETRIES = 5
LLM_RETRY_BASE_DELAY = 5.0
LLM_INTER_CALL_DELAY = 0.3


def _load_deepseek_api_key() -> str | None:
    env_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    if DS_API_FILE.exists():
        return DS_API_FILE.read_text(encoding="utf-8").strip()
    return None


DEEPSEEK_API_KEY = _load_deepseek_api_key()
