# TOPCon boron-emitter process–profile–performance framework

Code accompanying the manuscript:

> **Machine Learning and Symbolic Regression Enabled Process–Performance Prediction and Inverse Design for TOPCon Solar Cells**

This repository contains the full computational chain of the paper: from Silvaco TCAD
generation of the process–profile–performance dataset, through two-stage forward
prediction and differential-evolution inverse design, to SHAP interpretation and
symbolic regression of transferable formula templates.

The repository is **code-only**. The dataset and the simulation curves are not
version-controlled here — see [§3 Data setup](#3-data-setup) for the exact files the
code expects and where to put them.

---

## 1. Repository layout

```
.
├── datagen/                  # Silvaco Athena/Atlas data generation (Windows, local workstation)
│   ├── se.in                 #   Athena process deck (SE / boron diffusion)
│   ├── se_athena_scan.py     #   parameterised Athena sweep driver
│   ├── topcon_datagen_main.py#   end-to-end main entry: .str -> curves -> IV -> dataset
│   ├── cd_txt_scan.py        #   Atlas device run from an external doping curve
│   ├── analyze_scans.py      #   IV CSV -> Voc / Jsc / FF / Eff
│   ├── extract_athena_defect_features.py
│   ├── tcadmodel.py          #   TOPCon device model (topcon_n, topcon_n_cd)
│   ├── tcaddata.py           #   Silvaco command objects (auto-generated, large)
│   ├── tcadutils.py
│   ├── python_tcad.py        #   subprocess wrapper around athena.exe / atlas.exe
│   ├── template.lib          #   Atlas user-function template library
│   └── *_nk.nk               #   optical constants for the device stack
│
├── topcon_experiments/       # the paper's analysis pipeline (Linux, GPU/CPU workstation)
│   ├── config.py             #   single source of truth: paths, feature lists, hyper-parameters
│   ├── common/               #   shared layer (data, forward chain, metrics, plotting)
│   ├── exp1_forward/         #   Sec. 3.2  two-stage surrogate
│   ├── exp2_inverse/         #   Sec. 3.2  DE + L-BFGS-B inverse design
│   ├── exp3_classifier/      #   Sec. 3.3  efficiency-tier classifier + SHAP
│   ├── exp4_symbolic/        #   Sec. 3.3  symbolic regression + grey-box curve model
│   ├── exp5_correlation/     #   Sec. 3.1  physical + semantic correlation analysis
│   ├── exp6_experimental/    #   literature benchmark (D1–D10 profiles)
│   ├── tmp_fig5/             #   Fig. 5 panel assembly
│   ├── COMPUTATION_INDEX.md  #   detailed strategy / script / accuracy-report index (Chinese)
│   ├── run_all.py            #   sequential entry for exp1–exp5
│   ├── run_sr_full.py        #   full PySR pipeline incl. LLM simplification
│   └── requirements.txt
│
└── docs/
    ├── DEPENDENCY_MAP.md     # auto-generated import graph + portability audit
    └── WORKSTATION_REPORT.md # where each file lives on the original workstation
```

### Why two top-level modules

The two halves ran on different machines and different operating systems and are kept
separate on purpose:

| | `datagen/` | `topcon_experiments/` |
|---|---|---|
| Runs on | Windows PC with Silvaco installed | Linux workstation, conda env `autogluon` |
| Requires | Silvaco Athena + Atlas licence | AutoGluon, PySR, SHAP, pymoo |
| Produces | the dataset, doping/defect curves | models, figures, formulas, metrics |
| Language | mostly Chinese docstrings | English |

The dependency arrow points one way only: `datagen` → `topcon_experiments`.
`topcon_experiments` never calls Silvaco; it consumes files.

---

## 2. Pipeline overview

```
                     datagen/  (Silvaco TCAD)
  Athena: 8 process parameters ──> Na(x), Nt(x) profiles
     │                                   │
     │                                   ├──> saomiao3csv/<base>.txt          (doping curves)
     │                                   └──> trap_curves/<base>_trap_vacancies.txt (defect curves)
     └──> Atlas: IV sweep ──> Voc, Jsc, FF, PCE
                                     │
                                     ▼
                        new_data/topcon_dataset_v1.csv   (10,159 x 50)
                                     │
                       ┌─────────────┴──────────────┐
                       ▼                            ▼
        topcon_experiments/exp1_forward    topcon_experiments/exp2_inverse
        Model1: process -> 9 descriptors   DE + L-BFGS-B
        Model2: process+desc -> IV         -> optimal recipe (PCE 27.0%)
                       │
                       ├──> exp3_classifier  : 4-tier PCE classifier + SHAP
                       ├──> exp4_symbolic    : PySR formulas + LLM simplification
                       │                       + grey-box curve model (double-Gaussian / offset prior)
                       ├──> exp5_correlation : physical correlation + LLM-semantic panels
                       └──> exp6_experimental: literature D1–D10 benchmark
```

The two-stage surrogate follows the device-physics causal chain rather than mapping
process parameters to IV directly, which is what keeps the intermediate descriptors
(`Na,peak`, `djunc`, `Na,eff`, …) physically interpretable and comparable with
ECV/SIMS and four-point-probe measurements.

---

## 3. Data setup

`topcon_experiments/config.py` fixes the data locations relative to the repository
root. To run the analysis you must place the following files so that the tree looks
like this:

```
<repo root>/
├── new_data/
│   ├── topcon_dataset_v1.csv          # main dataset, 10,159 rows x 50 cols
│   ├── saomiao3csv/                   # doping curves: <file_base>.txt
│   └── trap_curves/                   # defect curves: <file_base>_trap_vacancies.txt
└── topcon_experiments/                # this package
```

This mirrors the original workstation layout, where `new_data/` sat next to
`topcon_experiments/` under the project root. If you would rather keep the data
elsewhere, edit `DATA_CSV`, `DOPING_CURVE_DIR` and `DEFECT_CURVE_DIR` at the top of
`topcon_experiments/config.py` — they are plain `pathlib.Path` objects and nothing
else hardcodes the location.

**Column contract.** The dataset must retain the `file_base` column: it is the join
key that links every row to its curve files in `saomiao3csv/` and `trap_curves/`.
Process features are prefixed `athena_`, doping descriptors `doping_`, defect
descriptors `defect_vac_`, and the four targets are `iv_Voc`, `iv_Jsc`, `iv_FF`,
`iv_Eff`.

**Not included**, and why:

| Excluded | Size | Reason |
|---|---|---|
| `saomiao3csv/` | ~239 MB | 10,159 raw Athena curve text files |
| `trap_curves/` | ~240 MB | 10,159 defect curve text files |
| `saomiao3jv/` | ~161 MB | raw Atlas IV outputs |
| `outputs/` | ~9.8 GB | trained AutoGluon models, checkpoints, figures — all regenerable |
| `models/`, `models_pv/`, `boronscan2/` | ~7 GB | earlier-generation models, not used by the manuscript |

The dataset is available from the corresponding author on reasonable request; see the
manuscript's Data Availability Statement.

---

## 4. Environment

### `topcon_experiments` — Linux, conda

The original runs used a conda environment named `autogluon`.

```bash
conda create -n autogluon python=3.10
conda activate autogluon
pip install -r topcon_experiments/requirements.txt
```

PySR additionally needs a Julia backend; it installs one automatically on first run
(`juliaup` or the bundled `juliacall`), which requires network access the first time.

`exp5_correlation` is the only module that needs a local embedding server:

```bash
ollama pull qwen3-embedding:8b     # OLLAMA_BASE_URL defaults to http://localhost:11434
```

`exp4_symbolic` LLM simplification calls the DeepSeek API. The key is read from, in
order: `$DEEPSEEK_API_KEY`, `$OPENAI_API_KEY`, then a file `topcon_experiments/ds_api`.
**`ds_api` is git-ignored — never commit it.** If the key is absent the LLM steps skip
cleanly instead of failing.

### `datagen` — Windows, Silvaco

```bash
pip install -r datagen/requirements.txt
set TCAD_EXE_DIR=D:\silvaco\exe      # directory containing athena.exe / atlas.exe
```

`TCAD_EXE_DIR` is the only environment-specific setting required. A number of
the older helper scripts (`summarize_boron_scan.py`, `random_param_scan.py`,
`analyze_scans.py`) still carry the author's absolute Windows paths
(`D:\code\silvaco\...`, `C:\sedatools\exe\...`) as defaults; these are historical
and are used only by those optional analysis helpers, not by
`topcon_datagen_main.py`, which resolves everything from its own location. They are
inventoried in `docs/DEPENDENCY_MAP.md` §4. Fix them locally before using those
particular scripts.

---

## 5. Reproducing the paper

Run everything from the **repository root** — the package is imported as
`topcon_experiments.*`, so the root must be on `sys.path`. (Individual scripts also
insert the root themselves, so running them directly works too.)

```bash
# ---------- A. dataset generation (Windows + Silvaco) ----------
cd datagen
python topcon_datagen_main.py                 # DATAGEN_MODE=full_pipeline
# writes new_data/topcon_dataset_v1.csv, saomiao3csv/, trap_curves/

# ---------- B. forward prediction (Sec. 3.2) ----------
python -m topcon_experiments.exp1_forward.train
python -m topcon_experiments.exp1_forward.evaluate
python -m topcon_experiments.exp1_forward.plot

# ---------- C. inverse design (Sec. 3.2) ----------
python -m topcon_experiments.exp2_inverse.de_single      # DE + L-BFGS-B per target
python -m topcon_experiments.exp2_inverse.oracle_report  # feasibility + envelope
python -m topcon_experiments.exp2_inverse.pareto_multi   # NSGA-II Pareto front

# ---------- D. interpretation (Sec. 3.3) ----------
python -m topcon_experiments.exp3_classifier.train_tiers
python -m topcon_experiments.exp3_classifier.evaluate_tiers
python -m topcon_experiments.exp3_classifier.shap_analysis

# ---------- E. symbolic regression + grey-box curve model (Sec. 3.3) ----------
python -m topcon_experiments.exp4_symbolic.preprocess_curves
python -m topcon_experiments.exp4_symbolic.run_process_to_dg_chain
python -m topcon_experiments.exp4_symbolic.run_process_to_theta_sr
python -m topcon_experiments.exp4_symbolic.run_theta_autogluon
python -m topcon_experiments.exp4_symbolic.eval_symbolic_chain_on_sim
python -m topcon_experiments.exp4_symbolic.run_process_to_defect_theta
python -m topcon_experiments.exp4_symbolic.package_six_variant_results

# ---------- F. correlation + literature benchmark ----------
python run_exp5_correlation.py
python -m topcon_experiments.exp6_experimental.run_literature_benchmark
```

All results land under `topcon_experiments/outputs/`, which is git-ignored. For the
accuracy-report index that ties each script to its output CSV, see
`topcon_experiments/COMPUTATION_INDEX.md`.

---

## 6. Code ↔ manuscript map

| Manuscript | Code |
|---|---|
| Sec. 2.1 Dataset generation | `datagen/` |
| Sec. 2.2 Modeling and optimisation | `topcon_experiments/exp1_forward/`, `exp2_inverse/` |
| Sec. 2.3 Physical interpretation | `exp3_classifier/`, `exp4_symbolic/` |
| Sec. 3.1 Distribution & correlation | `exp2_inverse/plot_fig3_extension.py`, `exp5_correlation/` |
| Sec. 3.2 Forward prediction & inverse design | `exp1_forward/`, `exp2_inverse/` |
| Sec. 3.3 SHAP & symbolic regression | `exp3_classifier/shap_analysis.py`, `exp4_symbolic/` |
| Fig. 5 panels | `tmp_fig5/` |
| Equation S1–S2 (double-Gaussian, offset prior) | `exp4_symbolic/defect_shape_prior.py`, `exp6_experimental/literature_benchmark/fit_double_gaussian_to_sim.py` |

Note that the manuscript is the authoritative description of the method; where a
script name and the paper's wording differ, the paper's terminology is the one to
cite.

---

## 7. Notes and known rough edges

- **`ds_api` must never be committed.** It holds a plaintext API key. It was
  excluded from this repository and is git-ignored; if the key was ever shared,
  rotate it.
- The repository is a *consolidation* of code that lived in several directories;
  some scripts are exploratory and were left in place rather than deleted, so that
  the numbers reported in the paper remain traceable. Files prefixed `_patch` and
  `_run_exp2_`, and the `tmp_fig5/` directory, are internal utilities and figure
  assembly rather than part of the method.
- `datagen/` was developed on Windows and is not POSIX-clean (it shells out to
  `athena.exe` / `atlas.exe`).
- Three `_run_exp2_*.py` drivers execute at import time and take `sys.argv`, so
  they are meant to be run as scripts, not imported.
- The absolute workstation paths that the original sources carried in their
  `sys.path` bootstrapping and in `launch_parallel.sh` have been replaced with
  locations derived from `__file__`. `launch_parallel.sh` additionally reads
  `CONDA_SH` / `CONDA_ENV` if your conda lives outside `~/anaconda3`. The full
  list of consolidations edits is in `docs/WORKSTATION_REPORT.md`.
- `docs/` documents the provenance and the dependency graph. For the
  script-to-results index, see `topcon_experiments/COMPUTATION_INDEX.md`.

---

## 8. Licence and citation

Released under the Apache License 2.0; see [LICENSE](LICENSE).

If you use this code, please cite the manuscript above. The associated dataset
is archived separately on Zenodo; see the manuscript's data availability
statement for the DOI.
