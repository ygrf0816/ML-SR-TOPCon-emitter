# Provenance and transfer record

This repository consolidates code that originally lived in two separate locations.
This file records what came from where, so the consolidation is auditable.

## Sources

| Source | Origin | Contents taken |
|---|---|---|
| Workstation `10.10.20.87` (`zch@pcdlab-C621-WD12-IPMI`, Ubuntu 22.04) | `/home/zch/code/tcadproject/topcon_experiments/` | all analysis code (112 modules) |
| Local Windows PC | `E:\workspace\datagen\` (package `se-TOPCon`) | all Silvaco data-generation code (30 files) |

Transfer method: `tar czf - … | tar xzf -` over SSH with batch-mode key auth.
Excluded at the source: `outputs/`, `__pycache__/`, `*.pyc`,
`_backup_pre_std_fix_20260927_163801/`, `ds_api`, and the two
`*.bak_20260927_*` backups.

## Integrity

All 112 transferred Python modules were verified **byte-identical** to their
workstation originals by comparing the sorted set of per-file MD5 digests:

```
md5(concat(sorted(md5sum of each .py))) = c49ad432aca725dcb023cde11a6c47e0
```

computed independently on both machines and matching exactly.

## Modifications made during consolidation

These are the only edits applied on top of the transferred sources. Everything
else is verbatim.

| File | Change | Reason |
|---|---|---|
| `_patch2_numpy_safe.py` | `ROOT` → `Path(__file__).resolve().parents[0]` | was `_patch2_numpy_safe.py:16` hardcoded to `/home/zch/code/tcadproject/topcon_experiments` |
| `_patch_de_trace_std.py` | `ROOT` → `parents[0]`; docstring reference de-localised | same hardcoded path |
| `_run_exp2_finalize.py` | `ROOT` → `parents[1]` | was `/home/zch/code/tcadproject` |
| `_run_exp2_one_metric.py` | `ROOT` → `parents[1]` | was `/home/zch/code/tcadproject` |
| `_run_exp2_rerun.py` | `ROOT` → `parents[1]` | was `/home/zch/code/tcadproject` |
| `exp4_symbolic/export_sr_scatter_all.py` | `ROOT` → `parents[2]` | was `/home/zch/code/tcadproject` |
| `requirements.txt` | added `httpx>=0.18` | `exp4_symbolic/llm_analysis.py` and `exp5_correlation/ollama_embedding.py` import `httpx` (`BaseTransport` needs ≥0.18); it was undeclared |
| `launch_parallel.sh` | `cd` derived from `$0`; log dir → `logs/`; conda path via `CONDA_SH`/`CONDA_ENV` env vars | had two absolute `/home/zch` paths |
| `COMPUTATION_INDEX.md` | 2 absolute root paths → `<repo>` placeholders | documentation portability |
| `.gitignore` | reduced to a repo-root file covering both modules | the per-package file ignored `outputs/` only; also added data/secrets patterns |
| `datagen/` | flattened from `se-TOPCon/`; dropped `saomiao3/*.str`, `topcon_0.str`, `pipeline_*.log` | generated simulation artifacts (23 MB → 2 MB), not source |

`config.py` was **not** modified: its `parents[1]` already resolves to the repo
root in this layout, exactly as it did on the workstation.

## Credential note

The workstation file `topcon_experiments/ds_api` contained a **plaintext DeepSeek
API key**. It was deliberately excluded from the transfer and is covered by
`.gitignore`. It is not present anywhere in this repository. If that key was ever
committed to another repository or shared, rotate it.

## Not transferred

| Item | Size | Reason |
|---|---|---|
| `topcon_experiments/outputs/` | ~9.8 GB | regenerable; contains AutoGluon models, PySR checkpoints, figures |
| `new_data/` (dataset + curves) | ~490 MB | data, not code — see README §3 for the layout the code expects |
| `models/`, `models_pv/`, `boronscan2/` | ~7 GB | earlier-generation artefacts, not used by the manuscript |
| `docs/`, `readme.md`, `CLAUDE.md` at the `tcadproject` level | ~20 MB | describe the older GA/cINN line of work, not this paper; `CLAUDE.md` was `0600` and unreadable |

## Verification performed after consolidation

- 58 `Path(__file__).resolve().parents[N]` bootstraps checked — all resolve to the
  repo root or the package root (0 bad).
- 132 Python files parse cleanly (`ast.parse`).
- 488 `topcon_experiments.*` imports all resolve to real modules (0 unresolved).
- Every third-party import is declared in a `requirements.txt`.
- Empirical import test of all 112 package modules with heavy dependencies stubbed:
  **105 OK**. The 7 non-OK are all explainable and pre-existing, not caused by the
  move:
  - `_run_exp2_finalize.py`, `_run_exp2_one_metric.py`, `_run_exp2_rerun.py` —
    these are run-time driver scripts that execute at import (`sys.argv[1]`,
    loading `models_meta.json`); they require CLI arguments and trained models, so
    they cannot be imported as libraries. Expected.
  - `exp4_symbolic/llm_analysis.py`, `analyze_prior_formula_with_llm.py`,
    `analyze_theta_formulas_with_llm.py`, `run_doping_r_sheet_pipeline.py` — fail
    on the *local* machine only, because its `httpx` is 0.13.3 while these modules
    need ≥0.18 for `from httpx import BaseTransport`. Declared in
    `requirements.txt`.
