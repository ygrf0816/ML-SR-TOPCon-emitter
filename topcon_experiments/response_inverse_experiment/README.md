# `response_inverse_experiment/` — supplementary-sample chain and inverse-design re-runs

Everything in this directory belongs to the revision experiments that answer the
reviewers' questions on the 27 % recipe, the surrogate–TCAD offset and the absent
end-to-end baseline. The numbered sub-directories are the data-generation chain
that produced the 50 supplementary TCAD samples; the flat `*.py` files at this
level are the analysis and re-run drivers.

> **All paths inside `00_design/`, `01_tcad/` and `02_augment/` are placeholders.**
> Every module-level path variable keeps its name but holds the fixed string
> `"<PATH>"`. Set them to your own locations before running; nothing else in those
> files depends on the original directory layout.

---

## 1. The chain

```
00_design/make_candidates.py      dataset + de_best_iv_Eff.csv
        │                         -> candidates_50.csv, design_spec.json
        ▼
        [ Athena -> .str -> doping.txt -> Atlas -> I-V ]
        │      the batch driver is not distributed; its output is shipped
        │      as data/results_50.csv
        ▼
02_augment/make_augmented.py      .str + v1.csv + the generator's own descriptors
                                  -> topcon_dataset_v1_aug.csv  (50 x 28)
```

| Stage | Entry point | Produces | Backs |
|---|---|---|---|
| 00 design | `00_design/make_candidates.py` | `candidates_50.csv`, `design_spec.json` | Note S4 sampling recipe, Table S15 |
| — TCAD | not distributed; product shipped as `data/results_50.csv` | the 50 TCAD labels | Table S15, Fig. S8 |
| 01 helpers | `01_tcad/tcad_lib.py` (deck builders, `.str` reader, Atlas wrapper) | — | Note S2 Step 1/3 |
| 01 checks | `01_tcad/validate_{athena,end2end,thick_format}.py` | reproduction checks against archived samples | Note S2 Step 1/3 |
| 02 augment | `02_augment/make_augmented.py` | `topcon_dataset_v1_aug.csv` | Table S15, Fig. S8, Table S16 |

The 50-row table and the raw TCAD batch result are both shipped under `data/`, so
the downstream analysis runs without re-generating anything:
`data/topcon_dataset_v1_aug.csv` (50 x 28, the dataset schema) and
`data/results_50.csv` (the per-candidate Athena/Atlas metrics).

## 2. Sampling recipe (`00_design/make_candidates.py`)

Fixed seed `20261021`, box = min/max of each process parameter over the dataset,
perturbations applied in box-normalised coordinates `u = (x - box_lo)/(box_hi - box_lo)`:

| Group | n | Rule |
|---|---|---|
| A1 | 10 | Gaussian around the DE optimum, σ_u = 0.010, dECL additionally capped at 0.030 µm |
| A2 | 20 | same, σ_u = 0.020 |
| B | 20 | Latin hypercube inside the top-20-PCE envelope widened to 1.2× its own width, clipped back to the box |

Result: 50/50 inside the box, max |z| = 1.733, PCE 25.70–27.06 %, median 26.89 %,
44 of 50 above 26.5 %, against a database ceiling of 26.02 %.

## 3. Re-run drivers (flat files at this level)

| Script | Answers | Output |
|---|---|---|
| `run_r1_4_pipeline.py` | surrogate-vs-TCAD offset, retrain on train+50, before/after on the same test set | `out/surrogate_vs_tcad_50.csv`, `out/test_accuracy_before_after.csv`, `out/de_best_{before,after}.csv`, `out/de_trace_{before,after}.csv` |
| `run_multiseed.py` | ten independent seeds on the same chain | `out/multiseed/ms_recipes_*.csv`, `out/multiseed/ms_trace_*.csv` |
| `run_de_target.py` | per-target DE (Voc, Jsc, FF, PCE) on the retrained chain | `out/de_trace_after_<Tgt>.csv`, `out/de_population_after_<Tgt>.csv` |
| `dump_de_population.py` | final-generation population of the PCE run | rain-cloud data behind Fig. S3I |

`fwd.py` is a copy of the forward-chain pieces with two vectorised helpers
(`predict_curve_frame`, `predict_iv_frame`) needed by the batch scoring above; it
reads its settings from `config.py`, which re-exports `topcon_experiments/config.py`
so that no second copy of the hyper-parameters exists.

## 4. External modules that are **not** in this repository

`02_augment/make_augmented.py` calls the post-processing of the original dataset
generator instead of re-implementing it, so that the descriptors of the new
samples are produced by the same code path as the 10,159 archived ones. It needs
two modules that are not distributed here:

| Module | Provides | Configured by |
|---|---|---|
| `topcon_datagen_main.py` | `_parse_str_1d`, `_build_doping_curve`, `_build_defect_curves`, `_doping_descriptors`, `_curve_descriptors`, `_replace_one_with_nearby_max` | `GEN` |
| `analyze_scans.py` | `compute_metrics`, `read_iv_from_csv` | `GEN`, and `DATAGEN` in `01_tcad/tcad_lib.py` |

They are available from the corresponding author on reasonable request.

## 5. Run order

```bash
# 00 — design the batch (needs the dataset and one DE optimum)
python 00_design/make_candidates.py

# 02 — assemble the augmentation table in the dataset schema
#      (needs the Athena/Atlas outputs of the 50 candidates; the shipped
#       data/results_50.csv is that product)
python 02_augment/make_augmented.py

# re-run drivers, in this order
python run_r1_4_pipeline.py --aug data/topcon_dataset_v1_aug.csv --out out
python run_multiseed.py --case before --seeds 0-9
python run_multiseed.py --case after  --seeds 0-9
python run_de_target.py --target iv_Voc --out out
python dump_de_population.py
```

## 6. Two conventions that must not be changed silently

1. **`thick` is written with two decimals** (`.2f`) in the Athena deck. The
   generator quantised the BSG thickness to 0.01 µm; `.4f` breaks reproduction
   (median curve deviation 0.47 %, max 5209 %, against 0.000 % with `.2f`).
   See `01_tcad/validate_thick_format.py`.
2. **`compute_metrics(..., scale_i=1.16)`** is the label convention of the
   deposited dataset. The 50 new rows must carry the same factor, otherwise they
   are not comparable with the 10,159 training labels. `run_atlas` also records the
   unscaled value as `Eff_raw`, so both numbers are available.
