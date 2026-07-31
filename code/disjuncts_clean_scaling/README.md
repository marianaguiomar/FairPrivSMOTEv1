# #2 (ENN-clean) scaling experiment

Tests whether ENN-cleaning the focused synthetic mass (**#2** = `FPS_DISJUNCT_CLEAN`) reduces
small-disjunct error **in proportion to a dataset's absolute pocket error (`err_small`)**, and
**not** its error concentration (EC).

## Background

Earlier runs established (seed-gated on dataset 33): #2 lifts minority recall ~+0.027, F1 ~+0.024,
err ~−0.016 on a high-error pocket, robust across 10 RF seeds — while doing nothing on low-error /
ceiling datasets (e.g. compas: top EC 0.96 but `err_small` only 0.059). This experiment scales that
to the full dataset spread to see if the relationship `Δrecall ∝ err_small` (not `∝ EC`) holds.

## What it does

1. **generate_clean.py** — produces the `disjunct_addon_clean` synthetic outputs (#2) for every
   dataset in `config.DATASETS` that doesn't already have them. Stock (`small_disjuncts`) synth is
   assumed already present.
2. **eval_impact.py** — reuses `code/disjuncts_textbook/impact.py` to compute the pooled small-stratum
   error metrics (err, minority recall, AUPRC, Brier, F1) for stock and clean, upserting per dataset.
3. **analyze.py** — joins the two, computes clean−stock deltas, and reports
   `corr(err_small, Δrecall)` vs `corr(EC, Δrecall)` (+ Δerr, ΔF1), plus a scatter.

## Run it (from repo root, full env)

```bash
priv39/bin/python code/disjuncts_clean_scaling/run_all.py
# or force-regenerate all synth:
priv39/bin/python code/disjuncts_clean_scaling/run_all.py --force
```

Or step by step:

```bash
priv39/bin/python code/disjuncts_clean_scaling/generate_clean.py
priv39/bin/python code/disjuncts_clean_scaling/eval_impact.py
priv39/bin/python code/disjuncts_clean_scaling/analyze.py
```

## Outputs

- `exploratory_metadata/disjuncts_impact_small_disjuncts.csv` (stock, upserted)
- `exploratory_metadata/disjuncts_impact_disjunct_addon_clean.csv` (clean, upserted)
- `exploratory_metadata/clean_scaling_summary.csv` (per-dataset deltas)
- `exploratory_metadata/clean_scaling_recall_vs_err.png` (the key scatter)

## Reading the result

The hypothesis is confirmed if **`|corr(err_small, Δrecall)|` is large and positive while
`|corr(EC, Δrecall)|` is near zero** — i.e. the recall gain tracks how much error the pocket
actually has, not how concentrated it is. If both correlations are flat, the 33 win doesn't
generalize and the honest write-up is the negative ("error immovable") backed by the threshold-free
metrics.

## Notes / knobs

- **Datasets**: edit `config.DATASETS`. Adult is intentionally excluded (large / dropped).
- **Classifier seed**: `FPS_RF_SEED` env var (default 57) — `impact.py` honours it. To rule out RF
  variance on a candidate, run `eval_impact.py` under a few seeds and compare.
- **Synthesis variance** (Laplace/SMOTE) is *not* averaged here — the synth files are fixed. To test
  that dimension, `--force` regenerate under different synthesis conditions.
- **Privacy** (linkability/inference) is intentionally out of scope here (slow, orthogonal to the
  error hypothesis). Run per dataset if wanted:
  `priv39/bin/python code/disjuncts_textbook/privacy_impact.py disjunct_addon_clean <dataset>`
- Fixed grid throughout: `eps=1.0, k=3, knn=5, aug=0.4`, Weiss coverage 0.2, threshold 0.5.
- Disparate Impact is always reported **raw** — never |DI−1|.
