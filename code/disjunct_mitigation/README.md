# Disjunct-aware FP-SMOTE (small-disjunct mitigation)

A variant of the FP-SMOTE core that targets the **small-disjunct disadvantage** established by
the paired test (`code/disjuncts_textbook/paired_test.py`): small disjuncts are systematically
worse on **F1**, **|AOD|**, and **linkability**, but not on EOD/SPD/inference. This pipeline tries
to recover small-disjunct F1 and tighten AOD **without** regressing linkability.

It is a self-contained copy of the core trio (it does not touch `code/main`):

```
fair_priv_smote.py    path shim + smote_v3 forwarding the per-fold small_disjunct_mask
generate_samples.py   modified new_apply: focused allocation (1) + cross-group borrowing (2)
pipeline.py           driver: computes the per-fold mask, runs synthesis + metrics
```

## What it does (gated by `FPS_DISJUNCT_AWARE=1`)

1. **Focused allocation.** Within each minority (class×protected) subgroup's existing oversampling
   budget, a data-driven share is generated from **small-disjunct seeds**, sized by the within-pocket
   protected deficit (AOD-oriented, not a flat count boost):
   ```
   m_small[c]   = max over protected p of small-disjunct count in class c
   target[c,p]  = aug_disjunct * m_small[c]
   deficit[c,p] = max(target[c,p] - small_count[c,p], 0)   # capped by the subgroup budget
   ```
   The favored in-pocket side gets deficit≈0 (behaves like baseline); the starved side pulls focus.
2. **Cross-group borrowing.** Only when the in-group small-disjunct pocket has `< knn+1` rows, the
   focused rows are interpolated using neighbours drawn from the **whole same-class pool** (any
   protected value), with the subgroup identity stamped back. Avoids degenerate-blob synthesis in the
   tiniest pockets. (Piece (3), pocket-scaled DP noise, was intentionally **not** built.)

The remainder of each budget falls through to unchanged FP-SMOTE augmentation, so privacy behaviour
(single-out replacement, Laplace noise at `1/epsilon`) is preserved.

The small-disjunct flag is the same Weiss / fully-grown-tree definition as
`code/disjuncts_textbook/flag_folds.py` (bottom ~20% coverage), computed inline per fold so it is
aligned by position to the training rows actually synthesised.

## Run it (from repo root)

```bash
priv39/bin/python code/disjunct_mitigation/pipeline.py
```

Writes:
- synth files -> `datasets/outputs/outputs_4/disjunct_mitigation/<ds>/fold<N>/`
- metrics     -> `results_metrics/{fairness,linkability}_results/outputs_4/disjunct_mitigation/`

over all 13 datasets in `datasets/inputs/test`, grid `eps∈{0.1,0.5,1.0}, k∈{3,5}, knn=5, aug=0.4`.

## Knobs (env vars)

| var | default | meaning |
|-----|---------|---------|
| `FPS_DISJUNCT_AWARE` | `1` (set in `__main__`) | enable the mitigation; set `0` for a baseline in the same group |
| `FPS_DISJUNCT_AUG` | `augmentation_rate` (0.4) | in-pocket augmentation target (raise to focus harder) |
| `FPS_DISJUNCT_COVERAGE` | `0.2` | coverage fraction defining "small" disjuncts |

## A/B against baseline

For a clean comparison, run once as-is, then run a baseline into a different group:
```bash
FPS_DISJUNCT_AWARE=0 priv39/bin/python code/disjunct_mitigation/pipeline.py   # edit final_folder_name first
```
or simply compare against the existing `small_disjuncts` group (stock FP-SMOTE on the same 13 datasets).
