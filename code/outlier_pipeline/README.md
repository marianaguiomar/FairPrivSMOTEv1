# code/outlier_pipeline/

Modified copies of the three core FairPrivSMOTE files, with **outlier-aware
de-isolation** wired into the synthesis. They target the rows that actually drive
QI-based linkability — those that are **both** a k-anonymity single-out **and** a
geometric outlier (`Distance` or `Density == "Outlier"` in the `TypologyDetector`
output) — and de-isolate them **without spending any oversampling budget** and
**without changing any subgroup counts** (so fairness balance is untouched).

| file | mirrors | role |
|------|---------|------|
| `pipeline.py`         | `code/main/pipeline.py`         | builds the per-fold outlier mask, passes it into synthesis |
| `fair_priv_smote.py`  | `code/main/fair_priv_smote.py`  | threads `outlier_mask` through `smote_v3` |
| `generate_samples.py` | `code/main/generate_samples.py` | the actual handling inside `new_apply` |

Every change is bracketed with `# === OUTLIER CHANGE === ... # === END OUTLIER CHANGE ===`
so you can diff against `code/main/` to see exactly what differs.

## Running

These run **in place** (a path shim at the top of each file puts this folder first
on `sys.path`, then `code/main/` and `code/` for the unchanged dependencies). From
the **repo root**, with the full env:

```bash
priv39/bin/python code/outlier_pipeline/pipeline.py
```

Configure the run by editing the globals at the bottom of `pipeline.py` exactly as
in the original (`input_folder_name`, `final_folder_name`, `binning`,
`removal_strategy`, …). Unlike the original, the `smote_v3(...)` call in the fold
loop is **already un-commented** here so the method actually synthesises.

To instead deploy as a drop-in replacement, back up and copy each file over its
`code/main/` twin — the imports are otherwise identical.

## What changed, step by step (`generate_samples.py::new_apply`)

1. **New param `outlier_mask`** — a per-row boolean array (Distance∨Density Outlier),
   aligned by position to the fold. `pipeline.py` builds it from the same
   `typology_metadata` it already computes per fold (no disk round-trip).
2. **Step 1** — attach `dataset['is_outlier']`. The de-isolation target is the
   **intersection** `single_out & is_outlier`.
3. **Step 5 (majority) & Step 6a (minority)** — replacement covers **all** single-outs
   (same coverage as stock — privacy-safe). It is a 1:1 replacement, so **count-neutral**:
   zero budget cost, subgroup sizes unchanged (`df_rest + replaced == df_subset`). The
   only behavioural change vs stock is robustness: when a cell has too few single-outs to
   build the KNN graph, originals are kept and the fold is **not** aborted.
   **Important:** do NOT narrow replacement to the `outlier ∩ single-out` intersection —
   on high-single-out datasets (3/13, ~100% single-outs) that leaves raw single-out
   originals in the release and linkability rises sharply (validated empirically;
   risk jumped from ~0.05 to ~0.80). The outlier flag is used only as the Step 6b add-on.
4. **Step 6b (augmentation)** — the **fixed** `samples_to_increase` budget is *seeded*
   from the `outlier ∩ single-out` rows (via the `highest_risk` selector inside
   `newPrivateSMOTE.over_sampling`), so synthetic decoys densify/dilute the most
   linkable rows. Total generated rows is unchanged → **budget unchanged**. Falls back
   to all single-outs when the intersection is empty in a cell. This is the one place the
   outlier signal actually steers synthesis.
5. **Step 7.5 / Step 8** — `is_outlier` is excluded from the Tomek-link feature matrix
   and dropped from the final output (like `single_out`/`synthetic`).

## Why it respects the fairness cap

| operation | rows added | budget cost |
|---|---|---|
| Step 5 majority replace   | 0 (1:1 swap)                       | none |
| Step 6a minority replace  | 0 (`df_rest + replaced == df_subset`) | none |
| Step 6b augmentation      | exactly `samples_to_increase`      | unchanged |

Validated on datasets 3 & 13: per-fold output row counts match the stock pipeline
exactly, confirming the oversampling budget and subgroup balance are preserved.

Final per-subgroup counts are **identical** to the stock pipeline
(`len(df_subset) + samples_to_increase`), so DI/SPD/AOD targets do not move. The
de-isolation comes entirely from (i) *free* count-neutral replacement of the
`outlier ∩ single-out` rows and (ii) *redirecting* the existing augmentation budget
toward them — never from extra rows.

## Design notes / knobs

- **Why replacement stays on ALL single-outs.** It is tempting to narrow replacement to
  the `outlier ∩ single-out` intersection (the rows most linkable in theory). Empirically
  this backfires on high-single-out datasets: leaving the non-outlier single-outs as raw
  originals in the release spikes linkability (3/13 went from ~0.05 to ~0.80). So full
  single-out replacement is the privacy-safe baseline, and the outlier signal is applied
  only as the Step 6b augmentation focus. **To experiment** with intersection-only
  replacement, append `& (df_subset['is_outlier'] == 1)` to `sub_target` in Step 6a (and
  the majority analogue in Step 5) — useful only on datasets with *moderate* single-out
  prevalence.
- **Where the outlier lever actually bites.** On saturated datasets (3/13, ~100%
  single-outs) full replacement already floors linkability, so the Step 6b focus has no
  headroom and the method is ≈ stock. The lever is expected to matter on datasets with
  *moderate* single-out prevalence, where replacement touches few rows. Measuring it
  cleanly needs averaging several synthesis runs per arm (or seeding `np.random` in
  `privatesmote_old`), since synthesis is stochastic.
- **Detectors.** Only `Distance` and `Density` at the `Outlier` tier feed the mask —
  the tiers our correlation analysis tied to linkability. `LOF`/`Tukey` and the
  `Rare`/`Borderline` tiers are intentionally excluded. Change the mask expression in
  `pipeline.py` to adjust.
- **Seed concentration is a knob.** Step 6b currently concentrates the whole budget on
  outlier single-outs. If that starves diversity in the rest of a cell, soften it to a
  *weighted* `np.random.choice` (flagged rows e.g. 3× weight) — a small addition to
  `over_sampling` in `privatesmote_old.py`, not a structural change here.
- **All-single-out cells** (e.g. datasets 3 and 13, where every minority row is a
  single-out): in-subgroup decoys are themselves sparse, so dilution is limited. Turn
  on `binning` (`uniform`/`quantile`/`kmeans`) to coarsen the QIs so exact groups can
  reach `k` and decoys land on shared tuples.
- **Graceful degradation.** If outlier detection fails for a fold, `outlier_mask` is
  `None`, `is_outlier` is all-zero, and the method behaves like the stock pipeline
  (replaces nothing extra; augmentation seeds fall back to all single-outs).
