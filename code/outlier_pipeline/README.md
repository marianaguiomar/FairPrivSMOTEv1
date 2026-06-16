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
3. **Step 5 (majority) & Step 6a (minority)** — replacement is **retargeted** from
   *all* single-outs to the intersection. This is a 1:1 replacement, so it is
   **count-neutral**: zero budget cost, subgroup sizes unchanged. (`df_rest + replaced
   == df_subset`.) When a cell has no outlier single-outs, the originals are kept and
   the fold is **not** aborted.
4. **Step 6b (augmentation)** — the **fixed** `samples_to_increase` budget is *seeded*
   from the intersection rows (via the `highest_risk` selector inside
   `newPrivateSMOTE.over_sampling`), so synthetic decoys densify/dilute the most
   linkable rows. Total generated rows is unchanged → **budget unchanged**. Falls back
   to all single-outs when the intersection is empty in a cell.
5. **Step 7.5 / Step 8** — `is_outlier` is excluded from the Tomek-link feature matrix
   and dropped from the final output (like `single_out`/`synthetic`).

## Why it respects the fairness cap

| operation | rows added | budget cost |
|---|---|---|
| Step 5 majority replace   | 0 (1:1 swap)                       | none |
| Step 6a minority replace  | 0 (`df_rest + replaced == df_subset`) | none |
| Step 6b augmentation      | exactly `samples_to_increase`      | unchanged |

Final per-subgroup counts are **identical** to the stock pipeline
(`len(df_subset) + samples_to_increase`), so DI/SPD/AOD targets do not move. The
de-isolation comes entirely from (i) *free* count-neutral replacement of the
`outlier ∩ single-out` rows and (ii) *redirecting* the existing augmentation budget
toward them — never from extra rows.

## Design notes / knobs

- **Why the intersection, not all single-outs.** Anonymeter linkability matches on QI
  *near*-neighbours, so an exact-match single-out sitting in a crowded QI region is
  already hard to link. The rows that actually leak are those that are both QI-unique
  **and** geometric outliers. Targeting the intersection also shrinks the perturbation
  footprint (better utility). **To revert to replacing all single-outs**, set
  `sub_target = (df_subset['single_out'] == 1)` in Step 6a (and the majority analogue
  in Step 5).
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
