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
2. **Step 1** — attach `dataset['is_outlier']`, then **de-isolate `single_out ∩ outlier`
   rows in place** (`_deisolate_outlier_singleouts`): for each `(class, protected)`
   subgroup, move those rows' **continuous** QI columns toward the **centroid of their k
   in-subgroup neighbours** by a random `gap~U(0,1)` (SMOTE-style, no tunable strength)
   plus Laplace(0,1/ε) DP noise. Pulling toward the centroid collapses clusters of
   isolated outliers toward a shared point — that is what reduces QI isolation. **Never**
   moves the class/protected columns (even when the protected attr is itself a QI, e.g.
   `sex` in german) and skips binary/low-card QIs. Count-neutral, in-subgroup.
3. **Step 5 (majority) & Step 6a (minority)** — standard PrivateSMOTE replacement covers
   the **non-outlier** single-outs (`single_out & ~is_outlier`); the outlier single-outs
   are already de-isolated in step 2 and kept as originals, so they are excluded here to
   avoid double-processing. Together, **every** single-out is de-isolated — via two
   mechanisms. 1:1 replacement → **count-neutral** (`df_rest + replaced == df_subset`).
   Also more robust: too-few-single-outs no longer aborts the fold.
   **Do NOT** narrow standard replacement to drop the non-outlier single-outs — on
   high-single-out datasets that spikes linkability (validated: ~0.05 → ~0.80).
4. **Step 6b (augmentation)** — the **fixed** `samples_to_increase` budget is allocated by
   `_focused_seed_indices`: (1) cover each outlier single-out once, (2) cover each other
   single-out once, (3) focus the extra on outliers up to **1.5× of all single-outs**,
   (4) spill the remainder uniformly across all single-outs. Seeds (with multiplicity) are
   handed to the local `FocusedPrivateSMOTE`. Total generated == `samples_to_increase` →
   **budget unchanged**. The "1.5×" is a fixed rule, not a swept parameter.
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
(`len(df_subset) + samples_to_increase`), so DI/SPD/AOD targets do not move. De-isolation
comes from (i) *in-place* centroid relocation of the outlier single-outs (count-neutral),
(ii) standard count-neutral replacement of the other single-outs, and (iii) *re-targeting*
the existing augmentation budget — never from extra rows.

## Evaluation status (honest)

Correctness is solid: runs on every dataset, per-fold row counts match stock exactly
(budget/fairness preserved), and the protected attribute is never moved (the de-isolation
bug where `sex` got perturbed is fixed). **Privacy benefit is not yet proven.** In
stock-vs-new linkability checks the effect is small and dominated by synthesis RNG noise
(`privatesmote` doesn't seed `np.random`): single-run 3/13 comparisons flip sign between
runs; a 5-run average on german/55 shows new ≤ stock on 8/10 folds with 55 lower on all
5, but the effect size is comparable to the noise. **To get a trustworthy verdict, run a
paired comparison with `np.random` seeded identically per (fold, run) for both arms, and
average more runs.** Until then, treat the linkability improvement as *plausible but
unconfirmed*; the fairness/budget guarantees are confirmed.

## Design notes / knobs

- **Why standard replacement still covers the non-outlier single-outs.** Dropping them
  (replacing only outlier single-outs) backfires on high-single-out datasets: leaving raw
  single-out originals in the release spikes linkability (3/13 went ~0.05 → ~0.80). So all
  single-outs are de-isolated — outlier ones via centroid relocation, the rest via stock
  PrivateSMOTE replacement.
- **Continuous-only de-isolation.** The centroid pull moves only continuous QI columns
  (from `continuous_attributes.csv`, or a >2-unique-values fallback), never class/protected
  and never binary/low-card QIs — moving a 0/1 QI to a fractional value is meaningless and
  (when it is the protected attr) shatters the subgroup grouping.
- **No tunable pull strength.** The interpolation fraction is a random `U(0,1)`; the
  centroid target (not a single neighbour) is what does the de-isolation, so there is no
  λ to sweep.
- **Detectors.** Only `Distance`/`Density` at the `Outlier` tier feed the mask — the tiers
  the correlation analysis tied to linkability. `LOF`/`Tukey`, `Rare`/`Borderline` excluded.
- **Augmentation focus cap.** `_focused_seed_indices` caps outlier concentration at 1.5× of
  all single-outs (fixed rule) before spilling across all single-outs — avoids piling many
  near-duplicate decoys on a few rows.
- **All-single-out cells** (3/13): in-subgroup neighbours of an outlier are themselves
  sparse, so the centroid pull collapses them together (helps) but dilution is limited;
  turn on `binning` to coarsen QIs so exact groups can reach `k`.
- **Graceful degradation.** If outlier detection fails for a fold, `outlier_mask` is `None`,
  `is_outlier` is all-zero, no de-isolation happens, and the method reduces to stock.
