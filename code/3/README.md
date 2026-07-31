# code/3 — subgroup-mitigation pipeline (dataset 3)

A self-contained pipeline for the tiny-subgroup methods developed for dataset 3,
where the `(Class=1, V25=1)` subgroup has only ~6–8 rows and would otherwise be
oversampled 20–28× into a degenerate blob. Kept separate so `code/main` stays the
clean baseline (same split as the `code/pca_pipeline` arrangement).

## Methods

Selected by the `method` variable near the bottom of `pipeline.py`; it sets the
env-var toggles read at runtime by `generate_samples.new_apply`:

| method         | toggle(s)                                             | what it does |
|----------------|-------------------------------------------------------|--------------|
| `baseline`     | (none)                                                | unchanged FairPrivSMOTE control |
| `borrow`       | `FPS_BORROW_NEIGHBORS=1`, `FPS_BORROW_THRESHOLD=25`   | seed from the subgroup's single-outs but draw kNN neighbors from the whole same-class pool, then stamp the protected attr back. Improves EOD/F1, privacy ~flat. |
| `undersample`  | `FPS_UNDERSAMPLE=1`, `FPS_UNDERSAMPLE_TARGET=120`     | randomly downsample every subgroup above the target, shrinking the oversampling burden. Strongest EOD lever; worsens SPD. |

**"borrow + threshold"** is not a separate synthesis method — it's `borrow` followed
by the per-group decision-threshold post-processor (see below). The threshold fixes
DI/SPD at decision time and does not change synthesis.

(`FPS_BORROW_THRESHOLD` is a subgroup-*size* gate — borrow only fires for subgroups
smaller than that many rows — and is unrelated to the decision threshold. A cap lever
`FPS_CAP_MULTIPLE`/`FPS_CAP_FACTOR` also exists but was weak/privacy-hurting and is not
wired into the selector.)

## Running a full experiment

From the **repo root**, once per method (each writes to its own output folder):

```bash
# edit `method = "..."` in code/3/pipeline.py, then:
priv39/bin/python code/3/pipeline.py
```

- Input:  `datasets/inputs/3/3.csv`  (created for this; mirror it to add datasets)
- Grid:   eps {0.1,0.5,1,5,10} × k {3,5} × knn {3,5} × aug {0.3,0.4}, 5 folds, all QI sets
- Output synthetic: `datasets/outputs/outputs_4/3_<method>/3/foldN/...`
- Metrics: `results_metrics/fairness_results/outputs_4/3_<method>/` (AOD/EOD/SPD/DI/F1)
           and `results_metrics/linkability_results/outputs_4/3_<method>/` (linkability, inference, ...)

Outlier detection (`TypologyDetector`) is intentionally **disabled** here.

## The threshold variant ("with / without the threshold thing")

DI / SPD are decision-stage properties, not synthesis ones, so the per-group
threshold is a **post-processing** step — run it on a finished method's output:

```bash
priv39/bin/python code/3/apply_group_threshold.py 3_borrow
priv39/bin/python code/3/apply_group_threshold.py 3_undersample --alt-thresh 0.15
```

It re-scores each generated file with a lower decision threshold for the
disadvantaged protected group (default value `1`, threshold `0.20`), driving DI
toward 1, and writes `results_metrics/fairness_results/outputs_4/<folder>_thresh/
group_threshold_results.csv`. "Without threshold" = just read the pipeline's own
fairness output; "with threshold" = this script's output.

## Files

- `pipeline.py`         — copy of the main pipeline; method selector + full grid; outliers off.
- `generate_samples.py` — the method changes live here (borrow / undersample / cap, all env-gated).
- `pipeline_helper.py`  — baseline copy (needed because `pipeline.py` bare-imports it; running
                          `code/3/pipeline.py` puts this dir first on `sys.path`).
- `apply_group_threshold.py` — the per-group threshold post-processor.

Everything else (`main.privatesmote*`, `metrics.*`, `others.*`) is shared with `code/main`.
See memory `tiny-subgroup-mitigation-3.md` for the small-scale findings behind these methods.
