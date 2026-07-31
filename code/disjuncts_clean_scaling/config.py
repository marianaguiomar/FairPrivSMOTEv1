"""Shared configuration for the #2 (ENN-clean) scaling experiment.

Hypothesis under test
---------------------
ENN-cleaning the focused synthetic mass (#2 / FPS_DISJUNCT_CLEAN) reduces small-disjunct
classification error in proportion to a dataset's ABSOLUTE pocket error (err_small), NOT its
error CONCENTRATION (EC). i.e. it repairs pockets that are genuinely hard, and is harmless on
pockets the model already nails (high-EC-but-low-error "ceiling" datasets like compas).

Confirmed so far (single-dataset, seed-gated): on dataset 33 (err_small 0.354) #2 lifts
minority recall +0.027 / F1 +0.024 / err -0.016, robust across 10 RF seeds. This experiment
scales that to the full spread of datasets to see whether the slope vs err_small is real.
"""

# All `test`-group datasets EXCEPT adult (too large / intentionally dropped). The spread of
# err_small across these is what powers the correlation: high-error (law, 33, student, 23),
# mid (37, german, credit), low/ceiling (compas, oulad, 3, 13).
DATASETS = ["3", "13", "23", "33", "37", "55",
            "compas", "credit", "german", "law", "oulad", "student"]

STOCK_GROUP = "small_disjuncts"          # baseline FP-SMOTE synth (already generated for all)
CLEAN_GROUP = "disjunct_addon_clean"     # #2 = addon focused mass + Wilson-ENN cleaning

# repo-root staging folder the synthesis driver reads (method_3 lists this dir directly)
STAGE_DIR = "scratch_inputs_clean_scale"

INPUT_FOLDER = "datasets/inputs/test"
OUT_ROOT = "exploratory_metadata"
