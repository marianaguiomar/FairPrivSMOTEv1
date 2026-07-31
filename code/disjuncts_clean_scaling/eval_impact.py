"""Step 2 -- compute the stratified error metrics for stock and clean, per dataset.

Reuses code/disjuncts_textbook/impact.py (err / recallMin / AUPRC / Brier on the small stratum,
pooled across folds). Calls it with `only=<dataset>` so each dataset is UPSERTed into the group
CSV without recomputing the others. Writes/updates:
    exploratory_metadata/disjuncts_impact_small_disjuncts.csv      (stock)
    exploratory_metadata/disjuncts_impact_disjunct_addon_clean.csv (clean)

Run from repo root (priv39):
    priv39/bin/python code/disjuncts_clean_scaling/eval_impact.py
Optional: set FPS_RF_SEED to change the classifier seed (default 57).
"""
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join("code", "disjuncts_textbook"))

import config as C   # noqa: E402
import impact as I   # noqa: E402  (brings in the new err/recallMin/AUPRC/Brier metrics + `only`)


def main():
    for ds in C.DATASETS:
        for group in (C.STOCK_GROUP, C.CLEAN_GROUP):
            print(f"\n--- impact {group} / {ds} ---")
            I.run(group=group, only=ds)


if __name__ == "__main__":
    main()
