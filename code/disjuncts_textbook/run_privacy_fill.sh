#!/usr/bin/env bash
# Fill the missing small-disjunct PRIVACY evaluations for the combined
# focused-allocation + ENN-cleaning strategy, then regenerate the two LaTeX
# tables (overall metrics + per-dataset ranks) and print the significance verdict.
#
# One command, from the repo root:
#     bash code/disjuncts_textbook/run_privacy_fill.sh
#
# The combined strategy already had linkability/inference on 5 datasets
# (law, adult, Bank, Churn, QSAR); this fills the other 8. The synthetic outputs
# already exist (impact.py ran on all 13), so this only runs the anonymeter attacks.
set -euo pipefail

GROUP="disjunct_addon_clean"
PY="priv39/bin/python"                 # full env: anonymeter / sdmetrics / recordlinkage
MISSING="compas,student,55,credit,german,23,13,oulad"

echo "==> [1/2] Privacy (linkability + inference) on the 8 missing datasets for '$GROUP'"
"$PY" code/disjuncts_textbook/privacy_impact.py "$GROUP" "$MISSING"

echo "==> [2/2] Regenerate tables + significance verdict"
"$PY" code/disjuncts_textbook/summarise_mitigation.py

echo
echo "Done. The rank and overall tables in _latex/text/small_disjuncts.tex pick up the new"
echo "values automatically on the next LaTeX build (they \\input the regenerated bodies)."
