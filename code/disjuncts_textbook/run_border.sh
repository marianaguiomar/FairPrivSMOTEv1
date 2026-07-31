#!/usr/bin/env bash
# Borderline-gating variant of FP-SMOTE on the small-disjunct stratum, end to end.
# Run from the repo root:   bash code/disjuncts_textbook/run_border.sh
#
# 1. synthesis (all 13 datasets)  2. impact  3. privacy (5-ds subset)  4. LaTeX table.
set -euo pipefail

GROUP="disjunct_addon_border"
PY="priv39/bin/python"          # full env with anonymeter/sdmetrics/etc.

echo "==> [1/4] Synthesis: group '$GROUP' (this is the slow step)"
FPS_GROUP="$GROUP" \
FPS_DISJUNCT_AWARE=1 FPS_DISJUNCT_ADDON=1 FPS_DISJUNCT_BORDERLINE=1 \
  "$PY" code/disjunct_mitigation/pipeline.py

echo "==> [2/4] Impact (fairness/utility, all 13 datasets)"
"$PY" code/disjuncts_textbook/impact.py "$GROUP"

echo "==> [3/4] Privacy (link/inf, 5-dataset subset)"
"$PY" code/disjuncts_textbook/privacy_impact.py "$GROUP" law,adult,33,37,3

echo "==> [4/4] LaTeX table"
"$PY" code/disjuncts_textbook/build_appendix_table.py "$GROUP" "Borderline-gating" \
  | tee exploratory_metadata/disjunct_addon_border_table.tex

echo
echo "Done. Table written to exploratory_metadata/disjunct_addon_border_table.tex"
echo "(also printed above) -- paste it into _latex/text/9-appendices.tex"
