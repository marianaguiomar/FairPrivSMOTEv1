"""One-shot runner: generate clean synth -> eval impact -> analyze.

Each step runs as its own subprocess (generation needs the #2 env flags set before the
pipeline import, so isolation matters). Run from repo root with the full env:

    priv39/bin/python code/disjuncts_clean_scaling/run_all.py
    priv39/bin/python code/disjuncts_clean_scaling/run_all.py --force   # regenerate all synth

The privacy stratum (linkability/inference) is NOT run here -- it's slow and orthogonal to the
error hypothesis. Run it separately if needed:
    priv39/bin/python code/disjuncts_textbook/privacy_impact.py disjunct_addon_clean <dataset>
"""
import os
import sys
import subprocess

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
force = ["--force"] if "--force" in sys.argv else []


def step(name, script, extra=()):
    print(f"\n{'='*70}\n[{name}] {script} {' '.join(extra)}\n{'='*70}", flush=True)
    r = subprocess.run([PY, os.path.join(HERE, script), *extra], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"[{name}] FAILED (exit {r.returncode})")


if __name__ == "__main__":
    step("generate", "generate_clean.py", force)
    step("eval", "eval_impact.py")
    step("analyze", "analyze.py")
    print("\nAll done. See exploratory_metadata/clean_scaling_summary.csv and "
          "clean_scaling_recall_vs_err.png")
