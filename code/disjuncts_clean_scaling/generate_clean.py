"""Step 1 -- generate the #2 (ENN-clean) synthetic outputs for every DATASET in config.

For each dataset that does NOT already have clean synth, copy its input CSV into the repo-root
staging folder, then run the disjunct_mitigation pipeline once with the #2 env flags set
(FPS_DISJUNCT_AWARE + ADDON + CLEAN). Stock synth (small_disjuncts) is assumed already present
for all datasets -- this script only produces the clean arm.

Run from repo root (priv39):
    priv39/bin/python code/disjuncts_clean_scaling/generate_clean.py            # skip done ones
    priv39/bin/python code/disjuncts_clean_scaling/generate_clean.py --force    # regenerate all
"""
import os
import sys
import glob
import shutil
import importlib.util

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import config as C   # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)

# ---- #2 flags MUST be set before the pipeline module is imported / used ----
os.environ["FPS_DISJUNCT_AWARE"] = "1"
os.environ["FPS_DISJUNCT_ADDON"] = "1"
os.environ["FPS_DISJUNCT_CLEAN"] = "1"
os.environ.setdefault("FPS_DISJUNCT_COVERAGE", "0.2")


def has_clean_synth(dataset):
    return bool(glob.glob(
        f"datasets/outputs/outputs_4/{C.CLEAN_GROUP}/{dataset}/fold5/*.csv"))


def main():
    force = "--force" in sys.argv
    todo = [d for d in C.DATASETS if force or not has_clean_synth(d)]
    skipped = [d for d in C.DATASETS if d not in todo]
    if skipped:
        print(f"[gen] already have clean synth, skipping: {skipped}")
    if not todo:
        print("[gen] nothing to generate -- all datasets already have clean synth.")
        return

    os.makedirs(C.STAGE_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(C.STAGE_DIR, "*.csv")):
        os.remove(f)
    for d in todo:
        shutil.copy(os.path.join(C.INPUT_FOLDER, f"{d}.csv"),
                    os.path.join(C.STAGE_DIR, f"{d}.csv"))
    print(f"[gen] staged {len(todo)} datasets -> {C.STAGE_DIR}: {todo}")

    spec = importlib.util.spec_from_file_location(
        "dm_pipeline", "code/disjunct_mitigation/pipeline.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    # the pipeline's own metric calls are skipped; eval is done by eval_impact.py
    P.process_fairness = P.process_fairness_for_seeds = P.process_linkability = \
        lambda *a, **k: None

    P.method_3(C.STAGE_DIR, [1.0], [3], [5], [0.4], final_folder_name=C.CLEAN_GROUP)

    # tidy the staging folder
    shutil.rmtree(C.STAGE_DIR, ignore_errors=True)
    print(f"[gen] done -> datasets/outputs/outputs_4/{C.CLEAN_GROUP}/<dataset>/")


if __name__ == "__main__":
    main()
