# pca_pipeline — PCA-experiment snapshot

Preserved copies of the `code/main` files that were modified for the PCA
neighbor-search experiments (Fix 1 = PCA-before-kNN, Fix 4 = latent-space
interpolation). These changes were removed from `code/main` on request; this
folder keeps the working PCA versions.

Files and what the PCA edits were:
- `pipeline_helper.py` — adds `reduce_for_neighbors()` (gated by `FPS_PCA_NEIGHBORS`
  / `FPS_PCA_VARIANCE`) and uses it in `build_fold_subgroup_cache`.
- `privatesmote_old.py` — `nearest_neighbours` projects through `reduce_for_neighbors`.
- `privatesmote.py` — `nearest_neighbours` projection + Fix 4 latent-space
  interpolation (gated by `FPS_PCA_INTERPOLATE`).
- `pipeline.py` — the PCA toggle config block (`use_pca_neighbors`, `pca_variance`,
  `FPS_PCA_*` env vars).

Note: this is a snapshot for preservation, not wired as a standalone runnable
package — it shares `generate_samples.py` and the rest of the pipeline with
`code/main`. To run the PCA variant, restore these files over `code/main` (or put
this directory first on `sys.path`) and set `FPS_PCA_NEIGHBORS=1`.

See memory `pca-neighbors-fix-ineffective-23.md` for the experiment findings.
