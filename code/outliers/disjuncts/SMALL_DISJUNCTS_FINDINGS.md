# Small disjuncts in FairPrivSMOTE: diagnostic, not a lever

A *small disjunct* (Holte et al. 1989; Weiss 2010) is a small, **same-class** pocket of the
feature space — a sub-concept the classifier must cover with very few training examples.
Unlike an outlier (geometric isolation, unsupervised), a small disjunct is **label-aware**.

**One-line result:** small disjuncts are a reliable *diagnostic* for where FairPrivSMOTE's
fairness and utility degrade, but **not an actionable lever** — once the evaluation is properly
powered, no mitigation we tested improves small-disjunct fairness, and none changes utility (F1).

## Definitions (`small_disjuncts.py::SmallDisjunctDetector`)

| name | how | rationale |
|------|-----|-----------|
| `leaf` | fit a decision tree on (X, y); a row is in a small disjunct if its **leaf covers < 10 training rows** | classic Holte absolute-size definition; hypothesis-relative |
| `cluster` | agglomerative clustering **within each (class × protected) subgroup**; clusters below `0.2·largest` (floor 5) are small | conditions on the subgroup structure FairPrivSMOTE balances |

Run end-to-end (all from repo root, `priv39` env):

```bash
priv39/bin/python code/outliers/disjuncts/detect_small_disjuncts.py        # per-fold flags -> exploratory_metadata/<ds>/fold_<N>_disjuncts.csv
priv39/bin/python code/outliers/disjuncts/correlate_disjuncts_linkability.py    # finding 1 (privacy)
priv39/bin/python code/outliers/disjuncts/analyse_disjunct_fairness.py          # finding 2 (fairness/utility)
priv39/bin/python code/outliers/disjuncts/mitigation_disjuncts.py cluster        # finding 3, per-dataset (and: ... leaf)
priv39/bin/python code/outliers/disjuncts/evaluate_disjuncts_powered.py cluster  # finding 3, POWERED + F1 (and: ... leaf)
priv39/bin/python code/outliers/disjuncts/summarise_small_disjuncts.py          # headline table
```

## Headline result (13 datasets, `datasets/inputs/test`)

| Definition | Linkability $r_{\text{fold}}$ | Error ratio small/large | Group gap small / large | Best mitigation (powered) ΔEOD / ΔF1 |
|---|---|---|---|---|
| leaf | +0.18 (n.s.) | **14.9×** (12/13) | 0.09 / 0.04 | +0.006 / +0.003 |
| cluster | −0.24 (n.s.) | 2.6× (9/13) | **0.18 / 0.06** | −0.017 / +0.002 |
| *isolated outlier (ref.)* | **+0.40 (p=0.001)** | — | — | — |

### 1. Small disjuncts are orthogonal to privacy
A dataset's Anonymeter linkability rises with its share of **isolated outliers** (r=+0.40,
p=0.001) but **not** with its share of small disjuncts (leaf +0.18 n.s.; cluster −0.24 n.s.).
This extends the existing outlier finding (only the geometric "Outlier" tier tracks
linkability): small disjuncts are a *different axis* — a fairness phenomenon, not a privacy one.
→ `plots/disjuncts/risk/`

### 2. They are where fairness *and utility* break
Out-of-fold RandomForest error concentrates sharply in small disjuncts (leaf median **14.9×**
the large-disjunct error, on 12/13 datasets; cluster 2.6×, 9/13), and the protected-vs-
unprotected **error gap is ~2–3× larger inside** them. The utility view agrees: pooled
**F1 ≈ 0.73 on small disjuncts vs ≈ 0.81 elsewhere**. `leaf` best flags raw error; `cluster`
best flags the fairness gap (it conditions on the subgroup). → `plots/disjuncts/fairness/`

### 3. No mitigation works — and the apparent win was a measurement artifact
We tried four oversampling/weighting **arms** at equal subgroup-balancing target
(`mitigation_disjuncts.py`), then evaluated them properly (`evaluate_disjuncts_powered.py`):

| arm | mechanism |
|-----|-----------|
| `uniform` | oversample subgroups to balance (the method baseline) |
| `coherent` | oversample only small disjuncts that are NOT geometric outliers (gate out likely-noise via `TypologyDetector`) |
| `weight_bal` | no synthesis; sample-weight subgroups to balance (cost-sensitive analogue of `uniform`) |
| `weight_disj` | `weight_bal` + extra ×3 weight on small-disjunct rows |

**Per-dataset metric (underpowered) said `weight_bal` beat uniform (+0.065).** But the
small-disjunct test subsets are tiny: the per-group recall gap saturates (±0.333/±0.5), with
many ties and NaNs. **Pooling small-disjunct test rows across datasets** (cluster 9,183; leaf
40,794) and scoring the large-sample **group error gap** + **F1** dissolves that result:

- The `weight_bal` "win" reverses — pooled, it *worsens* EOD on small disjuncts (−0.016 → +0.061).
- The only positive signal — `weight_disj` cutting the cluster small-disjunct **error gap**
  0.030 → 0.005 — **fails on EOD** (the repo's primary fairness metric) and **vanishes under
  the leaf definition** (all arms within ±0.006). Metric-specific, non-replicating.
- **Utility is flat**: F1 0.805–0.809 overall, 0.73–0.74 on small disjuncts, across *all* arms
  (±0.004). Nothing harms utility; nothing helps fairness.
- The earlier `law` swing was **not** a degenerate all-negative prediction (22 pos-true,
  24–27 pred-pos of 33) — just tiny-subset noise, now superseded.

→ `plots/disjuncts/mitigation/` (`mitigation_disjuncts_*.csv`, `powered_*_pooled.csv`,
`powered_*_per_dataset_f1.csv`)

## Conclusion
Across naive-focused, coherence-gated, cost-sensitive-balanced, and small-disjunct-boosted
interventions, **no arm robustly improves small-disjunct fairness, and utility never moves.**
Small disjuncts mark *where* FairPrivSMOTE fails but are not a usable *handle* for fixing it.

**Methodological lesson (worth a sentence in the paper):** the underpowered per-dataset recall
gap manufactured a false-positive mitigation result that pooling + the group error gap + a check
against EOD removed. Report pooled, large-sample fairness metrics for rare strata, not per-dataset
saturating ones.
