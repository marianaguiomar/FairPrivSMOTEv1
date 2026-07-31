"""
Small-disjunct detection -- TEXTBOOK (Holte 1989 / Weiss 2000) definition.

A *disjunct* is a single term of the disjunctive concept a learner induces. With a
decision tree, every LEAF is a disjunct and its *size* is the number of training rows
it covers. A row lies in a SMALL disjunct when its leaf is small.

Holte's original definition uses an ABSOLUTE floor (a leaf covering fewer than ~5-15
rows is "small"); Weiss generalised this to relative size. We lead with the absolute
floor and expose ``leaf_size`` so the threshold can be re-swept without refitting.

This is purely the classifier-relative (hypothesis-relative) definition: it knows about
the CLASS label (y) only -- it is deliberately blind to the protected attribute and to
the (class x protected) subgroup structure. That keeps it the canonical reference point.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold


def encode_features(df):
    """Return an all-numeric copy of df for tree fitting.

    Numeric columns pass through; everything else (strings, dates) is coerced to
    numeric where possible, otherwise factorised to integer codes. Categorical
    *order* is not meaningful to a decision tree's threshold splits, so codes are fine.
    """
    cols = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            cols[col] = s.to_numpy()
            continue
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().mean() > 0.9:          # mostly numeric -> use it
            cols[col] = coerced.fillna(coerced.median()).to_numpy()
        else:                                      # genuine categorical / date string
            cols[col] = pd.factorize(s, sort=True)[0]
    return pd.DataFrame(cols, index=df.index)


def fit_disjunct_tree(X, y, min_samples_leaf=2, random_state=42, ccp_alpha=0.0):
    """Fit the reference decision tree whose leaves define the disjuncts.

    ``ccp_alpha`` > 0 applies cost-complexity (post-)pruning, which is what the textbook
    definition assumes: Holte/Weiss measured disjuncts on the tree a learner would actually
    deploy (a pruned C4.5), not a tree grown to purity. With ccp_alpha=0 the tree shatters
    into thousands of size-1..2 leaves and "small disjunct" degenerates to "almost every row".
    """
    return DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        ccp_alpha=ccp_alpha,
    ).fit(np.asarray(X, dtype=float), np.asarray(y))


def flag_from_tree(tree, X, train_leaf_sizes, leaf_abs=10):
    """Flag rows of X using a tree fitted elsewhere, with leaf sizes measured on TRAIN.

    Used to label held-out rows by the same disjunct structure the model learned, so a
    test row is "in a small disjunct" iff it falls into a leaf that was small in training.
    """
    leaf_id = tree.apply(np.asarray(X, dtype=float))
    leaf_size = pd.Series(leaf_id).map(train_leaf_sizes).fillna(0).to_numpy().astype(int)
    return pd.DataFrame(
        {"leaf_id": leaf_id, "leaf_size": leaf_size,
         "small_disjunct": (leaf_size < leaf_abs).astype(int)},
        index=getattr(X, "index", None),
    )


def detect(X, y, leaf_abs=10, min_samples_leaf=2, random_state=42, ccp_alpha=0.0):
    """One-shot textbook detection on (X, y). Returns per-row leaf_id/leaf_size/small_disjunct.

    ``leaf_size`` is the TRAINING coverage of each row's leaf; ``small_disjunct`` = size < leaf_abs.
    """
    tree = fit_disjunct_tree(X, y, min_samples_leaf, random_state, ccp_alpha)
    leaf_id = tree.apply(np.asarray(X, dtype=float))
    train_leaf_sizes = pd.Series(leaf_id).value_counts()
    flags = flag_from_tree(tree, X, train_leaf_sizes, leaf_abs=leaf_abs)
    return flags, tree, train_leaf_sizes


# ======================================================================================
# Weiss & Hirsh (2000) error-concentration methodology
# --------------------------------------------------------------------------------------
# The textbook *quantitative* study of small disjuncts uses the DEPLOYED (pruned) tree and
# asks how concentrated the model's errors are in its smallest disjuncts -- a single,
# threshold-robust number -- rather than binary-flagging leaves on a tree grown to purity.
# ======================================================================================

def select_ccp_alpha(X, y, cv=3, random_state=42, one_se=True, max_alphas=40):
    """Pick the cost-complexity pruning level a learner would actually deploy.

    Cross-validate over the pruning path and take the alpha with best mean CV accuracy,
    or (default) the 1-SE rule: the simplest tree within one standard error of the best.
    This gives the pruned tree the textbook definition assumes, with no arbitrary floor.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    full = DecisionTreeClassifier(random_state=random_state).fit(X, y)
    alphas = np.unique(full.cost_complexity_pruning_path(X, y).ccp_alphas)
    alphas = alphas[alphas >= 0]
    if len(alphas) > max_alphas:                       # subsample for speed
        alphas = alphas[np.linspace(0, len(alphas) - 1, max_alphas).astype(int)]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    means, ses = [], []
    for a in alphas:
        sc = [DecisionTreeClassifier(random_state=random_state, ccp_alpha=a)
              .fit(X[tr], y[tr]).score(X[va], y[va])
              for tr, va in skf.split(X, y)]
        means.append(np.mean(sc)); ses.append(np.std(sc) / np.sqrt(cv))
    means, ses = np.array(means), np.array(ses)
    best = int(means.argmax())
    if one_se:
        ok = np.where(means >= means[best] - ses[best])[0]
        return float(alphas[ok[np.argmax(alphas[ok])]])    # simplest within 1 SE
    return float(alphas[best])


def small_flag_by_coverage(leaf_size_per_row, coverage_frac=0.2):
    """Flag rows in the smallest disjuncts that together cover the bottom ``coverage_frac``
    of examples. Returns (flag[int], size_threshold). Whole leaves are kept together, so
    realized prevalence is ~coverage_frac (a touch more at the tie boundary)."""
    s = np.asarray(leaf_size_per_row)
    n = len(s)
    if n == 0:
        return np.zeros(0, int), 0
    order = np.argsort(s, kind="mergesort")
    rank = np.searchsorted(np.arange(1, n + 1) / n, coverage_frac)
    size_threshold = int(s[order][min(rank, n - 1)])
    return (s <= size_threshold).astype(int), size_threshold


def ec_curve(leaf_size_per_row, error_per_row):
    """Weiss & Hirsh (2000) error-concentration curve.

    For each disjunct size n, plot the percentage of test ERRORS (y) against the percentage of
    correctly-classified test examples (x) covered by disjuncts of size <= n. Returns (x, y)
    with both axes as cumulative fractions in [0, 1], aggregated per disjunct size (ascending).
    """
    s = np.asarray(leaf_size_per_row)
    e = np.asarray(error_per_row, dtype=float)
    c = 1.0 - e                                       # correctly classified
    if len(e) == 0 or e.sum() == 0 or c.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    g = (pd.DataFrame({"s": s, "e": e, "c": c})
         .groupby("s")[["e", "c"]].sum().sort_index())
    x = np.concatenate([[0.0], np.cumsum(g["c"].to_numpy()) / g["c"].sum()])
    y = np.concatenate([[0.0], np.cumsum(g["e"].to_numpy()) / g["e"].sum()])
    return x, y


def error_concentration(leaf_size_per_row, error_per_row):
    """Weiss & Hirsh error concentration, a FRACTION in [-1, +1] (Weiss 2010, Sec. 2).

    EC is the fraction of the total area above the diagonal Y=X that falls below the error
    concentration curve: EC = (area_under_curve - 0.5) / 0.5. EC = 0 means errors are spread
    evenly across disjunct sizes; EC -> +1 means all errors come from the smallest disjuncts
    before a single correct example is covered; EC -> -1 means errors come from the largest.
    """
    x, y = ec_curve(leaf_size_per_row, error_per_row)
    return float(2.0 * np.trapz(y, x) - 1.0)


def pct_errors_at_correct(leaf_size_per_row, error_per_row, correct_frac=0.10):
    """Weiss' companion statistic: the percentage of test errors contributed by the smallest
    disjuncts that together cover ``correct_frac`` of the correctly-classified test examples."""
    x, y = ec_curve(leaf_size_per_row, error_per_row)
    return float(100.0 * np.interp(correct_frac, x, y))


def pct_correct_at_errors(leaf_size_per_row, error_per_row, error_frac=0.50):
    """Weiss' second companion statistic: the percentage of correctly-classified test examples
    covered by the smallest disjuncts that together contribute ``error_frac`` of the test errors."""
    x, y = ec_curve(leaf_size_per_row, error_per_row)
    return float(100.0 * np.interp(error_frac, y, x))


if __name__ == "__main__":
    # self-test: a big blob (two classes) + a tiny same-class pocket far away.
    rng = np.random.default_rng(0)
    big = rng.normal(0, 1, size=(200, 3))
    pocket = rng.normal(8, 0.2, size=(4, 3))          # tiny class-1 sub-concept
    X = pd.DataFrame(np.vstack([big, pocket]), columns=list("abc"))
    y = np.r_[np.zeros(150), np.ones(50), np.ones(4)].astype(int)
    flags, _, sizes = detect(X, y, leaf_abs=10)
    print(flags.tail(6))
    print("\nsmall-disjunct rows:", int(flags.small_disjunct.sum()),
          "| pocket flagged:", int(flags.small_disjunct.to_numpy()[-4:].sum()), "/4")
