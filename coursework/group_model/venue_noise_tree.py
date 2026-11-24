"""Group model assignment, v1: one decision tree that flags complaint-prone venues.

Question the model serves (Nightlife Autopilot product/ops lead): when a group asks where to go
tonight and we have no fresh Scout reports for a venue, should the app label it complaint-prone
(loud/rowdy) and demote it for groups who asked for a chill night?

Decision rule: if the tree predicts high_noise = 1, show a "complaint_risk" badge and multiply the
venue's rank score by RANK_PENALTY for chill-preference groups. Otherwise no badge.
Baseline: what the product does today, i.e. no badge for anyone (the most common class).

Data: REAL public data, table `venue_features` in data/manhattan_bars.db (one row per licensed
Manhattan on-premises venue; built by src/build_db.py from NY SLA licences, DOHMH inspections,
OpenStreetMap and NYC 311). Outcome `high_noise` = 1 when the venue's address has >= 3 NYC 311
noise complaints (location type Club/Bar/Restaurant) over the last 3 years, else 0
(src/build_db.py, build_features: threshold = 3).

Steps
  1. load venue_features read-only; drop the ID and the leakage column; one-hot the categoricals
  2. stratified 70/30 train/test split with --seed
  3. baseline = most common training class, predicted for every test row
  4. v1 = DecisionTreeClassifier(max_depth=4); min_samples_leaf picked by 5-fold CV on train only
  5. baseline vs v1 on the test rows: confusion matrix, accuracy, precision, recall; v1 train vs test
  6. decide(venue) on a realistic new venue (example_new_venue.json or --new-input)
  7. dollars per month, baseline vs v1, from the test confusion matrices and ASSUMPTIONS below
  8. write outputs/metrics.json, outputs/report.md and four PNG charts

Run (from the repo root):
  python coursework/group_model/venue_noise_tree.py
  python coursework/group_model/venue_noise_tree.py --seed 7 --new-input my_venue.json
"""
import argparse
import json
import platform
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = ROOT / "data" / "manhattan_bars.db"
OUT = HERE / "outputs"
EXAMPLE = HERE / "example_new_venue.json"

# ---------------------------------------------------------------------------------------------
# Business assumptions. Every figure here is MADE UP for the assignment (labelled ASSUMPTION);
# none comes from the data. Change them here and re-run: the value section uses nothing else.
# A "decision" = one group that asked for a chill night being sent to its top venue, where that
# venue has no fresh Scout reports (the only case the rule changes anything).
# ---------------------------------------------------------------------------------------------
ASSUMPTIONS = {
    "groups_per_month": 6000,          # ASSUMPTION: group requests per month in the launch area (LES)
    "share_chill_groups": 0.35,        # ASSUMPTION: share of groups that ask for a chill night
    "share_no_fresh_scout": 0.50,      # ASSUMPTION: share of top picks with no fresh Scout report
    "value_good_night_usd": 4.00,      # ASSUMPTION: referral fee + retention value of a night that fits
    "p_bad_night_if_loud": 0.50,       # ASSUMPTION: chance a complaint-prone venue is too loud that night
    "cost_bad_night_usd": 25.00,       # ASSUMPTION: group walks out and drops the app for the night
                                       #   (lost referral + some chance the group churns)
    "replacement_value_ratio": 0.85,   # ASSUMPTION: a demoted venue's replacement is worth 85% of a fit
    "cost_wrong_demotion_usd": 1.00,   # ASSUMPTION: false positive: a quiet partner venue loses the
                                       #   exposure (partner goodwill, listing fee risk)
}
RANK_PENALTY = 0.70  # ASSUMPTION: rank_score multiplier for badged venues, chill groups only
                     # (same multiplicative style as recs/scoring.py energy_penalty)

# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
TARGET = "high_noise"
ID_COL = "license_id"
# noise_complaints is the 311 count that high_noise is computed from (>= 3). Using it as a feature
# would hand the answer to the model (leakage), and a new venue would not have it anyway.
LEAKAGE = ["noise_complaints"]
CATEGORICAL = ["venue_type", "liquor_tier", "neighborhood", "cuisine", "latest_grade", "osm_amenity"]
# lat/lon are KEPT. 311 complaints depend on who lives nearby (residential blocks), which a 15-level
# neighbourhood only captures coarsely; the product has coordinates for every new licence address;
# and in a quick 5-fold CV on the training rows they raised balanced accuracy (about 0.62 -> 0.66).
# The cost is less readable splits ("lon <= -73.99"), which a depth-4 tree keeps to a few.
NUMERIC = ["dohmh_matched", "has_pest_violation", "osm_closes_late", "osm_outdoor_seating",
           "osm_has_website", "lat", "lon", "years_licensed", "n_additional_bars", "density_100m",
           "density_250m", "n_inspections", "mean_score", "max_score", "n_violations",
           "n_critical_violations"]
RAW_FEATURES = CATEGORICAL + NUMERIC

# v1 settings. Depth is fixed at 4 so the whole tree fits on one readable chart. Only
# min_samples_leaf is tuned. class_weight="balanced" because high_noise is 17% of rows: an
# unweighted tree, tuned for accuracy, flags almost no venue and so changes almost no decision.
MAX_DEPTH = 4
LEAF_GRID = [5, 10, 20, 50, 100]
CV_FOLDS = 5
CV_SCORING = "balanced_accuracy"  # mean of recall on each class; plain accuracy rewards "never flag"


def load_venues(db: Path = DB) -> pd.DataFrame:
    """Read venue_features read-only (mode=ro), so the database can never be changed by mistake."""
    if not db.exists():
        raise FileNotFoundError(f"database not found: {db}")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return pd.read_sql_query("SELECT * FROM venue_features", conn)


class Preprocessor:
    """Same preprocessing for training rows and for one new venue.

    Missing values (the committed table has none, because build_db.py already writes sentinels:
    scores -1 and counts 0 when no inspection matched; a new venue's dict may leave fields out):
      numeric NaN or missing key     -> the TRAINING median of that column
      categorical NaN or missing key -> the level "missing" (all one-hot columns 0 unless seen)
    One-hot columns are fixed from the training rows; a category never seen in training becomes
    all zeros, so a new venue always gets exactly the columns the tree was fitted on.
    """

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.medians = df[NUMERIC].median()
        self.columns = list(self._encode(df).columns)
        return self

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=RAW_FEATURES)
        num = df[NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        cat = df[CATEGORICAL].astype(object).where(df[CATEGORICAL].notna(), "missing").astype(str)
        return pd.concat([num, pd.get_dummies(cat, columns=CATEGORICAL, prefix_sep="=", dtype=int)], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._encode(df).reindex(columns=self.columns, fill_value=0)


def split(df: pd.DataFrame, seed: int):
    """Random 70/30 split, stratified on high_noise so both parts keep the 17% rate."""
    return train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)


def fit_v1(X: pd.DataFrame, y: pd.Series, seed: int):
    """Pick min_samples_leaf by 5-fold CV on the TRAINING rows only, then refit on all of them."""
    grid = GridSearchCV(
        DecisionTreeClassifier(max_depth=MAX_DEPTH, class_weight="balanced", random_state=seed),
        {"min_samples_leaf": LEAF_GRID},
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed),
        scoring=CV_SCORING, refit=True)
    grid.fit(X, y)
    cv = {int(p["min_samples_leaf"]): round(float(s), 4)
          for p, s in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"])}
    return grid.best_estimator_, int(grid.best_params_["min_samples_leaf"]), cv


def scores(y_true, y_pred) -> dict:
    """Confusion matrix (rows = actual 0/1, columns = predicted 0/1) and metrics for class 1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            # baseline never predicts 1, so its precision is 0/0: reported as 0 (zero_division=0)
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4)}


# Filled in by main() after fitting; decide() uses it when no model is passed explicitly.
FITTED: dict = {}


def decide(venue: dict, model: DecisionTreeClassifier = None, prep: Preprocessor = None) -> dict:
    """The decision rule for ONE new venue the product sees (raw venue_features columns).

    Returns {"badge": "complaint_risk" | "none", "rank_penalty": multiplier on rank_score
    (applied only for chill-preference groups; 1.0 = no change), "p_high_noise": tree probability}.
    p_high_noise is the class-weighted leaf share, so read it as a risk score, not a calibrated rate.
    """
    model = model or FITTED["model"]
    prep = prep or FITTED["prep"]
    row = prep.transform(pd.DataFrame([{k: venue.get(k) for k in RAW_FEATURES}]))
    p = float(model.predict_proba(row)[0, 1])
    flagged = int(model.predict(row)[0]) == 1
    return {"badge": "complaint_risk" if flagged else "none",
            "rank_penalty": RANK_PENALTY if flagged else 1.0,
            "rank_penalty_applies_to": "chill_preference_groups",
            "p_high_noise": round(p, 4)}


def monthly_value(cm: dict, n_test: int, a: dict = ASSUMPTIONS) -> dict:
    """Expected dollars per month from chill-group decisions, using test confusion-matrix rates.

    Per decision, by cell (actual, predicted):
      quiet, no badge (TN)       : +value_good_night
      loud,  no badge (FN)       : p_bad * (-cost_bad_night) + (1 - p_bad) * value_good_night
      loud,  badge    (TP)       : demoted, group goes to a replacement: ratio * value_good_night
      quiet, badge    (FP)       : replacement, minus the wrong-demotion cost
    Baseline has no badges, so all its loud venues are in the FN cell. Groups that did not ask
    for a chill night are not demoted under either rule, so they add the same amount to both
    and are left out. Note: the test rows are all of Manhattan, while the launch area is the LES,
    where the high_noise rate is higher (34% vs 17%), so these rates understate loud-venue cases.
    """
    v, c, pb, r, fpc = (a["value_good_night_usd"], a["cost_bad_night_usd"], a["p_bad_night_if_loud"],
                        a["replacement_value_ratio"], a["cost_wrong_demotion_usd"])
    per_cell = {"tn": v, "fn": pb * -c + (1 - pb) * v, "tp": r * v, "fp": r * v - fpc}
    decisions = a["groups_per_month"] * a["share_chill_groups"] * a["share_no_fresh_scout"]
    per_decision = sum(cm[k] / n_test * per_cell[k] for k in per_cell)
    return {"decisions_per_month": round(decisions, 1),
            "usd_per_decision_by_cell": {k: round(x, 4) for k, x in per_cell.items()},
            "usd_per_decision": round(per_decision, 4),
            "usd_per_month": round(per_decision * decisions, 2)}


# ---------------------------------------------------------------------------------------------
# Charts: plain matplotlib, black on white, labelled axes, no headline titles.
# ---------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "black", "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150})


def chart_confusion(base: dict, v1: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"wspace": 0.55})
    for ax, (name, s) in zip(axes, [("Baseline: no badge for any venue", base), ("v1: decision tree", v1)]):
        cm = np.array(s["confusion_matrix"])
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.sum() * 1.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15)
        ax.set_xticks([0, 1], ["no badge (0)", "complaint risk (1)"])
        ax.set_yticks([0, 1], ["not high noise (0)", "high noise (1)"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual (test rows)")
        ax.set_title(name, fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.9)
    fig.savefig(path)
    plt.close(fig)


def chart_bars(base: dict, v1: dict, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for off, (name, s, color) in zip([-0.18, 0.18], [("baseline", base, "#bdbdbd"), ("v1 tree", v1, "#404040")]):
        bars = ax.bar(x + off, [s[m] for m in metrics], width=0.36, label=name, color=color, edgecolor="black")
        ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(x, [f"{m}\n(high_noise class)" if m != "accuracy" else m for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score on test rows")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_tree(model, columns, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(30, 12))
    plot_tree(model, feature_names=columns, class_names=["not high noise", "high noise"], filled=False,
              impurity=False, proportion=True, rounded=True, precision=2, fontsize=10, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_importance(imp: pd.Series, path: Path) -> None:
    imp = imp[imp > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color="#404040", edgecolor="black")
    ax.set_xlabel("feature importance (share of impurity reduction)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(rows: list, header: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="seed for the split, the CV folds and the tree (default 42)")
    ap.add_argument("--new-input", type=Path, default=EXAMPLE,
                    help="JSON file with one new venue's raw features (default: example_new_venue.json)")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: coursework/group_model/outputs)")
    args = ap.parse_args(argv)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. load
    df = load_venues()
    df = df.drop(columns=[ID_COL] + LEAKAGE)
    print(f"[1] venue_features: {len(df):,} rows, {len(RAW_FEATURES)} raw features "
          f"({len(CATEGORICAL)} categorical one-hot encoded); dropped {ID_COL} and {LEAKAGE[0]} (leakage)")
    print(f"    outcome high_noise = 1 if >= 3 NYC 311 noise complaints at the address in 3 years; "
          f"rate {df[TARGET].mean():.1%}")

    # 2. split
    train, test = split(df, args.seed)
    prep = Preprocessor().fit(train)
    X_tr, X_te = prep.transform(train), prep.transform(test)
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    split_info = {"n_train": len(train), "n_test": len(test), "n_features_after_one_hot": len(prep.columns),
                  "train_high_noise_count": int(y_tr.sum()), "train_high_noise_rate": round(float(y_tr.mean()), 4),
                  "test_high_noise_count": int(y_te.sum()), "test_high_noise_rate": round(float(y_te.mean()), 4)}
    print(f"\n[2] split 70/30 stratified, seed {args.seed}")
    print(f"    train {len(train):,} rows: high_noise=1 {y_tr.sum():,} ({y_tr.mean():.1%}), =0 {len(y_tr) - y_tr.sum():,}")
    print(f"    test  {len(test):,} rows: high_noise=1 {y_te.sum():,} ({y_te.mean():.1%}), =0 {len(y_te) - y_te.sum():,}")
    print(f"    {len(prep.columns)} model columns after one-hot")

    # 3. baseline
    majority = int(pd.Series(y_tr).mode()[0])
    base_pred = np.full(len(y_te), majority)
    print(f"\n[3] baseline: predict the most common training class ({majority} = no badge) for every test row")

    # 4. v1 tree
    model, leaf, cv = fit_v1(X_tr, y_tr, args.seed)
    FITTED.update(model=model, prep=prep)
    print(f"\n[4] v1: DecisionTreeClassifier(max_depth={MAX_DEPTH}, class_weight='balanced')")
    print(f"    {CV_FOLDS}-fold CV on train rows, mean {CV_SCORING} by min_samples_leaf: {cv}")
    print(f"    chosen min_samples_leaf = {leaf}; refit on all {len(train):,} train rows; "
          f"{model.get_n_leaves()} leaves, depth {model.get_depth()}")
    tree_text = export_text(model, feature_names=prep.columns, show_weights=False, decimals=2)
    print("\n" + tree_text)
    imp = pd.Series(model.feature_importances_, index=prep.columns).sort_values(ascending=False)
    top = imp[imp > 0].head(8)
    print("    top feature importances:")
    for f, v in top.items():
        print(f"      {f:<34} {v:.3f}")
    root_f, root_t = prep.columns[model.tree_.feature[0]], model.tree_.threshold[0]
    left_rate = y_tr[X_tr[root_f].to_numpy() <= root_t].mean()
    right_rate = y_tr[X_tr[root_f].to_numpy() > root_t].mean()
    if "=" in root_f:  # one-hot column: "<= 0.5" means "not this category"
        print(f"    reading the splits: the tree first asks whether {root_f.replace('=', ' is ')}; "
              f"in training those venues are high_noise {right_rate:.0%} of the time, the rest {left_rate:.0%}.")
    else:
        print(f"    reading the splits: the tree first asks whether {root_f} <= {root_t:.2f}; "
              f"in training those venues are high_noise {left_rate:.0%} of the time, the rest {right_rate:.0%}.")
    # Fixed street-grid key for the coordinate splits (geography, true for any seed).
    print("    coordinate key: lat 40.73-40.74 is about 14th Street, 40.80 about 110th Street, "
          "40.83 about 145th Street; lon -74.00 is about Hudson Street / 7th Ave South, "
          "-73.98 about Fifth Avenue in Midtown.")
    print(f"    the top three features ({', '.join(top.index[:3])}) carry "
          f"{top.iloc[:3].sum():.0%} of the tree's importance, so the badge mostly turns on those.")

    # 5. evaluation
    base_s, v1_s = scores(y_te, base_pred), scores(y_te, model.predict(X_te))
    train_acc = round(float(accuracy_score(y_tr, model.predict(X_tr))), 4)
    base_train_acc = round(float((y_tr == majority).mean()), 4)
    print("\n[5] test rows: baseline vs v1 (precision and recall are for the high_noise = 1 class)")
    print(f"    {'':<22}{'baseline':>10}{'v1 tree':>10}")
    for m in ["accuracy", "precision", "recall"]:
        print(f"    {m:<22}{base_s[m]:>10.3f}{v1_s[m]:>10.3f}")
    for k, lab in [("tn", "TN quiet, no badge"), ("fp", "FP quiet, badge"), ("fn", "FN loud, no badge"),
                   ("tp", "TP loud, badge")]:
        print(f"    {lab:<22}{base_s[k]:>10,}{v1_s[k]:>10,}")
    print(f"    v1 accuracy: train {train_acc:.3f}, test {v1_s['accuracy']:.3f} "
          f"(gap {train_acc - v1_s['accuracy']:+.3f}); baseline train {base_train_acc:.3f}")

    # Reference only (not v1): the same depth-4 tree WITHOUT class weights, min_samples_leaf tuned
    # for plain accuracy. It shows why v1 is class-weighted: it hugs the baseline and flags few venues.
    ref = GridSearchCV(DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=args.seed),
                       {"min_samples_leaf": LEAF_GRID},
                       cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=args.seed),
                       scoring="accuracy").fit(X_tr, y_tr)
    ref_s = scores(y_te, ref.best_estimator_.predict(X_te))
    ref_leaf = int(ref.best_params_["min_samples_leaf"])
    print(f"    reference, not v1: unweighted depth-{MAX_DEPTH} tree (min_samples_leaf {ref_leaf}, tuned on accuracy): "
          f"test accuracy {ref_s['accuracy']:.3f}, precision {ref_s['precision']:.3f}, recall {ref_s['recall']:.3f}, "
          f"flags {ref_s['tp'] + ref_s['fp']} of {len(y_te):,} test venues")

    # 6. decision for a new venue
    venue = {k: v for k, v in json.loads(args.new_input.read_text()).items() if not k.startswith("_")}
    decision = decide(venue)
    print(f"\n[6] decide() on {args.new_input.name}: {venue['venue_type']}, {venue['neighborhood']}, "
          f"{venue['liquor_tier']}, closes late={venue['osm_closes_late']}, years licensed={venue['years_licensed']}")
    print(f"    -> {decision}")

    # 7. value
    base_v, v1_v = monthly_value(base_s, len(y_te)), monthly_value(v1_s, len(y_te))
    ref_v = monthly_value(ref_s, len(y_te))
    diff = round(v1_v["usd_per_month"] - base_v["usd_per_month"], 2)
    print(f"\n[7] value per month, launch area, chill-group decisions only "
          f"({base_v['decisions_per_month']:,.0f} decisions/month; all inputs are ASSUMPTIONS)")
    for k, v in ASSUMPTIONS.items():
        print(f"      ASSUMPTION {k:<26} {v}")
    print(f"    baseline ${base_v['usd_per_month']:,.2f}/month (${base_v['usd_per_decision']:.3f}/decision)")
    print(f"    v1 tree  ${v1_v['usd_per_month']:,.2f}/month (${v1_v['usd_per_decision']:.3f}/decision)")
    print(f"    v1 - baseline = ${diff:+,.2f}/month")
    print(f"    (reference unweighted tree: ${ref_v['usd_per_month']:,.2f}/month)")

    # 8. outputs
    metrics = {
        "seed": args.seed, "data": {"db": "data/manhattan_bars.db", "table": "venue_features", "n_rows": len(df),
                                    "outcome": "high_noise = 1 if noise_complaints >= 3 (NYC 311, 3 years)",
                                    "dropped": [ID_COL] + LEAKAGE, "high_noise_rate": round(float(df[TARGET].mean()), 4)},
        "split": split_info,
        "baseline": {"rule": f"predict {majority} (no badge) for every venue", "train_accuracy": base_train_acc,
                     "test": base_s},
        "v1": {"model": "DecisionTreeClassifier", "max_depth": MAX_DEPTH, "class_weight": "balanced",
               "min_samples_leaf": leaf, "cv_folds": CV_FOLDS, "cv_scoring": CV_SCORING,
               "cv_mean_score_by_min_samples_leaf": cv, "n_leaves": int(model.get_n_leaves()),
               "depth": int(model.get_depth()), "train_accuracy": train_acc, "test": v1_s,
               "train_minus_test_accuracy": round(train_acc - v1_s["accuracy"], 4),
               "feature_importance": {f: round(float(v), 4) for f, v in imp[imp > 0].items()},
               "root_split": {"feature": root_f, "threshold": round(float(root_t), 4),
                              "train_high_noise_rate_le": round(float(left_rate), 4),
                              "train_high_noise_rate_gt": round(float(right_rate), 4)}},
        "reference_unweighted_tree": {"note": "not v1; same depth, no class weights, leaf tuned on accuracy",
                                      "min_samples_leaf": ref_leaf, "test": ref_s, "value": ref_v},
        "decision_example": {"input_file": args.new_input.name, "input": venue, "output": decision},
        "value": {"assumptions": ASSUMPTIONS, "rank_penalty": RANK_PENALTY,
                  "baseline": base_v, "v1": v1_v, "v1_minus_baseline_usd_per_month": diff},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "tree.txt").write_text(tree_text)

    cm_rows = [[k.upper(), lab, base_s[k], v1_s[k]] for k, lab in
               [("tn", "actual 0, predicted 0"), ("fp", "actual 0, predicted 1"),
                ("fn", "actual 1, predicted 0"), ("tp", "actual 1, predicted 1")]]
    report = "\n\n".join([
        "# Venue noise tree: results (generated by venue_noise_tree.py; do not edit by hand)",
        f"Seed {args.seed}. Data: `venue_features` ({len(df):,} real venues). Outcome: high_noise = 1 if "
        f">= 3 NYC 311 noise complaints at the address in 3 years.",
        "## Split\n\n" + table([["train", f"{len(train):,}", f"{y_tr.sum():,}", f"{y_tr.mean():.1%}"],
                                 ["test", f"{len(test):,}", f"{y_te.sum():,}", f"{y_te.mean():.1%}"]],
                                ["part", "rows", "high_noise = 1", "rate"]),
        "## Baseline vs v1 on the test rows\n\nPrecision and recall are for the high_noise = 1 class. "
        "The baseline never predicts 1, so its precision is 0/0, shown as 0.\n\n"
        + table([[m, f"{base_s[m]:.3f}", f"{v1_s[m]:.3f}"] for m in ["accuracy", "precision", "recall"]]
                + [["train accuracy", f"{base_train_acc:.3f}", f"{train_acc:.3f}"]],
                ["metric", "baseline", "v1 tree"]),
        f"Reference, not v1: the same depth-{MAX_DEPTH} tree without class weights (min_samples_leaf "
        f"{ref_leaf}, tuned on accuracy) scores accuracy {ref_s['accuracy']:.3f}, precision "
        f"{ref_s['precision']:.3f}, recall {ref_s['recall']:.3f} and is worth "
        f"${ref_v['usd_per_month']:,.2f}/month under the same assumptions.",
        "## Confusion matrices (test rows)\n\n" + table(cm_rows, ["cell", "meaning", "baseline", "v1 tree"]),
        f"## v1 settings\n\nmax_depth {MAX_DEPTH}, class_weight balanced, min_samples_leaf {leaf} "
        f"(chosen by {CV_FOLDS}-fold CV on train rows, {CV_SCORING}), {model.get_n_leaves()} leaves.\n\n"
        + table([[k, v] for k, v in cv.items()], ["min_samples_leaf", f"CV mean {CV_SCORING}"])
        + "\n\nTop feature importances:\n\n"
        + table([[f, f"{v:.3f}"] for f, v in top.items()], ["feature", "importance"])
        + f"\n\nRoot split: `{root_f} <= {root_t:.2f}` (train high_noise rate {left_rate:.1%} if yes, "
        f"{right_rate:.1%} if no). Full tree: `tree.txt`, `tree.png`.",
        "## Decision for the example new venue\n\nInput: `" + args.new_input.name + "`\n\n"
        + table([[k, v] for k, v in decision.items()], ["field", "value"]),
        "## Value per month, chill-group decisions in the launch area\n\nAll inputs are made-up ASSUMPTIONS. "
        "Rates come from the test confusion matrices, scaled to "
        f"{base_v['decisions_per_month']:,.0f} decisions/month.\n\n"
        + table([[k, v] for k, v in ASSUMPTIONS.items()] + [["rank_penalty", RANK_PENALTY]],
                ["ASSUMPTION", "value"])
        + "\n\n" + table([[k.upper(), f"{v:.2f}"] for k, v in base_v["usd_per_decision_by_cell"].items()],
                         ["cell", "USD per decision"])
        + "\n\n" + table([["baseline", f"{base_v['usd_per_decision']:.3f}", f"{base_v['usd_per_month']:,.2f}"],
                          ["v1 tree", f"{v1_v['usd_per_decision']:.3f}", f"{v1_v['usd_per_month']:,.2f}"],
                          ["v1 - baseline", f"{v1_v['usd_per_decision'] - base_v['usd_per_decision']:+.3f}",
                           f"{diff:+,.2f}"]], ["rule", "USD per decision", "USD per month"]),
        "## Versions\n\n" + ", ".join(f"{k} {v}" for k, v in metrics["versions"].items()),
    ]) + "\n"
    (args.out / "report.md").write_text(report)

    chart_confusion(base_s, v1_s, args.out / "confusion_matrices.png")
    chart_bars(base_s, v1_s, args.out / "baseline_vs_v1.png")
    chart_tree(model, prep.columns, args.out / "tree.png")
    chart_importance(imp, args.out / "feature_importance.png")
    print(f"\n[8] wrote metrics.json, report.md, tree.txt and 4 PNG charts to {args.out}")
    return metrics


if __name__ == "__main__":
    sys.exit(main() and 0)
