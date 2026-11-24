"""Group model assignment, v1: multiple linear regression for a venue's 311 noise complaints.

Question the model serves (Nightlife Autopilot product/ops lead): when a group asks where to go
tonight and we have no fresh Scout reports for a venue, should the app label it complaint-prone
(loud/rowdy) and demote it for groups who asked for a chill night?

Decision rule: predict the venue's noise complaints; if the prediction is >= THRESHOLD, show a
"complaint_risk" badge and multiply its rank score by RANK_PENALTY for chill-preference groups.
Baseline: predict the TRAINING mean for every venue (so every venue gets the same decision).

Data: REAL public data, table `venue_features` in data/manhattan_bars.db (one row per licensed
Manhattan on-premises venue; built by src/build_db.py from NY SLA licences, DOHMH inspections,
OpenStreetMap and NYC 311). Outcome `noise_complaints` = number of NYC 311 noise complaints
(location type Club/Bar/Restaurant, last 3 years) matched to the venue's address.

Steps
  1. load venue_features read-only; drop license_id and high_noise (leakage); build 11 terms
  2. random 70/30 train/test split with --seed (not stratified)
  3. baseline = training mean of noise_complaints, predicted for every test row
  4. v1 = statsmodels OLS on log1p(noise_complaints); print the summary and the coefficients
  5. baseline vs v1 on the test rows, in complaints: R^2, RMSE, MAE; v1 train vs test R^2
  6. decide(venue) on a realistic new venue (example_new_venue.json or --new-input)
  7. dollars per month, baseline vs v1, classifying test venues as truly loud (actual >= THRESHOLD)
  8. write outputs/metrics.json, outputs/report.md and four PNG charts

Run (from the repo root):
  python coursework/group_model/venue_noise_regression.py
  python coursework/group_model/venue_noise_regression.py --seed 7 --new-input my_venue.json
"""
import argparse
import json
import platform
import sqlite3
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats  # noqa: E402
import statsmodels  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = ROOT / "data" / "manhattan_bars.db"
OUT = HERE / "outputs"
EXAMPLE = HERE / "example_new_venue.json"

# ---------------------------------------------------------------------------------------------
# Decision threshold and business assumptions. Every figure here is MADE UP for the assignment
# (labelled ASSUMPTION); none comes from the data. Change them and re-run.
# ---------------------------------------------------------------------------------------------
# ASSUMPTION: "complaint-prone" = predicted complaints >= the TRAINING 75th percentile of
# noise_complaints. With seed 42 that is 1 complaint in 3 years (two thirds of venues have 0),
# so the badge means "expected to draw at least one complaint". It is computed in main().
THRESHOLD_QUANTILE = 0.75
RANK_PENALTY = 0.70  # ASSUMPTION: rank_score multiplier for badged venues, chill groups only
                     # (same multiplicative style as recs/scoring.py energy_penalty)

# A "decision" = one group that asked for a chill night being sent to its top venue, where that
# venue has no fresh Scout reports (the only case where the rule can change anything).
ASSUMPTIONS = {
    "groups_per_month": 6000,          # ASSUMPTION: group requests per month in the launch area (LES)
    "share_chill_groups": 0.35,        # ASSUMPTION: share of groups that ask for a chill night
    "share_no_fresh_scout": 0.50,      # ASSUMPTION: share of top picks with no fresh Scout report
    "value_good_match_usd": 4.00,      # ASSUMPTION: referral fee + retention value of a night that fits
    "p_bad_night_if_loud": 0.40,       # ASSUMPTION: chance a truly loud venue ruins a chill group's night
    "cost_bad_night_usd": 25.00,       # ASSUMPTION: group walks out and drops the app for the night
    "cost_wrong_demotion_usd": 1.50,   # ASSUMPTION: a quiet partner venue wrongly loses the exposure
}
# Derived: expected cost of sending a chill group to a truly loud venue = 0.40 * $25 = $10.

# ---------------------------------------------------------------------------------------------
# Features. 11 readable terms chosen from the training-data profile; rare and similar categories
# are collapsed so each coefficient is one plain statement. The base case (all terms 0) is a
# beer/wine Restaurant, not tagged as a bar in OpenStreetMap, in "rest of Manhattan".
# Left out on purpose:
#   high_noise        derived from the outcome (noise_complaints >= 3): leakage
#   license_id        an ID
#   lat, lon          a straight-line effect of coordinates means nothing here; the
#                     neighbourhood groups carry location instead
#   inspection counts, scores, years_licensed, density_250m, website, outdoor seating:
#                     near-zero correlation with complaints, or p > 0.4 in a first fit
# ---------------------------------------------------------------------------------------------
TARGET = "noise_complaints"
DROP = ["license_id", "high_noise"]
UPPER_MANHATTAN = ["Harlem", "East Harlem", "Washington Heights/Inwood"]
MIDTOWN = ["Midtown West", "Midtown East/Murray Hill"]
OTHER_VENUE_TYPES = ["Hotel", "Club", "Venue/Theatre", "Nightlife"]  # 444 rows together ("Other")
OSM_NIGHTLIFE = ["bar", "pub", "nightclub"]                          # nightclub alone is 15 rows

# term -> (plain label, what it is compared with, business reading)
TERMS = {
    "bar_tavern": ("a Bar/Tavern licence", "a Restaurant",
                   "Bars draw more complaints than restaurants with the same other traits."),
    "other_venue_type": ("a Hotel, Club, Theatre or Nightlife licence", "a Restaurant",
                         "Hotel bars and venues are usually inside larger buildings, so fewer complaints reach 311."),
    "full_liquor": ("a full liquor licence", "beer and wine only",
                    "Spirits and later crowds: the licence tier alone is one of the strongest signals."),
    "osm_bar_pub_club": ("tagged bar, pub or nightclub in OpenStreetMap", "no such tag",
                         "Places the public map lists as nightlife behave like nightlife, whatever the licence says."),
    "osm_closes_late": ("posted hours past 1 am", "no late hours on record",
                        "Late closing puts the crowd on the street when neighbours are asleep."),
    "hood_les_chinatown": ("in LES/Chinatown", "rest of Manhattan",
                           "The launch area itself is a complaint hotspot, so the badge will fire often there."),
    "hood_east_village": ("in the East Village", "rest of Manhattan",
                          "Dense bars under apartments: similar to the LES."),
    "hood_upper_manhattan": ("in Harlem, East Harlem or Washington Heights/Inwood", "rest of Manhattan",
                             "More residents per venue means more people who call 311."),
    "hood_midtown": ("in Midtown West or Midtown East/Murray Hill", "rest of Manhattan",
                     "Office districts empty out at night, so fewer neighbours complain."),
    "density_100m": ("one more licensed venue within 100 m", "one fewer",
                     "Clusters of venues add complaints, but the effect per extra venue is small."),
    "grade_b_or_c": ("a latest DOHMH grade of B or C", "A, pending or no grade",
                     "Venues with weaker inspection grades also draw more noise complaints, a proxy for how well they are run."),
}
TERM_NAMES = list(TERMS)
RAW_FEATURES = ["venue_type", "liquor_tier", "osm_amenity", "osm_closes_late", "neighborhood",
                "density_100m", "latest_grade"]


def load_venues(db: Path = DB) -> pd.DataFrame:
    """Read venue_features read-only (mode=ro), so the database can never be changed by mistake."""
    if not db.exists():
        raise FileNotFoundError(f"database not found: {db}")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return pd.read_sql_query("SELECT * FROM venue_features", conn)


def design_matrix(df: pd.DataFrame, density_median: float) -> pd.DataFrame:
    """Raw venue columns -> the 11 model terms plus a constant. Same code for training and decide().

    NaN handling (the committed table has none; a new venue's dict may leave fields out):
    a missing category falls into the base case (term = 0); a missing density_100m or
    osm_closes_late gets the training median / 0.
    """
    d = df.reindex(columns=RAW_FEATURES)
    X = pd.DataFrame(index=d.index)
    X["bar_tavern"] = (d["venue_type"] == "Bar/Tavern").astype(int)
    X["other_venue_type"] = d["venue_type"].isin(OTHER_VENUE_TYPES).astype(int)
    X["full_liquor"] = (d["liquor_tier"] == "full_liquor").astype(int)
    X["osm_bar_pub_club"] = d["osm_amenity"].isin(OSM_NIGHTLIFE).astype(int)
    X["osm_closes_late"] = pd.to_numeric(d["osm_closes_late"], errors="coerce").fillna(0).astype(int)
    X["hood_les_chinatown"] = (d["neighborhood"] == "LES/Chinatown").astype(int)
    X["hood_east_village"] = (d["neighborhood"] == "East Village").astype(int)
    X["hood_upper_manhattan"] = d["neighborhood"].isin(UPPER_MANHATTAN).astype(int)
    X["hood_midtown"] = d["neighborhood"].isin(MIDTOWN).astype(int)
    X["density_100m"] = pd.to_numeric(d["density_100m"], errors="coerce").fillna(density_median)
    X["grade_b_or_c"] = d["latest_grade"].isin(["B", "C"]).astype(int)
    return sm.add_constant(X[TERM_NAMES], has_constant="add")


def split(df: pd.DataFrame, seed: int):
    """Random 70/30 split (not stratified: the outcome is a count)."""
    return train_test_split(df, test_size=0.30, random_state=seed)


def errors(y_true, y_pred) -> dict:
    """R^2, RMSE and MAE in complaint units."""
    return {"r2": round(float(r2_score(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4)}


def to_complaints(log_pred) -> np.ndarray:
    """Back from log1p units. expm1 of a mean-of-logs prediction estimates a TYPICAL venue (close
    to the median), not the mean, so it sits below the outliers; floored at 0 complaints."""
    return np.clip(np.expm1(np.asarray(log_pred, dtype=float)), 0, None)


def flag_counts(actual_loud: np.ndarray, flagged: np.ndarray) -> dict:
    """Confusion counts for the badge: truly loud = actual >= threshold, flagged = predicted >= threshold."""
    return {"tn": int((~actual_loud & ~flagged).sum()), "fp": int((~actual_loud & flagged).sum()),
            "fn": int((actual_loud & ~flagged).sum()), "tp": int((actual_loud & flagged).sum())}


# Filled in by main() after fitting; decide() uses it when nothing is passed explicitly.
FITTED: dict = {}


def decide(venue: dict, model=None, density_median: float = None, threshold: float = None) -> dict:
    """The decision rule for ONE new venue the product sees (raw venue_features columns).

    Returns {"predicted_complaints": expected 311 noise complaints over 3 years (typical venue
    with these traits), "badge": "complaint_risk" | "none", "rank_penalty": multiplier on
    rank_score for chill-preference groups (1.0 = no change)}.
    """
    model = model if model is not None else FITTED["model"]
    density_median = density_median if density_median is not None else FITTED["density_median"]
    threshold = threshold if threshold is not None else FITTED["threshold"]
    X = design_matrix(pd.DataFrame([{k: venue.get(k) for k in RAW_FEATURES}]), density_median)
    pred = float(to_complaints(model.predict(X))[0])
    flagged = pred >= threshold
    return {"predicted_complaints": round(pred, 3),
            "badge": "complaint_risk" if flagged else "none",
            "rank_penalty": RANK_PENALTY if flagged else 1.0,
            "rank_penalty_applies_to": "chill_preference_groups",
            "threshold": threshold}


def monthly_value(counts: dict, n_test: int, uniform: bool, a: dict = ASSUMPTIONS) -> dict:
    """Expected dollars per month from chill-group decisions, using test-row rates.

    Per decision, by cell (truly loud = actual >= threshold; flagged = predicted >= threshold):
      quiet, not flagged (TN) : +value_good_match
      loud,  not flagged (FN) : -(p_bad_night_if_loud * cost_bad_night)    chill group sent to a loud venue
      loud,  flagged     (TP) : +value_good_match                          steered to an unflagged venue
      quiet, flagged     (FP) : +value_good_match - cost_wrong_demotion    alternative is fine, venue loses out
    uniform=True is the baseline: when EVERY venue gets the same badge and penalty, the ranking is
    unchanged, so groups go where they would with no badge at all; loud venues then count as FN
    and quiet ones as TN, whatever the badge says. Groups that did not ask for a chill night are
    never demoted, so they add the same amount under both rules and are left out.
    """
    v, cost_loud, fpc = (a["value_good_match_usd"], a["p_bad_night_if_loud"] * a["cost_bad_night_usd"],
                         a["cost_wrong_demotion_usd"])
    per_cell = {"tn": v, "fn": -cost_loud, "tp": v, "fp": v - fpc}
    if uniform:
        loud, quiet = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
        counts = {"tn": quiet, "fp": 0, "fn": loud, "tp": 0}
    decisions = a["groups_per_month"] * a["share_chill_groups"] * a["share_no_fresh_scout"]
    per_decision = sum(counts[k] / n_test * per_cell[k] for k in per_cell)
    return {"decisions_per_month": round(decisions, 1), "effective_counts": counts,
            "usd_per_decision_by_cell": {k: round(x, 4) for k, x in per_cell.items()},
            "usd_per_decision": round(per_decision, 4), "usd_per_month": round(per_decision * decisions, 2)}


# ---------------------------------------------------------------------------------------------
# Charts: plain matplotlib, black on white, labelled axes, no headline titles.
# ---------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "black", "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150})
LOG_TICKS = [0, 1, 3, 10, 30, 100, 300]


def _log1p_axis(ax, which: str) -> None:
    """Axis in log1p units, labelled in complaints (0, 1, 3, 10, ...)."""
    ticks, labels = np.log1p(LOG_TICKS), [str(t) for t in LOG_TICKS]
    (ax.set_xticks if which == "x" else ax.set_yticks)(ticks, labels)


def chart_pred_vs_actual(y_te, pred, base_value, threshold, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6))
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(y_te))  # spreads the many zeros
    ax.scatter(np.log1p(pred), np.log1p(y_te) + jitter, s=9, alpha=0.35, color="black", label="v1, one test venue")
    top = np.log1p(max(y_te.max(), 300))
    ax.plot([0, top], [0, top], color="black", lw=1, label="perfect prediction")
    ax.axvline(np.log1p(base_value), color="black", ls="--", lw=1.2,
               label=f"baseline: training mean {base_value:.2f}")
    ax.axvline(np.log1p(threshold), color="#888888", ls=":", lw=1.5, label=f"badge threshold {threshold:g}")
    _log1p_axis(ax, "x")
    _log1p_axis(ax, "y")
    ax.set_xlim(-0.05, np.log1p(10))
    ax.set_xlabel("predicted noise complaints (log scale)")
    ax.set_ylabel("actual noise complaints, test rows (log scale, jittered)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_errors(base: dict, v1: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, (m, lab) in zip(axes, [("r2", "test R²"), ("rmse", "test RMSE (complaints)"),
                                   ("mae", "test MAE (complaints)")]):
        bars = ax.bar(["baseline", "v1"], [base[m], v1[m]], color=["#bdbdbd", "#404040"], edgecolor="black")
        ax.bar_label(bars, fmt="%.3f" if m == "r2" else "%.2f", padding=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = min(0.0, base[m], v1[m]), max(0.0, base[m], v1[m])
        pad = max(0.15 * (hi - lo), 0.02 if m == "r2" else 0.5)  # R² near 0: keep room for labels
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_coefficients(model, path: Path) -> None:
    b, ci = model.params[TERM_NAMES], model.conf_int().loc[TERM_NAMES]
    pct, lo, hi = np.expm1(b) * 100, np.expm1(ci[0]) * 100, np.expm1(ci[1]) * 100
    order = pct.sort_values().index
    labels = [t + (" (per venue)" if t == "density_100m" else "") for t in order]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(pct[order], range(len(order)), xerr=[pct[order] - lo[order], hi[order] - pct[order]],
                fmt="o", color="black", ecolor="black", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(order)), labels)
    ax.set_xlabel("% change in (complaints + 1), with 95% interval")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_residuals(model, path: Path) -> None:
    fitted, resid = model.fittedvalues, model.resid
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(fitted, resid, s=7, alpha=0.3, color="black")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("fitted log(1 + complaints), training rows")
    axes[0].set_ylabel("residual, log(1 + complaints)")
    axes[1].hist(resid, bins=60, color="#808080", edgecolor="black")
    axes[1].set_xlabel("residual, log(1 + complaints)")
    axes[1].set_ylabel("training venues")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(rows: list, header: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="seed for the train/test split (default 42)")
    ap.add_argument("--new-input", type=Path, default=EXAMPLE,
                    help="JSON file with one new venue's raw features (default: example_new_venue.json)")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: coursework/group_model/outputs)")
    args = ap.parse_args(argv)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. load
    df = load_venues().drop(columns=DROP)
    y_all = df[TARGET]
    print(f"[1] venue_features: {len(df):,} rows; outcome noise_complaints (NYC 311, 3 years, matched to the address)")
    print(f"    dropped {DROP} (high_noise = noise_complaints >= 3 is leakage); {len(TERM_NAMES)} model terms")
    print(f"    outcome: mean {y_all.mean():.2f}, median {y_all.median():.0f}, max {y_all.max()}, "
          f"zero for {(y_all == 0).mean():.1%} of venues")

    # 2. split
    train, test = split(df, args.seed)
    y_tr, y_te = train[TARGET].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)
    density_median = float(train["density_100m"].median())
    X_tr, X_te = design_matrix(train, density_median), design_matrix(test, density_median)
    print(f"\n[2] split 70/30 random (not stratified), seed {args.seed}")
    print(f"    train {len(train):,} rows: noise_complaints mean {y_tr.mean():.2f}, median {np.median(y_tr):.0f}")
    print(f"    test  {len(test):,} rows: noise_complaints mean {y_te.mean():.2f}, median {np.median(y_te):.0f}")

    # 3. baseline
    base_value = float(y_tr.mean())
    threshold = float(np.quantile(y_tr, THRESHOLD_QUANTILE))
    base_flags = base_value >= threshold
    print(f"\n[3] baseline: predict the training mean, {base_value:.3f} complaints, for every test row")
    print(f"    badge threshold (ASSUMPTION: training {THRESHOLD_QUANTILE * 100:.0f}th percentile) = {threshold:g} complaint(s);"
          f" the mean is {'above' if base_flags else 'below'} it, so the baseline gives EVERY venue "
          f"{'the complaint_risk badge' if base_flags else 'no badge'}")

    # 4. v1: choose the outcome scale on the TRAINING rows, then fit
    raw_fit = sm.OLS(train[TARGET].astype(float), X_tr).fit()
    log_fit = sm.OLS(np.log1p(train[TARGET]).rename("log1p_noise_complaints"), X_tr).fit()
    raw_skew, log_skew = float(scipy.stats.skew(raw_fit.resid)), float(scipy.stats.skew(log_fit.resid))
    # log1p is used: on the training rows it lifts adjusted R^2 about 4x and makes the residuals
    # close to symmetric (skew ~18 -> ~2), as OLS assumes. Errors are still reported in complaints.
    print(f"\n[4] outcome scale, training rows: raw OLS adj R² {raw_fit.rsquared_adj:.3f} (residual skew "
          f"{raw_skew:.1f}) vs log1p OLS adj R² {log_fit.rsquared_adj:.3f} (residual skew {log_skew:.1f}) "
          f"-> v1 uses log1p")
    model = log_fit
    FITTED.update(model=model, density_median=density_median, threshold=threshold)
    print(model.summary(title="v1: OLS on log(1 + noise_complaints), training rows"))

    coefs = []
    print("\n    reading the coefficients (each holding the other terms fixed; % change in complaints + 1):")
    for t in model.pvalues[TERM_NAMES].sort_values().index:
        b, p = float(model.params[t]), float(model.pvalues[t])
        pct = float(np.expm1(b))
        label, ref, meaning = TERMS[t]
        coefs.append({"term": t, "coef": round(b, 4), "pct_change": round(pct, 4), "p_value": round(p, 4),
                      "ci95": [round(float(x), 4) for x in model.conf_int().loc[t]], "meaning": meaning})
        sig = "" if p < 0.05 else " Not significant at 5%, so treat it as no clear effect."
        print(f"    - {t}: {label} vs {ref}: {pct:+.{1 if abs(pct) < 0.05 else 0}%} expected complaints "
              f"(p = {p:.3g}). {meaning}{sig}")

    # 5. evaluation in complaint units
    pred_tr, pred_te = to_complaints(model.predict(X_tr)), to_complaints(model.predict(X_te))
    base_s, v1_s = errors(y_te, np.full_like(y_te, base_value)), errors(y_te, pred_te)
    v1_train = errors(y_tr, pred_tr)
    print("\n[5] test rows, in complaints: baseline vs v1")
    print(f"    {'':<14}{'baseline':>10}{'v1':>10}")
    for m, lab in [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")]:
        print(f"    {lab:<14}{base_s[m]:>10.3f}{v1_s[m]:>10.3f}")
    print(f"    v1 adjusted R² (training fit, log scale): {model.rsquared_adj:.3f}  (R² {model.rsquared:.3f})")
    print(f"    v1 R² in complaints: train {v1_train['r2']:.3f}, test {v1_s['r2']:.3f}")
    test_log_r2 = float(r2_score(np.log1p(y_te), model.predict(X_te)))
    print(f"    v1 R² on the log scale it was fitted on: train {model.rsquared:.3f}, test {test_log_r2:.3f}")
    print("    note: in complaints, R² and RMSE are driven by a few venues with 50-500 complaints; v1 predicts a")
    print("    typical venue (median-like), so it cuts MAE but not the squared error from those outliers")

    actual_loud = y_te >= threshold
    base_counts = flag_counts(actual_loud, np.full(len(y_te), base_flags))
    v1_counts = flag_counts(actual_loud, pred_te >= threshold)
    print(f"    badge on test rows (truly loud = actual >= {threshold:g}; {actual_loud.sum():,} of {len(y_te):,}):")
    for k, lab in [("tn", "TN quiet, none"), ("fp", "FP quiet, badge"), ("fn", "FN loud, none"),
                   ("tp", "TP loud, badge")]:
        print(f"      {lab:<18}{base_counts[k]:>8,}{v1_counts[k]:>8,}")

    # 6. decision for a new venue
    venue = {k: v for k, v in json.loads(args.new_input.read_text()).items() if not k.startswith("_")}
    decision = decide(venue)
    print(f"\n[6] decide() on {args.new_input.name}: {venue.get('venue_type')}, {venue.get('neighborhood')}, "
          f"{venue.get('liquor_tier')}, closes late={venue.get('osm_closes_late')}, "
          f"years licensed={venue.get('years_licensed')}")
    print(f"    -> {decision}")

    # 7. value
    base_v = monthly_value(base_counts, len(y_te), uniform=True)
    v1_v = monthly_value(v1_counts, len(y_te), uniform=False)
    diff = round(v1_v["usd_per_month"] - base_v["usd_per_month"], 2)
    print(f"\n[7] value per month, launch area, chill-group decisions only "
          f"({base_v['decisions_per_month']:,.0f} decisions/month; all inputs are ASSUMPTIONS)")
    for k, v in ASSUMPTIONS.items():
        print(f"      ASSUMPTION {k:<26} {v}")
    print(f"      ASSUMPTION {'rank_penalty':<26} {RANK_PENALTY}")
    print(f"      ASSUMPTION {'threshold':<26} {threshold:g} (training {THRESHOLD_QUANTILE * 100:.0f}th percentile)")
    print("    the baseline's badge is on every venue, so the ranking is unchanged and it is valued as no badge")
    print(f"    baseline ${base_v['usd_per_month']:,.2f}/month (${base_v['usd_per_decision']:.3f}/decision)")
    print(f"    v1       ${v1_v['usd_per_month']:,.2f}/month (${v1_v['usd_per_decision']:.3f}/decision)")
    print(f"    v1 - baseline = ${diff:+,.2f}/month")

    # 8. outputs
    metrics = {
        "seed": args.seed,
        "data": {"db": "data/manhattan_bars.db", "table": "venue_features", "n_rows": len(df),
                 "outcome": "noise_complaints (NYC 311, 3 years, matched to the address)", "dropped": DROP,
                 "outcome_mean": round(float(y_all.mean()), 4), "outcome_median": float(y_all.median()),
                 "outcome_share_zero": round(float((y_all == 0).mean()), 4)},
        "split": {"n_train": len(train), "n_test": len(test),
                  "train_mean": round(float(y_tr.mean()), 4), "train_median": float(np.median(y_tr)),
                  "test_mean": round(float(y_te.mean()), 4), "test_median": float(np.median(y_te))},
        "scale_choice": {"raw_adj_r2_train": round(float(raw_fit.rsquared_adj), 4),
                         "log1p_adj_r2_train": round(float(log_fit.rsquared_adj), 4),
                         "raw_resid_skew": round(raw_skew, 2), "log1p_resid_skew": round(log_skew, 2),
                         "chosen": "log1p"},
        "baseline": {"rule": "predict the training mean for every venue", "prediction": round(base_value, 4),
                     "decision_for_every_venue": "complaint_risk" if base_flags else "none",
                     "test": base_s, "badge_counts": base_counts},
        "v1": {"model": "statsmodels OLS on log1p(noise_complaints)", "terms": TERM_NAMES,
               "adj_r2_train_log": round(float(model.rsquared_adj), 4), "r2_train_log": round(float(model.rsquared), 4),
               "r2_test_log": round(test_log_r2, 4), "n_train": int(model.nobs),
               "train": v1_train, "test": v1_s, "coefficients": coefs,
               "intercept": round(float(model.params["const"]), 4), "badge_counts": v1_counts},
        "threshold": {"value": threshold, "rule": f"training {THRESHOLD_QUANTILE * 100:.0f}th percentile of noise_complaints",
                      "test_share_truly_loud": round(float(actual_loud.mean()), 4)},
        "decision_example": {"input_file": args.new_input.name, "input": venue, "output": decision},
        "value": {"assumptions": ASSUMPTIONS, "rank_penalty": RANK_PENALTY, "baseline": base_v, "v1": v1_v,
                  "v1_minus_baseline_usd_per_month": diff},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "statsmodels": statsmodels.__version__, "scikit-learn": __import__("sklearn").__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "ols_summary.txt").write_text(str(model.summary()) + "\n")

    report = "\n\n".join([
        "# Venue noise regression: results (generated by venue_noise_regression.py; do not edit by hand)",
        f"Seed {args.seed}. Data: `venue_features` ({len(df):,} real venues). Outcome: noise_complaints "
        "(NYC 311, 3 years). v1 = OLS on log(1 + complaints); test errors are in complaints.",
        "## Split\n\n" + table([["train", f"{len(train):,}", f"{y_tr.mean():.2f}", f"{np.median(y_tr):g}"],
                                 ["test", f"{len(test):,}", f"{y_te.mean():.2f}", f"{np.median(y_te):g}"]],
                                ["part", "rows", "mean complaints", "median complaints"]),
        "## Fit\n\n" + table([["adjusted R², training fit (log scale)", f"{model.rsquared_adj:.3f}"],
                               ["R² train / test, log scale", f"{model.rsquared:.3f} / {test_log_r2:.3f}"],
                               ["R² train / test, complaints", f"{v1_train['r2']:.3f} / {v1_s['r2']:.3f}"],
                               ["raw-scale OLS adjusted R² (not used)", f"{raw_fit.rsquared_adj:.3f}"]],
                              ["measure", "v1"]),
        "## Baseline vs v1 on the test rows (complaints)\n\n"
        + table([[lab, f"{base_s[m]:.3f}", f"{v1_s[m]:.3f}"] for m, lab in
                 [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")]], ["metric", "baseline (training mean)", "v1"]),
        "## Coefficients (sorted by p-value)\n\n"
        + table([[c["term"], f"{c['coef']:+.3f}", f"{c['pct_change']:+.{1 if abs(c['pct_change']) < 0.05 else 0}%}", f"{c['p_value']:.3g}"] for c in coefs],
                ["term", "coef (log scale)", "% change in complaints + 1", "p-value"])
        + f"\n\nIntercept {model.params['const']:+.3f}. Full summary: `ols_summary.txt`.",
        f"## Badge on the test rows (threshold {threshold:g} complaint(s))\n\n"
        + table([[k.upper(), lab, base_counts[k], v1_counts[k]] for k, lab in
                 [("tn", "quiet, no badge"), ("fp", "quiet, badge"), ("fn", "loud, no badge"), ("tp", "loud, badge")]],
                ["cell", "meaning", "baseline", "v1"])
        + "\n\nThe baseline badges every venue (training mean above the threshold), so its ranking is "
        "unchanged and it is valued as no badge.",
        "## Decision for the example new venue\n\nInput: `" + args.new_input.name + "`\n\n"
        + table([[k, v] for k, v in decision.items()], ["field", "value"]),
        "## Value per month, chill-group decisions in the launch area\n\nAll inputs are made-up ASSUMPTIONS. "
        f"Rates come from the test rows, scaled to {base_v['decisions_per_month']:,.0f} decisions/month.\n\n"
        + table([[k, v] for k, v in ASSUMPTIONS.items()]
                + [["rank_penalty", RANK_PENALTY], ["threshold", f"{threshold:g}"]], ["ASSUMPTION", "value"])
        + "\n\n" + table([[k.upper(), f"{v:.2f}"] for k, v in v1_v["usd_per_decision_by_cell"].items()],
                         ["cell", "USD per decision"])
        + "\n\n" + table([["baseline", f"{base_v['usd_per_decision']:.3f}", f"{base_v['usd_per_month']:,.2f}"],
                          ["v1", f"{v1_v['usd_per_decision']:.3f}", f"{v1_v['usd_per_month']:,.2f}"],
                          ["v1 - baseline", f"{v1_v['usd_per_decision'] - base_v['usd_per_decision']:+.3f}",
                           f"{diff:+,.2f}"]], ["rule", "USD per decision", "USD per month"]),
        "## Versions\n\n" + ", ".join(f"{k} {v}" for k, v in metrics["versions"].items()),
    ]) + "\n"
    (args.out / "report.md").write_text(report)

    chart_pred_vs_actual(y_te, pred_te, base_value, threshold, args.out / "predicted_vs_actual.png")
    chart_errors(base_s, v1_s, args.out / "baseline_vs_v1_error.png")
    chart_coefficients(model, args.out / "coefficients.png")
    chart_residuals(model, args.out / "residuals.png")
    print(f"\n[8] wrote metrics.json, report.md, ols_summary.txt and 4 PNG charts to {args.out}")
    return metrics


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        main()
