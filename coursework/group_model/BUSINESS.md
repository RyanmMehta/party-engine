# BUSINESS.md: business framing, value worksheet and slide checklist

Reference document for the group model assignment. It collects facts, formulas and sources so the team can write the slides themselves. It contains no slide text: the assignment requires the slides to be written by the team, not generated.

Team name: `<TEAM NAME>` | Members: `<MEMBER 1>, <MEMBER 2>, <MEMBER 3>, ...`
Repo: https://github.com/RyanmMehta/manhattan-bars-ml | Script: `coursework/group_model/venue_noise_regression.py`
Data: see `coursework/group_model/DATA.md`. Model outputs (from the script, seed 42): `coursework/group_model/outputs/metrics.json`, `outputs/report.md`, `outputs/ols_summary.txt`, and the charts `outputs/predicted_vs_actual.png`, `outputs/baseline_vs_v1_error.png`, `outputs/coefficients.png`, `outputs/residuals.png`.

Every number below is copied from `outputs/metrics.json` as produced by the script at commit `aee9f51`. **The script wins:** if it is re-run and a number changes, update this file from `metrics.json`.

## Read this before presenting

- On the 1,712 test rows, v1 lowers the typical error: MAE falls from 5.15 complaints (mean baseline) to 3.45 (3.4546), about a third lower.
- But in complaint units R² is about 0 (baseline -0.000, v1 -0.007) and RMSE is slightly worse (baseline 14.64, v1 14.69).
- Why: a handful of venues with hundreds of complaints (max 567) dominate squared error, and a model fitted on log(1 + complaints) predicts what a typical venue with those traits gets, not the outliers.
- On the log scale, where the model was fitted, it explains about 12% of the variation: adjusted R² 0.116 on training rows, R² 0.118 train vs 0.124 test, so no sign of overfitting.
- The model's business value comes from sorting venues into likely-loud vs likely-quiet at the threshold, not from predicting exact counts.

## 1. Business question

**Who acts on it:** the Nightlife Autopilot product/ops lead.

**Question (one sentence, fixed wording shared with the model script):**
"For a venue with no fresh scout reports, how many noise complaints should we expect, and is that enough to badge it as complaint-prone and demote it for groups who asked for a chill night?"

**What the product does today without the model:** Every venue card, venue profile and recommendation row carries a `complaint_risk` object (`docs/CONTRACTS.md`, backend CR-19: `{label: "311 noise-complaint risk", noise_complaints_3y, level, score}`), but the app shows it only as a small line on the venue profile, the raw 311 count with a low/medium/high level (bands under 25, 25 to 99, 100 or more, marked as a placeholder in `backend/app/config.py`), and never on the recommendation list. By rule it never changes vibe, match score, rank or reasons (`docs/specs/SCORING.md` section 9, `docs/prd/frontend/RULES.md`), so a group that asked for a quiet night can be sent to a complaint-prone venue whenever scout reports are missing; in effect no venue gets a warning badge and none is demoted.

(Related, for context only: `recs/complaint_risk.py` already trains a gradient-boosting classifier for "3 or more complaints", out-of-fold ROC AUC 0.73, and the backend passes it as `complaint_risk.score`, but the frontend does not display or use it. Our v1 model is a separate, explainable linear regression for this assignment.)

## 2. Model, decision rule and baseline

**Model:** multiple linear regression, statsmodels OLS on `log1p(noise_complaints)` (`v1.model`), with 11 terms (`v1.terms`): `bar_tavern`, `other_venue_type`, `full_liquor`, `osm_bar_pub_club`, `osm_closes_late`, `hood_les_chinatown`, `hood_east_village`, `hood_upper_manhattan`, `hood_midtown`, `density_100m`, `grade_b_or_c`. The log scale was chosen over raw counts because raw OLS had adjusted R² 0.030 and residual skew 18.27, against 0.116 and 1.85 on log1p (`scale_choice`). Predictions are converted back with `expm1` (floored at 0), and every test error is in complaints.

**Threshold:** `T = 1` complaint in 3 years (`threshold.value`), set as the training 75th percentile of `noise_complaints` (`threshold.rule`; an ASSUMPTION). Because two thirds of venues have 0 complaints, the badge means "expected to draw at least one complaint". 31.4% of test venues are truly loud at this line (`threshold.test_share_truly_loud`). This is a separate line from `high_noise` (3 or more) in `DATA.md`.

**Decision rule in words:** If the predicted complaints are at or above `T` (1), the venue gets a complaint-risk badge, and for groups that asked for a chill night its rank score is multiplied by 0.7 (`value.rank_penalty`, an ASSUMPTION) so it drops down the list. Below `T`: no badge, no penalty. Groups that did not ask for a chill night are never demoted.

**Pseudocode:**

```
T = 1.0                                          # training 75th percentile (ASSUMPTION)
RANK_PENALTY = 0.7                               # ASSUMPTION
for each venue v with no fresh scout report:
    yhat = max(0, expm1(ols.predict(terms(v))))  # expected complaints in 3 years
    if yhat >= T:
        v.badge = "complaint_risk"
        if group.wants_chill:
            v.rank_score = v.rank_score * RANK_PENALTY
    else:
        v.badge = none
re-rank candidates by rank_score
```

**Result for a new input** (`decision_example`, input file `example_new_venue.json`): a Bar/Tavern with a full liquor license in LES/Chinatown, tagged `bar` in OSM, closes late, 27 other venues within 100 m, no DOHMH match. Output: `predicted_complaints` 4.643, `badge` = `complaint_risk`, `rank_penalty` 0.7, applied to chill-preference groups only, `threshold` 1.0.

**Baseline in words:** predict the training mean, 3.2833 complaints (`baseline.prediction`), for every venue. Since 3.28 is at or above `T = 1`, the baseline badges **every** venue (`baseline.decision_for_every_venue` = `complaint_risk`; badge counts TP 538, FP 1,174, FN 0, TN 0). But a penalty applied to every venue equally leaves the ranking exactly as it was, so groups end up where they would with no badge at all. The value calculation therefore treats the baseline as "no badge": every truly loud venue counts as FN and every quiet one as TN (effective TN 1,174, FN 538). That matches what the product does today.

## 3. Value-quantification worksheet

### 3.1 Assumptions (from `value.assumptions`, `value.rank_penalty`, `threshold`)

All are **ASSUMPTIONS** set in the script; none is measured yet.

| Key in `metrics.json` | Meaning | Figure | Label | Where a real number could come from later |
|---|---|---|---|---|
| `groups_per_month` | Group recommendation requests per month in the launch area (LES) | 6,000 | ASSUMPTION | Product analytics: count of `/api/recommend` calls per month after launch |
| `share_chill_groups` | Share of groups that ask for a chill night | 0.35 | ASSUMPTION | Distribution of onboarding preferences (low `crowd_pref`, no dancing) in stored Preferences |
| `share_no_fresh_scout` | Share of top picks with no fresh scout report (the only case where the rule changes anything) | 0.50 | ASSUMPTION | Live Venue State logs: share of recommended venues with confidence below the "fresh" cut-off at request time |
| `value_good_match_usd` | Value of a night that fits the group (referral fee plus retention) | $4.00 | ASSUMPTION | Affiliate/booking revenue per visit plus retention value; venue partnership fees per referred group |
| `p_bad_night_if_loud` | Chance a truly loud venue ruins a chill group's night | 0.40 | ASSUMPTION | Post-night feedback: share of chill groups rating a loud venue negatively |
| `cost_bad_night_usd` | Cost of a ruined night (group walks out, drops the app for the night) | $25.00 | ASSUMPTION | Drop in 30-day retention after a negative rating, times revenue per active user |
| `cost_wrong_demotion_usd` | Cost when a quiet partner venue is wrongly demoted (loses exposure) | $1.50 | ASSUMPTION | Lost referral revenue from the demoted venue and partner complaints; A/B test of demoted vs not demoted venues |
| `rank_penalty` | Rank-score multiplier for badged venues, chill groups only | 0.7 | ASSUMPTION | A/B test of penalty strengths against chill-group satisfaction |
| `threshold.value` | Badge threshold `T`, complaints in 3 years (training 75th percentile) | 1 | ASSUMPTION | Survey chill users on which complaint level felt "too loud"; or pick `T` to maximise value on a validation split |

### 3.2 Formula (as the script computes it, `monthly_value`)

**Decisions per month** = `groups_per_month × share_chill_groups × share_no_fresh_scout` = 6,000 × 0.35 × 0.5 = **1,050** (`value.v1.decisions_per_month`). A decision is one chill group sent to its top venue when that venue has no fresh scout report.

**Cells.** Using `T`: a test venue is *truly loud* if actual complaints ≥ 1, and *flagged* if the prediction ≥ 1. That gives TP, FP, FN, TN over the `n = 1,712` test rows.

**Dollars per decision by cell** (`value.v1.usd_per_decision_by_cell`):

| Cell | What happens | Formula | $ per decision |
|---|---|---|---|
| TN | quiet venue, not flagged; group goes, good night | `value_good_match` | +4.00 |
| FN | loud venue, not flagged; chill group sent there | `−p_bad_night_if_loud × cost_bad_night` = −0.40 × 25 | −10.00 |
| TP | loud venue, flagged; group steered to an unflagged venue | `value_good_match` | +4.00 |
| FP | quiet venue, flagged; the alternative is fine but the venue loses out | `value_good_match − cost_wrong_demotion` = 4.00 − 1.50 | +2.50 |

**Value per decision** = (TN × 4.00 + FN × (−10.00) + TP × 4.00 + FP × 2.50) / n
**Value per month** = value per decision × 1,050

Groups that did not ask for a chill night are never demoted, so they add the same amount under both rules and are left out.

### 3.3 Hand calculation

**v1** (`value.v1.effective_counts`: TN 988, FP 186, FN 320, TP 218; sum 1,712):

- 988 × 4.00 = 3,952.00
- 186 × 2.50 = 465.00
- 320 × (−10.00) = −3,200.00
- 218 × 4.00 = 872.00
- Total 2,089.00 / 1,712 = **$1.2202 per decision** (`value.v1.usd_per_decision`)
- × 1,050 = **$1,281.22 per month** (`value.v1.usd_per_month`)

**Baseline** (`value.baseline.effective_counts`: TN 1,174, FP 0, FN 538, TP 0; valued as "no badge", see §2):

- 1,174 × 4.00 = 4,696.00
- 538 × (−10.00) = −5,380.00
- Total −684.00 / 1,712 = **−$0.3995 per decision** (`value.baseline.usd_per_decision`)
- × 1,050 = **−$419.51 per month** (`value.baseline.usd_per_month`)

**Difference** = 1,281.22 − (−419.51) = **$1,700.73 per month** (`value.v1_minus_baseline_usd_per_month`).

Shortcut check: only the 218 caught loud venues and the 186 false alarms differ between the two rules. Each catch turns −10.00 into +4.00 (+14.00); each false alarm turns +4.00 into +2.50 (−1.50). (218 × 14.00 − 186 × 1.50) / 1,712 × 1,050 = (3,052 − 279) / 1,712 × 1,050 = $1,700.73.

## 4. Slide checklist

One row per item the assignment requires. "Source" gives the number and its path in `outputs/metrics.json`; the team writes the actual slide wording. `outputs/report.md` shows the same numbers in tables and is a cross-check.

### Title slide

| Item | Source | Chart |
|---|---|---|
| Project / model name | team's choice | none |
| Team name | `<TEAM NAME>` (placeholder) | none |
| Members | `<MEMBER 1>, <MEMBER 2>, ...` (placeholder) | none |
| Repo link | https://github.com/RyanmMehta/manhattan-bars-ml | none |
| Script path | `coursework/group_model/venue_noise_regression.py` | none |

### Content slide 1: data, model specification, baseline vs v1 fit

| Item | Number and source | Chart |
|---|---|---|
| Data source | `DATA.md` §1.1 (SLA `9s3h-dpkz`, DOHMH `43nn-pn8j`, 311 `erm2-nwe9`, OSM Overpass) | none |
| Real vs synthetic note | `DATA.md` §4 (real public data, downloaded October 2026) | none |
| Rows | 5,704 (`data.n_rows`) | none |
| One row per | licensed Manhattan venue (`DATA.md` §1) | none |
| Period | licenses as of 2026-10-20; 311 complaints 2023-10-21 to 2026-10-19 (`DATA.md` §1) | none |
| 70/30 split | 3,992 train / 1,712 test (`split.n_train`, `split.n_test`); random, not stratified, seed 42 (`seed`) | none |
| Outcome | `noise_complaints`, 3-year count: mean 3.29, median 0, 67.5% zero (`data.outcome_mean`, `data.outcome_median`, `data.outcome_share_zero`); fitted on log1p (`scale_choice.chosen`) | none |
| Features | 11 terms (`v1.terms`), listed in §2 | `coefficients.png` |
| Excluded columns and why | `license_id` (identifier), `high_noise` (leakage: derived from the outcome) (`data.dropped`) | none |
| Model family | multiple linear regression, statsmodels OLS on log1p (`v1.model`) | none |
| Key training fit statistic | adjusted R² 0.116, log scale (`v1.adj_r2_train_log`) | none |
| Baseline fit on test rows | R² -0.000, RMSE 14.64, MAE 5.15 complaints (`baseline.test.r2`, `.rmse`, `.mae`) | `baseline_vs_v1_error.png` |
| v1 fit on test rows | R² -0.007, RMSE 14.69, MAE 3.45 complaints (`v1.test.r2`, `.rmse`, `.mae`) | `baseline_vs_v1_error.png` |
| Train vs test R² (overfitting check) | log scale 0.118 vs 0.124 (`v1.r2_train_log`, `v1.r2_test_log`); complaint units -0.012 vs -0.007 (`v1.train.r2`, `v1.test.r2`) | none |
| Fit picture (optional) | predicted vs actual on test rows; residual pattern | `predicted_vs_actual.png`, `residuals.png` |

### Content slide 2: business decision and value

| Item | Number and source | Chart |
|---|---|---|
| Business question (product/ops lead's view) | §1 of this file | none |
| What the product does today | §1 of this file | none |
| One-sentence headline: what the model changes and what it is worth | team writes it, using $1,700.73/month (`value.v1_minus_baseline_usd_per_month`) | none |
| Decision rule | badge + 0.7 rank penalty for chill groups when predicted ≥ 1 complaint (`threshold.value`, `value.rank_penalty`); §2 | none |
| Result for a new input | predicted 4.643 complaints → badge, penalty 0.7 for chill groups (`decision_example.output.predicted_complaints`, `.badge`, `.rank_penalty`); input in `decision_example.input` | none |
| Baseline value | −$419.51/month (`value.baseline.usd_per_month`) | no value chart in the outputs; show as a small table |
| v1 value | $1,281.22/month (`value.v1.usd_per_month`) | same table |
| Difference | +$1,700.73/month (`value.v1_minus_baseline_usd_per_month`) | same table |
| Assumptions | §3.1, each labelled ASSUMPTION (`value.assumptions`) | same table |

## 5. How each number is calculated

The outcome is `noise_complaints`, the number of 311 complaints at a venue in 3 years. The model is fitted on log(1 + complaints); its predictions are turned back into complaints (`expm1`, floored at 0) before any complaint-unit error is computed. "Test" numbers use only the 30% of rows held out from fitting.

- **R² (coefficient of determination):** the share of the venue-to-venue variation that the model explains: 1 − (sum of squared prediction errors) / (sum of squared differences from the mean). 1 is perfect; 0 means no better than predicting the mean; below 0 means worse than the mean.
- **Adjusted R² (training fit):** R² on the training rows, reduced for the number of terms so adding useless terms cannot raise it: 1 − (1 − R²) × (n − 1) / (n − k − 1), here n = 3,992 and k = 11. It is on the log scale, where the model was fitted: 0.116.
- **Test R²:** the same R² formula on the 1,712 test rows. On the log scale 0.124; in complaint units -0.007, because a few extreme venues dominate squared error.
- **RMSE (root mean squared error), in complaints:** square each test error (actual − predicted), average them, take the square root. Big misses count heavily, so a few venues with hundreds of complaints dominate it (baseline 14.64, v1 14.69).
- **MAE (mean absolute error), in complaints:** the average size of the test error, ignoring sign: "on a typical venue the prediction is off by this many complaints". Less sensitive to extreme venues (baseline 5.15, v1 3.45).
- **Training vs test R²:** R² computed on the 70% of rows the model was fitted on and on the 30% it never saw, on the same scale. A training figure far above the test figure means overfitting; here 0.118 vs 0.124 on the log scale, so none.
- **Mean baseline:** the average complaint count of the training rows, 3.28, predicted for every venue. Its test R² is about 0 by construction. It gives every venue the same number, so it makes the same badge decision for every venue (at `T = 1`, badge all), which leaves the ranking unchanged.
- **70/30 split:** shuffle the 5,704 rows with seed 42 and put 70% (3,992) in training and 30% (1,712) in test. Random, not stratified, because the outcome is a count.
- **Threshold T:** the 75th percentile of complaints in the training rows: 75% of training venues have at most this many. Here 1 complaint.
- **Expected value:** mark each test venue truly loud or not (actual ≥ 1) and flagged or not (prediction ≥ 1); count the four cells; multiply each count by its dollars per decision (§3.2); divide by 1,712 for value per decision; multiply by 1,050 decisions per month. Do it for v1 and for the baseline (valued as no badge); the difference is the model's expected monthly value, $1,700.73. It is only as good as the assumptions in §3.1.
