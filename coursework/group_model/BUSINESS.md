# BUSINESS.md: business framing, value worksheet and slide checklist

Reference document for the group model assignment. It collects facts, formulas and sources so the team can write the slides themselves. It contains no slide text: the assignment requires the slides to be written by the team, not generated.

Team name: `<TEAM NAME>` | Members: `<MEMBER 1>, <MEMBER 2>, <MEMBER 3>, ...`
Repo: https://github.com/RyanmMehta/manhattan-bars-ml | Script: `coursework/group_model/venue_noise_regression.py`
Data: see `coursework/group_model/DATA.md`. Model outputs (written by the script, not in this branch yet): `coursework/group_model/outputs/metrics.json`, `outputs/report.md`, and the charts `outputs/predicted_vs_actual.png`, `outputs/baseline_vs_v1_error.png`, `outputs/coefficients.png`, `outputs/residuals.png`.

> **Key names below are proposed.** The script is being written in parallel, so `outputs/metrics.json` key names are not final. Before using a row of the checklist, open the real `metrics.json` and fix the key name in this file if it differs.

## 1. Business question

**Who acts on it:** the Nightlife Autopilot product/ops lead.

**Question (one sentence, fixed wording shared with the model script):**
"For a venue with no fresh scout reports, how many noise complaints should we expect, and is that enough to badge it as complaint-prone and demote it for groups who asked for a chill night?"

**What the product does today without the model:** Every venue card, venue profile and recommendation row carries a `complaint_risk` object (`docs/CONTRACTS.md`, backend CR-19: `{label: "311 noise-complaint risk", noise_complaints_3y, level, score}`), but the app shows it only as a small line on the venue profile, the raw 311 count with a low/medium/high level (bands under 25, 25 to 99, 100 or more, marked as a placeholder in `backend/app/config.py`), and never on the recommendation list. By rule it never changes vibe, match score, rank or reasons (`docs/specs/SCORING.md` section 9, `docs/prd/frontend/RULES.md`), so a group that asked for a quiet night can be sent to a complaint-prone venue whenever scout reports are missing; in effect no venue gets a warning badge and none is demoted.

(Related, for context only: `recs/complaint_risk.py` already trains a gradient-boosting classifier for "3 or more complaints", out-of-fold ROC AUC 0.73, and the backend passes it as `complaint_risk.score`, but the frontend does not display or use it. Our v1 model is a separate, explainable linear regression for this assignment.)

## 2. Model, decision rule and baseline

**Model:** multiple linear regression (statsmodels OLS) predicting `noise_complaints`, the 3-year count of 311 complaints matched to the venue. The script may fit on `log1p(noise_complaints)` because the count is heavily skewed (median 0, max 567; see `DATA.md` §2); predictions are then converted back with `expm1` and every error is reported in complaints.

**Decision rule in words:** For each venue, the regression predicts the expected number of complaints. If the prediction is at or above a threshold `T` (an ASSUMPTION set in the script), the app shows a complaint-risk badge, and for groups whose preferences ask for a chill night it multiplies the venue's rank score by a penalty so it drops down the list. Below `T`: no badge, no penalty. Groups that did not ask for a chill night see the badge but their ranking is unchanged.

**Pseudocode:**

```
T = <threshold from the script>                 # ASSUMPTION, complaints in 3 years
for each venue v in candidate list:
    yhat = regression.predict(features(v))      # expected complaints (back-transformed if log1p)
    if yhat >= T:
        v.badge = "complaint risk"
        if group.wants_chill:                    # e.g. low crowd_pref, want_dancing = false
            v.rank_score = v.rank_score * CHILL_PENALTY   # e.g. 0.5 (ASSUMPTION)
    else:
        v.badge = none
re-rank candidates by rank_score
```

Note: "chill night" is not a single field in today's Preferences object; the nearest signals are a low `crowd_pref` and `want_dancing = false` (see the match components in `docs/specs/SCORING.md`). The penalty factor is a design choice, not something the model estimates.

**Baseline in words:** predict the training-set mean (about 3.3 complaints) for every venue. Because every venue gets the same prediction, every venue gets the same decision: if the mean is below `T`, no venue is badged (this matches what the product does today); if the mean is at or above `T`, every venue is badged.

> **Watch the threshold.** The full-table mean is 3.29 complaints. With `T = 3` the mean baseline badges *every* venue, which no longer matches "the product today shows no badge". With any `T` above the training mean (for example 5), the baseline badges no venue, which does match today. Agree `T` with the script before the slides are written. Also note that `expm1` of a log-scale prediction sits below the raw mean, so a log1p model flags fewer venues at a given `T` than a raw-count model would.

## 3. Value-quantification worksheet

### 3.1 Assumptions

All six figures are **ASSUMPTIONS** chosen to be plausible for a small launch; none is measured yet. Keep the same figures in the script and on the slide; if the script uses different numbers, the script wins and this table should be updated to match.

| # | Assumption | Symbol | Proposed figure | Label | Where a real number could come from later |
|---|---|---|---|---|---|
| A1 | Group recommendation requests per month in the launch area (Lower East Side launch set, 35 venues in `data/launch_venues.csv`) | `R` | 5,000 requests / month | ASSUMPTION | Product analytics: count of `/api/recommend` calls per month after launch |
| A2 | Share of requests from groups who asked for a chill night | `s` | 30% | ASSUMPTION | Distribution of onboarding preferences (low `crowd_pref`, no dancing) in stored Preferences |
| A3 | Cost of a bad recommendation (chill group sent to a truly loud venue) | `c_bad` | $6.00 per occurrence | ASSUMPTION | Post-night feedback and churn: drop in 30-day retention after a negative rating, times revenue per active user |
| A4 | Value of a good recommendation (group goes and is satisfied) | `v_good` | $2.00 per occurrence | ASSUMPTION | Affiliate/booking revenue per visit plus retention value; venue partnership fees per referred group |
| A5 | Cost of wrongly demoting a venue that is not truly loud | `c_demote` | $1.00 per occurrence | ASSUMPTION | Lost referral revenue from the demoted venue and partner complaints; A/B test of demoted vs not demoted venues |
| A6 | Badge threshold: predicted complaints in 3 years at or above which a venue is badged; also the line for "truly loud" (actual complaints at or above it) | `T` | `<from the script>`; this file proposes 5 | ASSUMPTION | Survey chill-preference users after visits and find the complaint count above which they report "too loud"; or pick `T` to maximise value on a validation split |

Monthly chill-group recommendations: `C = R × s = 5,000 × 0.30 = 1,500`.

### 3.2 Formula (recompute by hand from the test rows)

The regression outputs a number, so each rule is scored by turning test venues into a 2 × 2 table using `T`:

- **Truly loud** = actual `noise_complaints >= T`.
- **Badged** = the rule's prediction `>= T` (for v1, the regression's back-transformed prediction; for the baseline, the training mean, so either all or none).

| | Truly loud (actual ≥ T) | Not truly loud |
|---|---|---|
| **Badged** (prediction ≥ T) | TP | FP |
| **Not badged** | FN | TN |

`n = TP + FP + FN + TN` = number of test rows (30% of 5,704, about 1,712). `P = TP + FN` truly loud test venues, `N = FP + TN` the rest; `P` and `N` are the same for both rules.

Value of one chill-group recommendation, by cell (treat each recommendation as landing on a venue drawn like a test row):

| Cell | What happens | Value |
|---|---|---|
| TP | Loud venue badged and demoted; group goes somewhere suitable | `+v_good` |
| FP | Quiet venue wrongly demoted; group still gets a suitable venue, but the demoted venue loses the referral | `+v_good − c_demote` |
| FN | Loud venue not badged; chill group sent there | `−c_bad` |
| TN | Quiet venue, no badge, recommended | `+v_good` |

**Value per chill recommendation, for any rule** (fill in that rule's four cells):
`V = [ (TP + TN) × v_good + FP × (v_good − c_demote) − FN × c_bad ] / n`

Compute it twice: `V_v1` with the regression's cells, `V_base` with the mean baseline's cells.

The baseline's cells are simple because it makes one decision for everybody:
- Mean below `T` (no badges): TP = 0, FP = 0, FN = P, TN = N, so `V_base = [ N × v_good − P × c_bad ] / n`.
- Mean at or above `T` (all badged): TP = P, FP = N, FN = 0, TN = 0, so `V_base = [ n × v_good − N × c_demote ] / n`.

**Gain of v1 over baseline, per chill recommendation:** `G = V_v1 − V_base`. Short forms (everything else cancels):
- Baseline badges nobody: `G = [ TP × (v_good + c_bad) − FP × c_demote ] / n`
- Baseline badges everybody: `G = [ TN × c_demote − FN × (v_good + c_bad) ] / n`

**Monthly figures:**
- Baseline monthly value: `C × V_base`
- v1 monthly value: `C × V_v1`
- Expected monthly value of the model: `C × G = R × s × G`

Reading the first short form: each truly loud venue the regression catches is worth `v_good + c_bad` ($8.00 with the figures above, a bad night avoided and a good one gained); each false alarm costs `c_demote` ($1.00).

### 3.3 Results (fill in from the script output)

| Quantity | Value | Source |
|---|---|---|
| Threshold `T` | `<from outputs/metrics.json>` | `metrics.json` → `decision.threshold` |
| Test rows `n` | `<from outputs/metrics.json>` | `metrics.json` → `n_test` |
| TP, FP, FN, TN for v1 (test) | `<from outputs/metrics.json>` | `metrics.json` → `decision.v1.confusion` |
| TP, FP, FN, TN for baseline (test) | `<from outputs/metrics.json>` | `metrics.json` → `decision.baseline.confusion` |
| Truly loud `P`, not loud `N` (test) | `<from outputs/metrics.json>` | derive: P = TP + FN, N = FP + TN |
| `V_base` per chill recommendation | `<from outputs/metrics.json>` | `metrics.json` → `value.baseline_per_rec` |
| `V_v1` per chill recommendation | `<from outputs/metrics.json>` | `metrics.json` → `value.v1_per_rec` |
| Baseline monthly value `C × V_base` | `<from outputs/metrics.json>` | `metrics.json` → `value.baseline_monthly` |
| v1 monthly value `C × V_v1` | `<from outputs/metrics.json>` | `metrics.json` → `value.v1_monthly` |
| Expected monthly gain `C × G` | `<from outputs/metrics.json>` | `metrics.json` → `value.gain_monthly` |

Check: `value.v1_monthly − value.baseline_monthly` should equal `value.gain_monthly` (rounding aside). Recompute `G` by hand from the cells once, as a team, before presenting.

## 4. Slide checklist

One row per item the assignment requires. "Source" says where the number or fact comes from; the team writes the actual slide wording. Key names in `metrics.json` are proposed (see the note at the top); `outputs/report.md` summarises the same numbers in prose and is a cross-check.

### Title slide

| Item | Source | Chart |
|---|---|---|
| Project / model name | team's choice | none |
| Team name | `<TEAM NAME>` (placeholder) | none |
| Members | `<MEMBER 1>, <MEMBER 2>, ...` (placeholder) | none |
| Repo link | https://github.com/RyanmMehta/manhattan-bars-ml | none |
| Script path | `coursework/group_model/venue_noise_regression.py` | none |

### Content slide 1: data, model specification, baseline vs v1 fit

| Item | Source | Chart |
|---|---|---|
| Data source | `DATA.md` §1.1 (SLA `9s3h-dpkz`, DOHMH `43nn-pn8j`, 311 `erm2-nwe9`, OSM Overpass) | none |
| Real vs synthetic note | `DATA.md` §4 (real public data, downloaded October 2026) | none |
| Rows | 5,704 (`DATA.md` §1; `metrics.json` → `n_rows`) | none |
| One row per | licensed Manhattan venue (`DATA.md` §1) | none |
| Period | licenses as of 2026-10-20; 311 complaints 2023-10-21 to 2026-10-19 (`DATA.md` §1) | none |
| 70/30 split | `metrics.json` → `n_train`, `n_test`, `split` (random, fixed seed) | none |
| Outcome | `noise_complaints`, 3-year count; mean 3.29, median 0, 67.5% zero (`DATA.md` §2); whether fitted on log1p: `metrics.json` → `model.target_transform` | none |
| Features | list in `DATA.md` §3 (Role = Feature); exact list and dummy coding: `metrics.json` → `features` | `outputs/coefficients.png` |
| Excluded column and why | `high_noise`, leakage: derived from the outcome (`DATA.md` §3) | none |
| Model family | multiple linear regression, statsmodels OLS (`metrics.json` → `model.family`) | none |
| Key training fit statistic | adjusted R² on training rows: `metrics.json` → `v1.train.adj_r2` | none |
| Baseline fit on test rows | mean baseline: `metrics.json` → `baseline.test.r2`, `.rmse`, `.mae` | `outputs/baseline_vs_v1_error.png` |
| v1 fit on test rows | `metrics.json` → `v1.test.r2`, `.rmse`, `.mae` (complaint units) | `outputs/baseline_vs_v1_error.png` |
| Train vs test R² (overfitting check) | `v1.train.r2` vs `v1.test.r2` | none |
| Fit picture (optional) | predicted vs actual on test rows; residual pattern | `outputs/predicted_vs_actual.png`, `outputs/residuals.png` |

### Content slide 2: business decision and value

| Item | Source | Chart |
|---|---|---|
| Business question (product/ops lead's view) | §1 of this file | none |
| What the product does today | §1 of this file | none |
| One-sentence headline: what the model changes and what it is worth | team writes it, using `value.gain_monthly` | none |
| Decision rule (with `T`) | §2 of this file; `metrics.json` → `decision.threshold` | none |
| Result for a new input | `metrics.json` → `example_new_input` (feature values, predicted complaints, badge yes/no, penalty yes/no) | none |
| Baseline value | `metrics.json` → `value.baseline_monthly` (formula §3.2) | no chart in the planned outputs; show as a small table |
| v1 value | `metrics.json` → `value.v1_monthly` (formula §3.2) | same table |
| Gain and assumptions | `value.gain_monthly`; assumptions table §3.1, each labelled ASSUMPTION | same table |

## 5. How each number is calculated

The outcome is `noise_complaints`, the number of 311 complaints at a venue in 3 years. All "test" numbers use only the 30% of rows held out from fitting. If the model is fitted on `log1p(noise_complaints)`, its predictions are turned back into complaints with `expm1` before any test error is computed, so every error below is in complaints.

- **R² (coefficient of determination):** the share of the venue-to-venue variation in complaints that the model explains: 1 − (sum of squared prediction errors) / (sum of squared differences from the mean). 1 is perfect; 0 means no better than predicting the mean; below 0 means worse than the mean.
- **Adjusted R² (training fit):** R² on the training rows, reduced to account for the number of features, so adding useless features cannot raise it: 1 − (1 − R²) × (n − 1) / (n − k − 1), with n training rows and k feature columns (dummies count). statsmodels reports it as `rsquared_adj`. If the fit is on log1p, this figure is on the log scale.
- **Test R²:** the same R² formula, computed on the test rows in complaint units, using the training mean as the reference. It answers "how much better than the mean baseline is the model on venues it never saw".
- **RMSE (root mean squared error), in complaints:** square each test error (actual − predicted), average them, take the square root. Big misses count heavily, so a few venues with hundreds of complaints dominate it.
- **MAE (mean absolute error), in complaints:** the average size of the test error, ignoring sign. "On a typical venue the prediction is off by this many complaints." Less sensitive to the extreme venues than RMSE.
- **Training vs test R²:** R² computed once on the 70% of rows the model was fitted on and once on the 30% it never saw. A training figure far above the test figure means the model fits quirks of the training rows (overfitting); a small gap means it generalises. Compare on the same scale.
- **Mean baseline:** take the average complaint count of the training rows (about 3.3) and predict it for every venue. Its test R² is about 0 by construction (slightly negative if the test mean differs from the training mean); its RMSE is close to the standard deviation of complaints (about 14.8 on the full table). Because it gives every venue the same number, it makes the same badge decision for every venue.
- **70/30 split:** shuffle the 5,704 rows with a fixed random seed, fit on 70% (about 3,992 rows) and score on the other 30% (about 1,712; scikit-learn rounds the test share up). The fixed seed makes the split repeatable.
- **Expected value:** using `T`, mark each test venue as truly loud or not (actual ≥ T) and as badged or not (prediction ≥ T); count the four cells; add up what each cell is worth (§3.2); divide by the number of test rows to get value per chill recommendation; multiply by the monthly number of chill-group recommendations (`R × s`). Do this for the regression and for the mean baseline; the difference is the model's expected monthly value. It is only as good as the six assumptions in §3.1.
