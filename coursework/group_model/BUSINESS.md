# BUSINESS.md: business framing, value worksheet and slide checklist

Reference document for the group model assignment. It collects facts, formulas and sources so the team can write the slides themselves. It contains no slide text: the assignment requires the slides to be written by the team, not generated.

Team name: `<TEAM NAME>` | Members: `<MEMBER 1>, <MEMBER 2>, <MEMBER 3>, ...`
Repo: https://github.com/RyanmMehta/manhattan-bars-ml | Script: `coursework/group_model/venue_noise_tree.py`
Data: see `coursework/group_model/DATA.md`. Model outputs: `coursework/group_model/outputs/metrics.json` and the chart PNGs in `coursework/group_model/outputs/` (written by the script; not in this branch yet).

> **Key names below are proposed.** The script is being written in parallel, so `outputs/metrics.json` keys and PNG file names are not final. Before using a row of the checklist, open the real `metrics.json` and fix the key name in this file if it differs.

## 1. Business question

**Who acts on it:** the Nightlife Autopilot product/ops lead.

**Question (one sentence, fixed wording shared with the model script):**
"When a group asks where to go tonight and we have no fresh scout reports for a venue, should the app label it complaint-prone (loud/rowdy) and demote it for groups who asked for a chill night?"

**What the product does today without the model:** Every venue card, venue profile and recommendation row carries a `complaint_risk` object (`docs/CONTRACTS.md`, backend CR-19: `{label: "311 noise-complaint risk", noise_complaints_3y, level, score}`), but the app shows it only as a small line on the venue profile, the raw 311 count with a low/medium/high level (bands under 25, 25 to 99, 100 or more, marked as a placeholder in `backend/app/config.py`), and never on the recommendation list. By rule it never changes vibe, match score, rank or reasons (`docs/specs/SCORING.md` section 9, `docs/prd/frontend/RULES.md`), so a group that asked for a quiet night can be sent to a complaint-prone venue whenever scout reports are missing; in effect no venue gets a warning badge and none is demoted.

(Related, for context only: `recs/complaint_risk.py` already trains a gradient-boosting score for the same `high_noise` outcome, out-of-fold ROC AUC 0.73, and the backend passes it as `complaint_risk.score`, but the frontend does not display or use it. Our v1 model is a separate, explainable decision tree for this assignment.)

## 2. Decision rule and baseline

**Decision rule in words:** For each venue, the tree predicts `high_noise` (1 = 3 or more 311 complaints in 3 years). If the prediction is 1, the app shows a complaint-risk badge on the venue, and for groups whose preferences ask for a chill night it multiplies the venue's rank score by a penalty so it drops down the list. If the prediction is 0, no badge and no penalty. Groups that did not ask for a chill night see the badge but their ranking is unchanged.

**Pseudocode:**

```
for each venue v in candidate list:
    p = tree.predict(features(v))          # 1 = predicted high_noise, 0 = not
    if p == 1:
        v.badge = "complaint risk"
        if group.wants_chill:               # e.g. low crowd_pref, no dancing
            v.rank_score = v.rank_score * CHILL_PENALTY   # e.g. 0.5 (ASSUMPTION)
    else:
        v.badge = none
re-rank candidates by rank_score
```

Note: "chill night" is not a single field in today's Preferences object; the nearest signals are a low `crowd_pref` and `want_dancing = false` (see `docs/specs/SCORING.md` section on match components). The penalty factor is a design choice, not something the model estimates.

**Baseline in words:** What the product does today: no venue gets a badge or a demotion. In model terms this is the most-common-class baseline: predict `high_noise = 0` for every venue. It is right for the 82.9% of venues that are not high-noise and wrong for every high-noise one (recall 0).

## 3. Value-quantification worksheet

### 3.1 Assumptions

All five figures are **ASSUMPTIONS** chosen to be plausible for a small launch; none is measured yet. Keep the same figures in the script and on the slide; if the script uses different numbers, the script wins and this table should be updated to match.

| # | Assumption | Symbol | Proposed figure | Label | Where a real number could come from later |
|---|---|---|---|---|---|
| A1 | Group recommendation requests per month in the launch area (Lower East Side launch set, 35 venues in `data/launch_venues.csv`) | `R` | 5,000 requests / month | ASSUMPTION | Product analytics: count of `/api/recommend` calls per month after launch |
| A2 | Share of requests from groups who asked for a chill night | `s` | 30% | ASSUMPTION | Distribution of onboarding preferences (low `crowd_pref`, no dancing) in stored Preferences |
| A3 | Cost of a bad recommendation (chill group sent to a complaint-prone venue) | `c_bad` | $6.00 per occurrence | ASSUMPTION | Post-night feedback and churn: drop in 30-day retention after a negative rating, times revenue per active user |
| A4 | Value of a good recommendation (group goes and is satisfied) | `v_good` | $2.00 per occurrence | ASSUMPTION | Affiliate/booking revenue per visit plus retention value; venue partnership fees per referred group |
| A5 | Cost of wrongly demoting a venue that is not complaint-prone | `c_demote` | $1.00 per occurrence | ASSUMPTION | Lost referral revenue from the demoted venue and partner complaints; A/B test of demoted vs not demoted venues |

Monthly chill-group recommendations: `C = R × s = 5,000 × 0.30 = 1,500`.

### 3.2 Formula (recompute by hand from the test confusion matrix)

Use the confusion matrix of the tree on the **test rows** (30% of 5,704, about 1,712 rows), with `high_noise = 1` as the positive class:

| | Actual 1 (high noise) | Actual 0 |
|---|---|---|
| **Predicted 1** | TP | FP |
| **Predicted 0** | FN | TN |

`n = TP + FP + FN + TN` (test rows). Treat each chill-group recommendation as landing on a venue drawn like a test row. Value of one chill-group recommendation, by cell:

| Cell | What happens | Value |
|---|---|---|
| TP | Noisy venue badged and demoted; group goes somewhere suitable | `+v_good` |
| FP | Quiet venue wrongly demoted; group still gets a suitable venue, but the demoted venue loses the referral | `+v_good − c_demote` |
| FN | Noisy venue not flagged; chill group sent there | `−c_bad` |
| TN | Quiet venue, no badge, recommended | `+v_good` |

**v1 (tree), value per chill recommendation:**
`V_tree = [ (TP + TN) × v_good + FP × (v_good − c_demote) − FN × c_bad ] / n`

**Baseline (no badge, predicts 0 for everyone):** all actual positives become FN, all actual negatives become TN. With `P = TP + FN` and `N = FP + TN`:
`V_base = [ N × v_good − P × c_bad ] / n`

**Gain of v1 over baseline, per chill recommendation** (subtract the two; everything else cancels):
`G = [ TP × (v_good + c_bad) − FP × c_demote ] / n`

**Monthly figures:**
- Baseline monthly value: `C × V_base`
- v1 monthly value: `C × V_tree`
- Expected monthly value of the model: `C × G = R × s × [ TP × (v_good + c_bad) − FP × c_demote ] / n`

Reading the gain formula: each caught noisy venue is worth `v_good + c_bad` ($8.00 with the figures above, a bad night avoided and a good one gained); each false alarm costs `c_demote` ($1.00). The model adds value only if `TP × 8 > FP × 1`, that is, if fewer than 8 of every 9 flags are false alarms.

### 3.3 Results (fill in from the script output)

| Quantity | Value | Source |
|---|---|---|
| Test rows `n` | `<from outputs/metrics.json>` | `metrics.json` → `n_test` |
| TP, FP, FN, TN (tree, test) | `<from outputs/metrics.json>` | `metrics.json` → `confusion_matrix_test` |
| Actual positives `P`, negatives `N` (test) | `<from outputs/metrics.json>` | derive: P = TP + FN, N = FP + TN |
| `V_base` per chill recommendation | `<from outputs/metrics.json>` | `metrics.json` → `value.baseline_per_rec` |
| `V_tree` per chill recommendation | `<from outputs/metrics.json>` | `metrics.json` → `value.v1_per_rec` |
| Baseline monthly value `C × V_base` | `<from outputs/metrics.json>` | `metrics.json` → `value.baseline_monthly` |
| v1 monthly value `C × V_tree` | `<from outputs/metrics.json>` | `metrics.json` → `value.v1_monthly` |
| Expected monthly gain `C × G` | `<from outputs/metrics.json>` | `metrics.json` → `value.gain_monthly` |

Check: `value.v1_monthly − value.baseline_monthly` should equal `value.gain_monthly` (rounding aside). Recompute G by hand from the four cells once, as a team, before presenting.

## 4. Slide checklist

One row per item the assignment requires. "Source" says where the number or fact comes from; the team writes the actual slide wording. Key names in `metrics.json` and PNG names are proposed (see the note at the top).

### Title slide

| Item | Source | Chart |
|---|---|---|
| Project / model name | team's choice | none |
| Team name | `<TEAM NAME>` (placeholder) | none |
| Members | `<MEMBER 1>, <MEMBER 2>, ...` (placeholder) | none |
| Repo link | https://github.com/RyanmMehta/manhattan-bars-ml | none |
| Script path | `coursework/group_model/venue_noise_tree.py` | none |

### Content slide 1: data, model specification, baseline vs v1 fit

| Item | Source | Chart |
|---|---|---|
| Data source | `DATA.md` §1.1 (SLA `9s3h-dpkz`, DOHMH `43nn-pn8j`, 311 `erm2-nwe9`, OSM Overpass) | none |
| Real vs synthetic note | `DATA.md` §4 (real public data, downloaded October 2026) | none |
| Rows | 5,704 (`DATA.md` §1; `metrics.json` → `n_rows`) | none |
| One row per | licensed Manhattan venue (`DATA.md` §1) | none |
| Period | licenses as of 2026-10-20; 311 complaints 2023-10-21 to 2026-10-19 (`DATA.md` §1) | none |
| 70/30 split | `metrics.json` → `n_train`, `n_test`, `split` (stratified, fixed seed) | none |
| Outcome | `high_noise` = 1 if 3 or more 311 complaints in 3 years; 17.11% positive (`DATA.md` §2; `metrics.json` → `positive_rate`) | optional: class-balance bar |
| Features | list in `DATA.md` §3 (Role = Feature); exact list used: `metrics.json` → `features` | optional: `outputs/feature_importance.png` |
| Excluded column and why | `noise_complaints`, leakage (`DATA.md` §3) | none |
| Model family | decision tree classifier; depth and settings: `metrics.json` → `model.params` (for example `max_depth`, `min_samples_leaf`) | optional: `outputs/tree.png` |
| Key training fit statistic | training accuracy: `metrics.json` → `v1.train.accuracy` | none |
| Baseline fit on test rows | `metrics.json` → `baseline.test.accuracy`, `.precision`, `.recall` | `outputs/baseline_vs_v1.png` |
| v1 fit on test rows | `metrics.json` → `v1.test.accuracy`, `.precision`, `.recall` | `outputs/baseline_vs_v1.png` |
| Train vs test gap (overfitting check) | `v1.train.accuracy` vs `v1.test.accuracy` | none |
| Confusion matrix (optional) | `metrics.json` → `confusion_matrix_test` | optional: `outputs/confusion_matrix.png` |

### Content slide 2: business decision and value

| Item | Source | Chart |
|---|---|---|
| Business question (product/ops lead's view) | §1 of this file | none |
| What the product does today | §1 of this file | none |
| One-sentence headline: what the model changes and what it is worth | team writes it, using `value.gain_monthly` | none |
| Decision rule | §2 of this file | none |
| Result for a new input | `metrics.json` → `example_new_input` (feature values, predicted class, badge yes/no, penalty yes/no) | none |
| Baseline value | `metrics.json` → `value.baseline_monthly` (formula §3.2) | `outputs/value_baseline_vs_v1.png` |
| v1 value | `metrics.json` → `value.v1_monthly` (formula §3.2) | `outputs/value_baseline_vs_v1.png` |
| Gain and assumptions | `value.gain_monthly`; assumptions table §3.1, each labelled ASSUMPTION | same chart |

## 5. How each number is calculated

Positive class throughout is `high_noise = 1` (complaint-prone venue). All "test" numbers use only the 30% of rows held out from training.

- **Accuracy:** the share of venues the model labels correctly, either way: (TP + TN) / all rows. With 17% positives, a model that never flags anything already scores about 83%, so accuracy alone says little here.
- **Precision:** of the venues the model flags as complaint-prone, the share that really are: TP / (TP + FP). Low precision means many unfair badges on quiet venues.
- **Recall:** of the venues that really are complaint-prone, the share the model flags: TP / (TP + FN). Low recall means many chill groups still get sent to loud venues.
- **Training vs test accuracy:** the same accuracy formula, computed once on the 70% of rows the tree learned from and once on the 30% it never saw. A training figure far above the test figure means the tree memorised the training rows (overfitting); a small gap means it generalises.
- **Most-common-class baseline:** count which class is more common in the training rows (0, not complaint-prone) and predict it for every venue. Its accuracy equals the share of 0s in the test rows; its precision is undefined (no flags) and its recall is 0. It stands for "the product today: no badges".
- **70/30 stratified split:** shuffle the 5,704 rows with a fixed random seed and put 70% in training and 30% in test, drawing separately within each class so both parts keep the same 17% / 83% balance as the full table.
- **Expected value:** for each chill-group recommendation, add up what each confusion-matrix cell is worth (§3.2), divide by the number of test rows to get value per recommendation, then multiply by the monthly number of chill-group recommendations (`R × s`). Do this for the tree and for the baseline; the difference is the model's expected monthly value, `R × s × [TP × (v_good + c_bad) − FP × c_demote] / n`. It is only as good as the five assumptions in §3.1.
