# coursework/group_model: group model assignment, v1 linear regression

One standalone script fits one multiple linear regression on the real `venue_features` table
(`data/manhattan_bars.db`, 5,704 licensed Manhattan venues). It predicts `noise_complaints`, the
count of NYC 311 noise complaints matched to the venue's address over 3 years. The model is
statsmodels OLS on log(1 + complaints) with 11 readable terms. It is compared with a baseline that
predicts the training mean. The script runs the badge decision rule on a new venue and puts a
monthly dollar value on both rules. It is not wired into the app.

The data dictionary is in `DATA.md` and the business question and assumptions are in
`BUSINESS.md`. Both are written separately.

## Run

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r coursework/group_model/requirements.txt
.venv/bin/python coursework/group_model/venue_noise_regression.py               # seed 42, example venue
.venv/bin/python coursework/group_model/venue_noise_regression.py --help
.venv/bin/python coursework/group_model/venue_noise_regression.py --seed 7 --new-input my_venue.json
.venv/bin/python -m pytest coursework/tests/test_group_model.py -q              # 3 tests, about 3 s
```

The run takes a few seconds. It opens the database read-only and rewrites everything in `outputs/`.
A `--new-input` file is one JSON object with raw `venue_features` columns. The model reads
`venue_type`, `liquor_tier`, `osm_amenity`, `osm_closes_late`, `neighborhood`, `density_100m` and
`latest_grade`. A missing category falls into the base case. A missing `density_100m` gets the
training median. Keys starting with `_` are ignored.

## Files

| File | What it is |
|---|---|
| `venue_noise_regression.py` | the whole pipeline: load, split, baseline, OLS, evaluation, `decide()`, value, outputs |
| `example_new_venue.json` | the new input: a newly licensed LES Bar/Tavern (zip 10002), full liquor, closes late, no inspections yet |
| `outputs/metrics.json` | every number the script prints, plus the example input and output and the versions |
| `outputs/report.md` | the same numbers as tables |
| `outputs/ols_summary.txt` | the full statsmodels summary: coefficients, p-values, adjusted R² |
| `outputs/predicted_vs_actual.png` | test rows: v1 prediction against actual complaints, with the baseline mean and the badge threshold, log axes |
| `outputs/baseline_vs_v1_error.png` | test R², RMSE and MAE in complaints, baseline and v1 |
| `outputs/coefficients.png` | each term's % change in (complaints + 1) with its 95% interval |
| `outputs/residuals.png` | training residuals on the log scale against fitted values, and their histogram |
| `requirements.txt` | package versions used for the committed outputs |

The test file is `coursework/tests/test_group_model.py`. It checks that the split is reproducible,
that no leakage column reaches the model, and that `decide()` returns a valid decision.

## Choices made in the code

Each choice is explained in a comment where it is made in `venue_noise_regression.py`:

- `high_noise` and `license_id` are dropped. `high_noise` is `noise_complaints >= 3`, so it is leakage.
- `lat` and `lon` are left out. The neighbourhood groups carry location instead.
- The outcome is modelled as log1p because it fits clearly better on the training rows.
  Test errors are reported back in complaints.
- Rare and similar categories are collapsed into grouped terms.
- The badge threshold is the training 75th percentile.
- Every dollar figure is in the `ASSUMPTIONS` dict at the top of the file.

## Versions used for the committed outputs

Python 3.14.3, pandas 3.1.1, numpy 2.5.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1,
matplotlib 3.11.2 (macOS, seed 42).
