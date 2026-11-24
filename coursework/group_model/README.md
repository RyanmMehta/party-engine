# coursework/group_model: group model assignment, v1 linear regression

One standalone script, `bar_popularity_regression.py`, predicts how popular a bar is, measured by
its Yelp review count. It uses only traits the app shows groups: price, crowd-reported noise level,
music (DJ, live, karaoke, jukebox), dancing, happy hour, late closing, outdoor seating, TVs, food
and bar type. The model is statsmodels OLS on log(1 + reviews), compared with a baseline that
predicts the training mean. The script runs the Scout-bounty decision on a new bar and puts a
monthly dollar value on both rules. It is not wired into the app.

- `DATA.md`: where the data comes from, rows, period, the filter, and the data dictionary.
- `BUSINESS.md`: the question, decision rule, baseline, assumptions, a slide checklist, and how
  every number is calculated.

## Run

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r coursework/group_model/requirements.txt
.venv/bin/python coursework/group_model/bar_popularity_regression.py          # seed 42, example bar
.venv/bin/python coursework/group_model/bar_popularity_regression.py --help
.venv/bin/python -m pytest coursework/tests/test_group_model.py -q            # 3 tests
```

The first run downloads the Yelp Open Dataset business file (about 119 MB) to
`data/external/yelp/`. That folder is gitignored because Yelp's data agreement allows academic use
but not redistribution. Later runs take about 10 seconds and rewrite everything in `outputs/`.

A `--new-input` file is one JSON object in Yelp's raw format: `categories`, `attributes` and
`hours`, as in `example_new_bar.json`. Price level and noise level are required.

## Outputs

| file | what it is |
|---|---|
| `outputs/metrics.json` | every number the script prints |
| `outputs/report.md` | the same numbers as tables |
| `outputs/ols_summary.txt` | the full statsmodels summary |
| `outputs/baseline_vs_v1_error.png` | test R² (log scale), MAE and median absolute error, baseline vs v1 |
| `outputs/predicted_vs_actual.png` | each test bar, predicted vs actual reviews, with the baseline and threshold |
| `outputs/coefficients.png` | % change in expected reviews for each trait |
| `outputs/residuals.png` | residuals on the test bars |

Versions used for the committed outputs are listed at the end of `outputs/report.md`.
