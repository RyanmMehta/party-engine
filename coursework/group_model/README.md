# coursework/group_model: group model assignment, v1 decision tree

One standalone script fits one decision tree on the real `venue_features` table
(`data/manhattan_bars.db`, 5,704 licensed Manhattan venues) to predict `high_noise`
(1 when the venue's address has 3 or more NYC 311 noise complaints in 3 years). It compares
the tree with a most-common-class baseline, runs the decision rule on a new venue, and puts a
monthly dollar value on both. It is not wired into the app.

The data dictionary is in `DATA.md` and the business question and assumptions are in
`BUSINESS.md`. Both are written separately.

## Run

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r coursework/group_model/requirements.txt
.venv/bin/python coursework/group_model/venue_noise_tree.py               # seed 42, example venue
.venv/bin/python coursework/group_model/venue_noise_tree.py --help
.venv/bin/python coursework/group_model/venue_noise_tree.py --seed 7 --new-input my_venue.json
.venv/bin/python -m pytest coursework/tests/test_group_model.py -q        # 3 tests, about 5 s
```

The run takes about 5 seconds. It opens the database read-only and rewrites everything in `outputs/`.
A `--new-input` file is one JSON object with the raw `venue_features` columns, without
`license_id`, `noise_complaints` and `high_noise`. Missing fields get the training median, or
the level "missing" for categories. Keys starting with `_` are ignored.

## Files

| File | What it is |
|---|---|
| `venue_noise_tree.py` | the whole pipeline: load, split, baseline, tree, evaluation, `decide()`, value, outputs |
| `example_new_venue.json` | the new input: a newly licensed LES Bar/Tavern (zip 10002), full liquor, closes late, no inspections yet |
| `outputs/metrics.json` | every number the script prints, plus the example input and output and the versions |
| `outputs/report.md` | the same numbers as tables |
| `outputs/tree.txt` | the fitted tree as text (`export_text`) |
| `outputs/confusion_matrices.png` | baseline and v1 confusion matrices on the test rows, side by side |
| `outputs/baseline_vs_v1.png` | test accuracy, precision and recall for baseline and v1 |
| `outputs/tree.png` | the fitted tree (`plot_tree`); `value` is the class-weighted share in each node |
| `outputs/feature_importance.png` | the tree's feature importances, nonzero ones only |
| `requirements.txt` | package versions used for the committed outputs |

The test file is `coursework/tests/test_group_model.py`. It checks that the split is reproducible
and stratified, that no leakage column reaches the model, and that `decide()` returns a valid
decision.

## Choices made in the code

Each choice is explained in a comment where it is made in `venue_noise_tree.py`:

- `noise_complaints` is dropped because `high_noise` is computed from it.
- `lat` and `lon` are kept.
- NaNs are imputed with the training median.
- Depth is fixed at 4. `min_samples_leaf` is tuned by 5-fold cross-validation on the training rows.
- `class_weight="balanced"` is used. A reference unweighted tree is printed next to v1 for comparison.
- Every dollar figure is in the `ASSUMPTIONS` dict at the top of the file.

## Versions used for the committed outputs

Python 3.14.3, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2 (macOS, seed 42).
