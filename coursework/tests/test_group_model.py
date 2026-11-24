"""Group model (venue_noise_regression.py): the split is reproducible and decide() returns a valid decision."""
import json
import math

import numpy as np
import pytest
import statsmodels.api as sm

from coursework.group_model import venue_noise_regression as vnr

pytestmark = pytest.mark.skipif(not vnr.DB.exists(), reason="data/manhattan_bars.db not present")


@pytest.fixture(scope="module")
def venues():
    return vnr.load_venues().drop(columns=vnr.DROP)


def test_split_is_reproducible(venues):
    tr1, te1 = vnr.split(venues, 42)
    tr2, te2 = vnr.split(venues, 42)
    assert list(tr1.index) == list(tr2.index) and list(te1.index) == list(te2.index)
    assert len(te1) == math.ceil(0.30 * len(venues))  # sklearn rounds the test size up
    assert list(vnr.split(venues, 7)[1].index) != list(te1.index)


def test_no_leakage_in_model_terms(venues):
    X = vnr.design_matrix(venues, 10.0)
    assert list(X.columns) == ["const"] + vnr.TERM_NAMES
    assert not {"noise_complaints", "high_noise", "license_id"} & set(X.columns)
    assert 8 <= len(vnr.TERM_NAMES) <= 15


def test_decide_returns_valid_decision(venues):
    train, _ = vnr.split(venues, 42)
    med = float(train["density_100m"].median())
    model = sm.OLS(np.log1p(train[vnr.TARGET]), vnr.design_matrix(train, med)).fit()
    example = {k: v for k, v in json.loads(vnr.EXAMPLE.read_text()).items() if not k.startswith("_")}
    # the example, plus a venue with almost every field missing and an unseen category
    for venue in (example, {"venue_type": "Rooftop", "neighborhood": "LES/Chinatown"}):
        d = vnr.decide(venue, model=model, density_median=med, threshold=1.0)
        assert d["badge"] in {"complaint_risk", "none"}
        assert d["predicted_complaints"] >= 0
        assert (d["badge"] == "complaint_risk") == (d["predicted_complaints"] >= 1.0)
        assert d["rank_penalty"] == (vnr.RANK_PENALTY if d["badge"] == "complaint_risk" else 1.0)
