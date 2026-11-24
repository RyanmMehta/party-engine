"""Group model (bar_popularity_regression.py): reproducible split, clean features, valid decision."""
import json
import math

import numpy as np
import pytest
import statsmodels.api as sm

from coursework.group_model import bar_popularity_regression as bpr

pytestmark = pytest.mark.skipif(not bpr.YELP_FILE.exists(),
                                reason="Yelp business file not downloaded (run the script once)")


@pytest.fixture(scope="module")
def bars():
    return bpr.load_bars()


def test_split_is_reproducible(bars):
    tr1, te1 = bpr.split(bars, 42)
    tr2, te2 = bpr.split(bars, 42)
    assert list(te1.index) == list(te2.index)
    assert len(te1) == math.ceil(0.30 * len(bars))


def test_features_have_no_outcome_or_missing_flags(bars):
    X = bpr.design_matrix(bars)
    assert list(X.columns) == ["const"] + bpr.TERMS
    assert not {"review_count", "stars", "price_unknown", "noise_unknown", "city"} & set(X.columns)


def test_decide_returns_valid_decision(bars):
    train, _ = bpr.split(bars, 42)
    model = sm.OLS(np.log1p(train[bpr.TARGET]), bpr.design_matrix(train)).fit()
    bar = {k: v for k, v in json.loads(bpr.EXAMPLE.read_text()).items() if not k.startswith("_")}
    out = bpr.decide(bar, model, 200.0)
    assert out["predicted_reviews"] >= 0 and isinstance(out["scout_bounty"], bool)
    with pytest.raises(ValueError):
        bpr.decide({"categories": "Bars, Nightlife"}, model, 200.0)
