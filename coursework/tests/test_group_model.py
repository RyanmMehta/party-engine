"""Group model (venue_noise_tree.py): the split is reproducible and decide() returns a valid decision."""
import json
import math

import pytest

from coursework.group_model import venue_noise_tree as vnt

pytestmark = pytest.mark.skipif(not vnt.DB.exists(), reason="data/manhattan_bars.db not present")


@pytest.fixture(scope="module")
def venues():
    return vnt.load_venues().drop(columns=[vnt.ID_COL] + vnt.LEAKAGE)


def test_split_is_reproducible_and_stratified(venues):
    tr1, te1 = vnt.split(venues, 42)
    tr2, te2 = vnt.split(venues, 42)
    assert list(tr1.index) == list(tr2.index) and list(te1.index) == list(te2.index)
    assert len(te1) == math.ceil(0.30 * len(venues))  # sklearn rounds the test size up
    assert abs(tr1[vnt.TARGET].mean() - te1[vnt.TARGET].mean()) < 0.005
    assert list(vnt.split(venues, 7)[1].index) != list(te1.index)


def test_no_leakage_in_model_columns(venues):
    prep = vnt.Preprocessor().fit(venues)
    assert not any(c.startswith(("noise_complaints", "high_noise", "license_id")) for c in prep.columns)


def test_decide_returns_valid_decision(venues):
    train, _ = vnt.split(venues, 42)
    prep = vnt.Preprocessor().fit(train)
    model, _, _ = vnt.fit_v1(prep.transform(train), train[vnt.TARGET].to_numpy(), 42)
    example = {k: v for k, v in json.loads(vnt.EXAMPLE.read_text()).items() if not k.startswith("_")}
    # the example, plus a venue with almost every field missing and an unseen category
    for venue in (example, {"venue_type": "Rooftop", "neighborhood": "LES/Chinatown"}):
        d = vnt.decide(venue, model=model, prep=prep)
        assert d["badge"] in {"complaint_risk", "none"}
        assert 0.0 <= d["p_high_noise"] <= 1.0
        assert d["rank_penalty"] == (vnt.RANK_PENALTY if d["badge"] == "complaint_risk" else 1.0)
