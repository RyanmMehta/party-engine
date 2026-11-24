# Group model assignment: team plan

Team: Nightlife Autopilot (Party Engine), Group 7. Meira Berlow, Abhiviraj, Ryan Mehta, Dawid Litwiniec.

**Hand in:** one PDF per team: a title slide plus two slides for a business owner.
**Rule:** AI may write code and charts. The slide text must be written by us, in our own words.

## What we have (done)

- **The question:** which bars will groups ask about most in a new neighbourhood, so the ops lead
  pays Scout bounties to keep their live info fresh. See `BUSINESS.md` §1.
- **The data:** real Yelp Open Dataset, 5,406 open bars in 11 North American metros (not New York),
  with price, crowd-reported noise, music, dancing, happy hour, hours and bar type. See `DATA.md`.
- **The 70/30 split:** 3,784 training bars, 1,622 test bars, seed 42.
- **The model (v1):** linear regression on log(1 + Yelp reviews), 25 traits. Adjusted R² 0.344.
- **Baseline vs v1 on test bars:** R² (log scale) −0.26 vs 0.33. Typical miss 130 vs 57 reviews.
- **Decision rule as code:** `decide()` in `bar_popularity_regression.py`. Bounty if predicted
  reviews ≥ 233. Example new LES cocktail lounge: 82 predicted reviews, no bounty.
- **Value:** baseline $0 a month vs v1 +$1,813 a month, all dollar inputs labelled ASSUMPTION.
- **Charts:** `outputs/baseline_vs_v1_error.png` (for slide 2), `outputs/coefficients.png`,
  `outputs/predicted_vs_actual.png`, `outputs/residuals.png`.
- **One script reproduces every number:** `bar_popularity_regression.py`. How to run: `README.md`.
- **Explanations for every number:** `BUSINESS.md` §7, so each of us can answer questions.

## What we don't have yet

- The three slides and the PDF.
- Abhiviraj's last name for the title slide.
- A check that our professor accepts the Yelp data, since it is not New York and not our app's own
  data. Yelp's terms allow course use; do not post the slides publicly.
- Any live data (crowds, lines, cover). No free public source exists; Scouts will collect it later.

## Priorities, in order

1. **Read `BUSINESS.md`** (10 minutes, everyone). Sections 3 and 7 explain every number.
2. **Title slide:** team name, four names, the repo link and script path:
   `https://github.com/RyanmMehta/party-engine/blob/nightlife/integration/coursework/group_model/bar_popularity_regression.py`
3. **Slide 1, data and model:** source, rows, period, 70/30 split; outcome, features, model
   family, adjusted R². Facts in `DATA.md` and `BUSINESS.md` §2.
4. **Slide 2, results and value:** paste `baseline_vs_v1_error.png`; one-sentence headline; the
   rule and the result for the example bar; baseline vs v1 dollars with assumptions labelled.
   Facts in `BUSINESS.md` §3 to §5.
5. **Review together:** each person explains one number out loud using `BUSINESS.md` §7.
6. **Export one PDF** and submit once for the team.

Style the professor asked for: short black text on a white background, written by us.

## Suggested split of work

| Person | Task |
|---|---|
| Ryan | Title slide, final check of numbers against `outputs/report.md`, submit |
| Teammate 2 | Slide 1 (data and model) |
| Teammate 3 | Slide 2 (chart and fit comparison) |
| Teammate 4 | Slide 2 (decision rule, example and value) |
