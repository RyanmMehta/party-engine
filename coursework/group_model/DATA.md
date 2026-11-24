# DATA.md: dataset description and data dictionary

Reference document for the group model assignment (Nightlife Autopilot, venue noise tree).
Team name: `<TEAM NAME>` | Members: `<MEMBER 1>, <MEMBER 2>, <MEMBER 3>, ...`
Repo: https://github.com/RyanmMehta/manhattan-bars-ml | Script: `coursework/group_model/venue_noise_tree.py`

Everything below was read from `src/scrape.py`, `src/build_db.py`, the README ("Public-data pipeline") and from read-only queries on `data/manhattan_bars.db`. Numbers marked "computed" were computed with sqlite on the committed database.

## 1. Dataset description

| Item | Value |
|---|---|
| Table used | `venue_features` in `data/manhattan_bars.db` |
| One row per | licensed on-premises drinking venue in Manhattan (one NY State Liquor Authority license, after duplicate renewals are collapsed to the oldest) |
| Rows | 5,704 (computed; 5,704 distinct `license_id`) |
| Columns | 25 (the README says 24: it does not count `high_noise`) |
| Period covered | Licenses and inspections as they stood on the scrape date, 2026-10-20 (`data/raw/meta.json`: `scraped_at = 2026-10-20T17:36:46`). The outcome counts 311 complaints created from 2023-10-21 to 2026-10-19 (3 years; computed from `complaints_311.created_date`). |
| How collected | Downloaded from public open-data portals in October 2026 (scrape date 2026-10-20) with `src/scrape.py`, then joined by `src/build_db.py`. No API keys. Not survey data, not synthetic. |
| Scraped or synthetic? | Scraped/downloaded from public open-data portals: **real data**. See section 4. |

### 1.1 Source datasets

| Source | Official dataset | Portal / id | Raw rows (Oct 2026) | What it gives |
|---|---|---|---|---|
| NY State Liquor Authority (SLA), active licenses | "Liquor Authority Current List of Active Licenses" (premises county = New York) | data.ny.gov, dataset `9s3h-dpkz` | 9,917 Manhattan licenses | every licensed venue: name, address, coordinates, license class, issue date |
| NYC Department of Health and Mental Hygiene (DOHMH) | "DOHMH New York City Restaurant Inspection Results" (borough = Manhattan) | data.cityofnewyork.us, dataset `43nn-pn8j` | 109,845 rows / 12,603 establishments | inspection scores, grades, violations, cuisine |
| NYC 311 Service Requests | "311 Service Requests from 2010 to Present", location type `Club/Bar/Restaurant`, borough Manhattan, last 3 years | data.cityofnewyork.us, dataset `erm2-nwe9` | 31,190 complaints | complaint with address and coordinates (source of the outcome) |
| OpenStreetMap | amenity = bar, pub, nightclub, biergarten inside a Manhattan bounding box, via the Overpass API | overpass-api.de (and two mirrors) | 1,523 venues | opening hours, outdoor seating, website |

The official portal titles above are the portal names for those dataset ids as we know them; the repo itself only records the ids and short descriptions. Check the title on the portal page before putting it on a slide. Yelp and Google Places were deliberately not used (they need keys and forbid bulk scraping).

### 1.2 How rows were joined (`src/build_db.py`)

1. Start from SLA licenses whose class is an on-premises drinking class (bar/tavern, restaurant with liquor, club, nightlife, hotel, venue/theatre). 5,737 venues result.
2. **DOHMH:** same normalised address and zip with the best name match; otherwise within 60 m with a strong name match. 87% of venues matched (87.0% of 5,737, computed: 4,561 by address, 431 by distance plus name; 87.4% of the 5,704 rows in `venue_features`).
3. **OSM:** within 60 m with a name match. 70% of the OSM bars inside Manhattan matched (README), which is only 11.6% of all venues because most venues are restaurants that OSM does not list as bars.
4. **311:** same normalised address; otherwise the nearest complaint address within 30 m with the same house number. Venues that share an address share its complaint count. 32% of venues have at least one complaint (32.3% of 5,737; 32.5% of the 5,704 rows, computed). 1,759 venues matched a complaint address directly, 93 by the 30 m same-house-number rule; the rest have 0 complaints.
5. "Additional bar" licenses become the `n_additional_bars` feature of their parent venue, not separate rows. Duplicate licenses for the same venue are collapsed to the oldest.
6. 33 of the 5,737 venues are dropped when `venue_features` is built because lat, lon or `years_licensed` is missing: 5,704 remain.

### 1.3 Which columns come from which source

| Source | Columns |
|---|---|
| SLA licenses | `license_id`, `venue_type`, `liquor_tier`, `neighborhood` (derived from license zip), `lat`, `lon`, `years_licensed`, `n_additional_bars`, `density_100m`, `density_250m` |
| DOHMH inspections | `dohmh_matched`, `cuisine`, `latest_grade`, `has_pest_violation`, `n_inspections`, `mean_score`, `max_score`, `n_violations`, `n_critical_violations` |
| OpenStreetMap | `osm_amenity`, `osm_closes_late`, `osm_outdoor_seating`, `osm_has_website` |
| NYC 311 | `noise_complaints`, `high_noise` |

## 2. Outcome and class balance

**Outcome: `high_noise`** (binary, 1 = high noise).
Definition, from `src/build_db.py` (`build_features`, `threshold = 3`): `high_noise = 1` if `noise_complaints >= 3`, else 0. `noise_complaints` is the number of 311 complaints at the venue's address over the 3 years before the scrape (2023-10-21 to 2026-10-19), counted from complaints with location type Club/Bar/Restaurant in Manhattan. The 311 export is filtered by location type, not complaint type; 97.5% of the 31,190 complaints are "Noise - Commercial" (30,396); the rest are Drinking (780) and a few other types, so "noise complaints" is a close but not exact description.

Class balance of `high_noise` over all 5,704 rows (computed):

| `high_noise` | Rows | Share |
|---|---|---|
| 0 (not high noise) | 4,728 | 82.89% |
| 1 (high noise) | 976 | 17.11% |

The most-common-class baseline predicts 0 for every venue and is right 82.89% of the time on the full table (the test-set figure will differ slightly; use the one in `outputs/metrics.json`). It catches none of the 976 high-noise venues.

`noise_complaints` itself: mean 3.29, max 567, 1,852 venues with at least one complaint, 3,852 venues with zero.

## 3. Data dictionary (`PRAGMA table_info(venue_features)`)

Role legend: **Outcome** = what the model predicts. **Excluded as leakage** = never a feature, because it is the outcome before thresholding. **Identifier** = row key, not a feature. **Feature** = model input. Types are the sqlite column types. "% with value 1" or ranges are computed on all 5,704 rows.

| # | Column | Type | Meaning and derivation | Source | Role |
|---|---|---|---|---|---|
| 0 | `license_id` | TEXT | SLA license permit id (for example `0138-19-112974`); unique per row | SLA | **Identifier** |
| 1 | `venue_type` | TEXT | Venue type from the license class: Restaurant (3,959), Bar/Tavern (1,301), Hotel (280), Club (86), Venue/Theatre (50), Nightlife (28) | SLA | Feature |
| 2 | `liquor_tier` | TEXT | `full_liquor` (4,175) or `beer_wine` (1,529), from the license class | SLA | Feature |
| 3 | `neighborhood` | TEXT | One of 14 named neighborhoods mapped from the license zip code, else `Other Manhattan` (64 rows) | SLA zip, mapping in `build_db.py` | Feature |
| 4 | `cuisine` | TEXT | DOHMH cuisine, top 12 kept, else `Other`; `Not matched` if no DOHMH match (716 rows) | DOHMH | Feature |
| 5 | `latest_grade` | TEXT | Most recent letter grade: A (3,772), B (285), C (74), N (342), Z (229); `None` (1,002) if no grade. N and Z are DOHMH "not yet graded" and "grade pending" codes | DOHMH | Feature |
| 6 | `osm_amenity` | TEXT | OSM tag `bar` (496), `pub` (154), `nightclub` (15); `not_listed` (5,039) if no OSM match | OSM | Feature |
| 7 | `dohmh_matched` | INTEGER | 1 if the venue was matched to a DOHMH establishment (4,988 rows) | DOHMH join | Feature |
| 8 | `has_pest_violation` | INTEGER | 1 if any inspection ever cited a pest code (04K, 04L, 04M, 04N, 08A); 0 if none or no match (2,609 rows are 1) | DOHMH | Feature |
| 9 | `osm_closes_late` | INTEGER | 1 if the OSM opening-hours text closes at or after 2 am or is 24/7 (290 rows). 0 means "no" **or** "no OSM hours" (not separable) | OSM | Feature |
| 10 | `osm_outdoor_seating` | INTEGER | 1 if OSM `outdoor_seating=yes` (63 rows); 0 also when unmatched | OSM | Feature |
| 11 | `osm_has_website` | INTEGER | 1 if the matched OSM venue lists a website (430 rows); 0 also when unmatched | OSM | Feature |
| 12 | `lat` | REAL | Latitude of the licensed premises | SLA | Feature (see odd columns) |
| 13 | `lon` | REAL | Longitude of the licensed premises | SLA | Feature (see odd columns) |
| 14 | `years_licensed` | REAL | (scrape date − original license issue date) / 365.25, years, 2 decimals. Range 0.00 to 2.93, mean 0.93 | SLA | Feature |
| 15 | `n_additional_bars` | INTEGER | Count of "additional bar" licenses attached to this license (0 to 50; 715 rows above 0) | SLA | Feature |
| 16 | `density_100m` | REAL | Number of other on-premises venues within 100 m (0 to 64) | derived from SLA venues | Feature |
| 17 | `density_250m` | REAL | Number of other on-premises venues within 250 m (0 to 143) | derived from SLA venues | Feature |
| 18 | `n_inspections` | REAL | Distinct DOHMH inspection visits; 0 if no match (0 to 13; 1,002 rows are 0) | DOHMH | Feature |
| 19 | `mean_score` | REAL | Mean DOHMH inspection score (higher is worse); **-1 = no score** (1,031 rows) | DOHMH | Feature (sentinel) |
| 20 | `max_score` | REAL | Worst DOHMH inspection score; **-1 = no score** (1,031 rows) | DOHMH | Feature (sentinel) |
| 21 | `n_violations` | REAL | Violation rows across all inspections; 0 if no match (0 to 66) | DOHMH | Feature |
| 22 | `n_critical_violations` | REAL | Violation rows flagged Critical (0 to 38) | DOHMH | Feature |
| 23 | `noise_complaints` | INTEGER | Count of 311 complaints at this address over the 3 years to 2026-10-19 (0 to 567) | NYC 311 | **Excluded as leakage** |
| 24 | `high_noise` | INTEGER | **Outcome.** 1 if `noise_complaints >= 3`, else 0. Class balance: 17.11% ones (976 of 5,704) | NYC 311 (derived) | **Outcome** |

Why `noise_complaints` is excluded: `high_noise` is `noise_complaints >= 3`, so keeping it would let the tree read the answer directly (accuracy near 100% and meaningless). The tree is also not given `license_id` (a unique key carries no pattern).

## 4. Data honesty

This table is real public data, not synthetic. Every row is one actual NY State licensed venue, and the features and the outcome come from the four open-data sources in section 1.1, scraped and downloaded in October 2026 (scrape date 2026-10-20) by `src/scrape.py` and combined by `src/build_db.py`. Nothing in `venue_features` was simulated, resampled or generated. The same database file also contains tables named `synthetic_cart_10000`, `synthetic_copula_10000` and `synthetic_independent_10000` (made by `src/simulate.py` for other course work); this model does not read them, and neither the 5,704-row table nor the metrics we report depend on them. What is *not* perfect about the real data is the matching: DOHMH, OSM and 311 are joined by address and name matching, so some links are wrong or missing (see section 5), and 311 complaint counts measure complaining, not actual noise.

## 5. Things the model session and the slides should know

1. **Shared complaint counts and the 70/30 split.** Venues that share a street address share one 311 count: 324 addresses host 769 venues (13.5% of rows). A random 70/30 split can put two venues from the same address in train and test with the same label, which slightly flatters test accuracy. The assignment asks for a stratified 70/30 split, so keep that, but state this in one line (or optionally split by address as a robustness check).
2. **Heavy missingness is encoded as sentinels, not NULL.** No column has SQL NULLs. Missing inspection data shows up as `mean_score`/`max_score` = -1 (1,031 rows; 716 unmatched to DOHMH plus 315 matched with no score), `n_inspections` = 0 (1,002), `latest_grade` = `None` (1,002), `cuisine` = `Not matched` (716). Missing OSM data shows up as `osm_amenity` = `not_listed` (5,039 of 5,704, 88%) and 0 in the three OSM binary columns. A decision tree handles -1 fine (it just splits on it); do not impute or take means over -1.
3. **OSM coverage is thin and informative.** Only 665 rows (11.7%) have an OSM match, but those rows have a much higher `high_noise` rate: `osm_amenity = bar` 40.1%, `osm_closes_late = 1` 41.7%, against 14.7% for `not_listed`. Expect these to be top tree splits. Also expect `venue_type`, `neighborhood`, `liquor_tier` and location to matter (Bar/Tavern 25.4% high-noise vs Club 4.7%, Hotel 13.2%; LES/Chinatown 33.8%, Harlem 34.8%, Washington Heights/Inwood 34.5%, East Village 27.4% vs Midtown East/Murray Hill 8.1%).
4. **`years_licensed` is the age of the current license record, not of the venue.** The SLA active-license list only holds records issued from 2022-05-23, so the range is 0 to 2.93 years and 3,078 rows are under 1 year. Do not describe it as "how long the bar has existed".
5. **19 rows have coordinates outside Manhattan** (latitude up to 43.05, longitude down to -76.07; bad SLA coordinates). None of the 19 is `high_noise`. Harmless for a tree, but they stretch the `lat`/`lon` ranges and would be wrong on a map.
6. **README column count is stale:** it says 24 columns and 5,737 venues for the venue table; `venue_features` has 25 columns and 5,704 rows (33 venues dropped for missing coordinates or issue date). Use 5,704 and 25.
7. **Class imbalance:** 17.11% positives. Plain accuracy of the baseline is already 82.89%, so accuracy alone barely separates baseline and tree; precision and recall on `high_noise = 1` carry the story (see `BUSINESS.md`, "How each number is calculated").
8. **The app's current complaint bands are different from this outcome.** The backend shows 311 risk as low/medium/high from raw counts (under 25, 25 to 99, 100 or more; a placeholder), whereas `high_noise` is 3 or more complaints. Do not mix the two thresholds on a slide.
