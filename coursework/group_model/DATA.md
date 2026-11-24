# DATA.md: dataset description and data dictionary

Reference document for the group model assignment (Nightlife Autopilot, venue noise-complaint regression).
Team name: `<TEAM NAME>` | Members: `<MEMBER 1>, <MEMBER 2>, <MEMBER 3>, ...`
Repo: https://github.com/RyanmMehta/manhattan-bars-ml | Script: `coursework/group_model/venue_noise_regression.py`

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
| NYC 311 | `noise_complaints` (outcome), `high_noise` (derived from it, excluded) |

## 2. Outcome and its distribution

**Outcome: `noise_complaints`** (count, integer, 0 or more).
Definition, from `src/build_db.py` (`build_venues`): the number of 311 complaints matched to the venue's address over the 3 years before the scrape (created 2023-10-21 to 2026-10-19), counted from complaints with location type Club/Bar/Restaurant in Manhattan. A complaint is matched by the same normalised address, or else by the nearest complaint address within 30 m with the same house number; venues with no match get 0. Venues that share an address get the same count. The 311 export is filtered by location type, not complaint type; 97.5% of the 31,190 complaints are "Noise - Commercial" (30,396), the rest are Drinking (780) and a few other types, so "noise complaints" is a close but not exact description.

Distribution of `noise_complaints` over all 5,704 rows (computed; percentiles by linear interpolation, as numpy/pandas do by default):

| Statistic | Value (complaints in 3 years) |
|---|---|
| Mean | 3.29 |
| Standard deviation | 14.78 |
| Median (50th percentile) | 0 |
| 75th percentile | 1 |
| 90th percentile | 6.7 |
| 95th percentile | 16 |
| 99th percentile | 62.0 |
| Max | 567 (Alice Restaurant, 126 W 13th St) |
| Share of venues with zero complaints | 67.53% (3,852 of 5,704) |

Venues at or above common cut-offs: 1 or more 1,852 (32.5%); 3 or more 976 (17.1%); 5 or more 712 (12.5%); 10 or more 447 (7.8%); 25 or more 182 (3.2%); 100 or more 22 (0.4%).

What this means for a linear regression: the count is heavily right-skewed (mean 3.29 but median 0; the top 1% of venues, 57 of them, account for 33.9% of all matched complaints). A straight OLS fit on the raw count is pulled hard by a few extreme venues. Fitting on `log1p(noise_complaints)` (mean 0.54, sd 0.98 on that scale) and converting predictions back with `expm1` is the usual remedy; errors should then be reported back in complaint units, as the script plans. Note that back-transformed log predictions estimate something closer to the typical (median-like) count than the mean, so they tend to run below the raw mean.

The mean baseline predicts the training-set mean (about 3.3 complaints) for every venue; the exact figure is in `outputs/metrics.json`.

## 3. Data dictionary (`PRAGMA table_info(venue_features)`)

Role legend: **Outcome** = what the model predicts. **Excluded as leakage** = never a feature, because it is computed from the outcome. **Identifier** = row key, not a feature. **Feature** = model input. Types are the sqlite column types. "% with value 1" or ranges are computed on all 5,704 rows.

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
| 23 | `noise_complaints` | INTEGER | **Outcome.** Count of 311 complaints matched to this venue's address over the 3 years to 2026-10-19 (0 to 567; mean 3.29, median 0, 67.5% zero) | NYC 311 | **Outcome** |
| 24 | `high_noise` | INTEGER | 1 if `noise_complaints >= 3`, else 0 (976 rows are 1, 17.11%). Threshold `3` set in `build_features` | NYC 311 (derived from the outcome) | **Excluded as leakage** |

Why `high_noise` is excluded: it is computed directly from `noise_complaints` (`noise_complaints >= 3`), so as a feature it would hand the regression most of the answer and inflate R² meaninglessly. The model is also not given `license_id` (a unique key carries no pattern). The categorical features (`venue_type`, `liquor_tier`, `neighborhood`, `cuisine`, `latest_grade`, `osm_amenity`) must be turned into dummy (0/1) columns for OLS, with one level per column dropped as the reference.

## 4. Data honesty

This table is real public data, not synthetic. Every row is one actual NY State licensed venue, and the features and the outcome come from the four open-data sources in section 1.1, scraped and downloaded in October 2026 (scrape date 2026-10-20) by `src/scrape.py` and combined by `src/build_db.py`. Nothing in `venue_features` was simulated, resampled or generated. The same database file also contains tables named `synthetic_cart_10000`, `synthetic_copula_10000` and `synthetic_independent_10000` (made by `src/simulate.py` for other course work); this model does not read them, and neither the 5,704-row table nor the metrics we report depend on them. What is *not* perfect about the real data is the matching: DOHMH, OSM and 311 are joined by address and name matching, so some links are wrong or missing (see section 5), and 311 complaint counts measure complaining, not actual noise.

## 5. Things the model session and the slides should know

1. **Shared complaint counts and the 70/30 split.** Venues that share a street address share one 311 count: 324 addresses host 769 venues (13.5% of rows). A random 70/30 split can put two venues from the same address in train and test with the same outcome, which slightly flatters test R². Keep the 70/30 split the assignment asks for, but state this in one line (or optionally split by address as a robustness check).
2. **Heavy missingness is encoded as sentinels, not NULL.** No column has SQL NULLs. Missing inspection data shows up as `mean_score`/`max_score` = -1 (1,031 rows; 716 unmatched to DOHMH plus 315 matched with no score), `n_inspections` = 0 (1,002), `latest_grade` = `None` (1,002), `cuisine` = `Not matched` (716). Missing OSM data shows up as `osm_amenity` = `not_listed` (5,039 of 5,704, 88%) and 0 in the three OSM binary columns. **For a linear regression the -1 sentinel is a problem:** OLS treats -1 as a real score just below 0, so a single slope gets fitted across "no record" and real scores. Either add a "has score" indicator and set the sentinel to 0 (or to the mean of real scores), or rely on `dohmh_matched` plus a separate indicator for the 315 matched-but-unscored rows. Do not take means over -1.
3. **OSM coverage is thin and informative.** Only 665 rows (11.7%) have an OSM match, but those rows have far higher counts: mean `noise_complaints` is 14.6 for `osm_amenity = nightclub`, 8.76 for `bar`, 3.95 for `pub` and 2.70 for `not_listed`; 9.46 when `osm_closes_late = 1` vs 2.96 otherwise. Expect large positive coefficients on these. Also expect `venue_type`, `neighborhood`, `liquor_tier` and location to matter: Bar/Tavern mean 4.84 vs Club 0.52 and Venue/Theatre 0.10; full liquor 4.02 vs beer/wine 1.29; Washington Heights/Inwood 9.35, LES/Chinatown 8.51, Harlem 6.65, East Village 4.83 vs Midtown West 1.73 and Midtown East/Murray Hill 1.46.
4. **`years_licensed` is the age of the current license record, not of the venue.** The SLA active-license list only holds records issued from 2022-05-23, so the range is 0 to 2.93 years and 3,078 rows are under 1 year. Do not describe it as "how long the bar has existed".
5. **19 rows have coordinates outside Manhattan** (latitude up to 43.05, longitude down to -76.07; bad SLA coordinates). None of the 19 has 3 or more complaints. For a linear model `lat`/`lon` enter as straight slopes, so these outliers can tilt the location coefficients; consider dropping or clipping them, and note that a straight north-south slope is a crude stand-in for neighborhood (which is already a feature).
6. **README column count is stale:** it says 24 columns and 5,737 venues for the venue table; `venue_features` has 25 columns and 5,704 rows (33 venues dropped for missing coordinates or issue date). Use 5,704 and 25.
7. **Skew and outliers dominate the error metrics.** With median 0 and a max of 567, RMSE on raw counts is driven by a handful of venues and will look large next to MAE; the mean baseline's RMSE is roughly the standard deviation (about 14.8). Expect R² to be modest. MAE and RMSE in complaint units, compared with the mean baseline, carry the story (see `BUSINESS.md`, "How each number is calculated").
8. **The app's current complaint bands are different from the model's badge threshold.** The backend shows 311 risk as low/medium/high from raw counts (under 25, 25 to 99, 100 or more; a placeholder), whereas the model script sets its own badge threshold as an ASSUMPTION. Do not mix the two on a slide.
