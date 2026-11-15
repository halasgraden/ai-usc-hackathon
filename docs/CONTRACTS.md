# Contracts

The code in `api/schemas.py` is the source of truth; this page is the map.
Change a schema there first and tell the other lanes, because the dashboard's
types are generated from `/openapi.json`.

## Endpoints

| Method and path | Auth | Request | Response | Status |
| --- | --- | --- | --- | --- |
| `GET /health` | none | | `{status, hall}` | done |
| `GET /menu/{date}` | none | | `DailyMenu` (`menu/models.py`) | done |
| `POST /plates` | device token | `PlateIn` | `PlateAccepted` | done; idempotent on `plate_id` |
| `POST /plates/image` | device token | multipart `image`, `meal` | `PlateImageAccepted` (counts + measured `regions`) | done; used by the rig loop (`python -m capture`) and the phone page; 422 with a reason when the photo can't be measured (nothing stored), 503 until the model is installed |
| `GET /dashboard/today?meal=&day=` | staff | | `TodayOut` (now with `assumptions`) | done; `day` defaults to today in hall time |
| `GET /dashboard/dishes?start=&end=&meal=` | staff | | `DishesOut` | done; default last 7 days, both ends included |
| `GET /dashboard/dishes/{dish_name}?start=&end=&meal=` | staff | | `DishDetailOut` | done; default last 28 days; 404 if no plates |
| `GET /dashboard/trends?weeks=` | staff | | `TrendsOut` | done; 1-52 weeks (default 8); empty weeks are left out, not zero |
| `GET /report/weekly?week_start=` | public, aggregated | | `ReportOut` | done; any date maps to its Monday; default last week |

Auth: device = `Authorization: Bearer $LEFTOVER_DEVICE_TOKEN`; staff =
`Authorization: Bearer $LEFTOVER_STAFF_TOKEN`. An unset token refuses every request.
Keep the staff token out of `NEXT_PUBLIC_*` variables: call staff endpoints from Next
server routes, or accept the exposure for the demo only.

Review queue (`GET /review`, `POST /review/{id}`) is not specified yet; it should show
the matcher's `MatchDetail.reason` and candidate probabilities (`menu/matcher.py`).

## Rules every lane follows

1. **No identifiers.** No student, card, device-owner or face field, anywhere.
2. **Seeded data is labelled.** Any response built from generated demo data sets
   `data_source: "seeded"`, and the UI shows a visible "simulated data" tag.
3. **Cost and CO2e are assumptions.** They come from editable factors, appear
   with the `assumptions` map, and are never shown as measured.
4. **Percent left is an index, not a share of what the student took.** Halls are
   self-serve; it is leftover volume against one reference portion per dish.
5. **No invented accuracy.** Numbers about model quality come only from `eval/results/`.

## Who edits what (parallel lanes)

| Lane | Owns | Must not edit |
| --- | --- | --- |
| Dashboard | `dashboard/` | everything else |
| Backend | `api/` (except `schemas.py`), `analytics/` | `core/`, `api/schemas.py` without asking |
| Vision | `capture/`, `vision/`, `labeling/`, `augment/`, `depth/` | `core/`, `api/` |
| Match and eval | `menu/` (matcher), `eval/`, `portion/` | `core/` |
| Integration (lead) | `core/`, `api/schemas.py`, `tests/` integration, `README.md`, CI | |

## Dashboard additions (November 15)

Existing field meanings remain unchanged. In particular, `waste_rate_percent`
and `mean_percent_left` remain reference-portion indices.

- `Headline`, `DishRow`, and `TrendPoint` now include `kg_low`, `kg_high`, and
  nullable `serving_comparison`. The latter exposes simulated `served_kg`,
  `servings`, `returned_percent`, its lower/upper bounds, source and basis.
- `DishRow` adds `mean_grams` and bounds, `previous_total_kg`, and
  `previous_mean_percent_left`. The existing `trend` still compares the portion
  index; the UI's change-in-kg column uses `previous_total_kg` instead.
- `/dashboard/dishes` accepts `station` and case-insensitive `search`, and returns
  the entire matched/served menu, filtered headline, distinct plate count,
  unmatched mass, cost assumptions, and available stations/meals.
- `/dashboard/trends` additionally accepts `start`, `end`, `meal`, `station`,
  `search`, and returns bulk `dishes[].weeks`, explicit bounds and partial-week
  flags. Omitted weeks mean missing observations, never zero waste.
- `/report/weekly` additionally accepts `meal`, `station`, `search`, and returns
  all `rows` as well as the existing top five, distinct plate count, unmatched
  mass, week end, partial status, and the same enriched headline.
- `ServingRecord` stores seeded day/meal/dish totals with a complete-capture
  flag. The seed creates serving and return quantities together. Reseeding
  replaces seeded serving rows alongside seeded plates and preserves live data.
- `serving_comparison` is absent for live or incompatible observation cohorts.
  Unknown-food mass is included in the overall kg total but never assigned to
  a dish or included in the identified-return percentage numerator.

The Next.js dashboard makes allowlisted, read-only server requests with the
staff Bearer token. The browser does not hold that token. The local demo proxy
has no user authentication; keep it bound to loopback until institutional auth
is implemented. See `dashboard/README.md` for generation and integration steps.
