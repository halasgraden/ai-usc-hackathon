# Contracts

The code in `api/schemas.py` is the source of truth; this page is the map.
Change a schema there first and tell the other lanes, because the dashboard's
types are generated from `/openapi.json`.

## Endpoints

| Method and path | Auth | Request | Response | Status |
| --- | --- | --- | --- | --- |
| `GET /health` | none | | `{status, hall}` | done |
| `GET /menu/{date}` | none | | `DailyMenu` (`menu/models.py`) | done |
| `POST /plates` | device token | `PlateIn` | `PlateAccepted` | returns 501 |
| `POST /plates/image` | device token | multipart image, `meal` | `PlateAccepted` | not started; the phone `/capture` page uses it |
| `GET /dashboard/today?meal=&day=` | staff | | `TodayOut` | not started |
| `GET /dashboard/dishes?start=&end=&meal=` | staff | | `DishesOut` | not started |
| `GET /dashboard/dishes/{dish_name}` | staff | | `DishDetailOut` | not started |
| `GET /dashboard/trends?weeks=` | staff | | `TrendsOut` | not started |
| `GET /report/weekly?week_start=` | public, aggregated | | `ReportOut` | not started |

Review queue (`GET /review`, `POST /review/{id}`) is not specified yet.

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
