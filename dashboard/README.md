# Dining operations dashboard

Two tabs: **Menu performance** and **Trends**. All 37 seeded dishes are discoverable. Select a chart mark or table row to inspect evidence without navigating away. Export filtered CSV, a printable weekly report, or an email preview from the toolbar.

## Run the complete demo

Tested with Node 24 and Python 3.13. Use Python 3.11+ and Node 20.9+.

From the repository root, in terminal 1:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m api.seed
LEFTOVER_STAFF_TOKEN=local-demo-staff LEFTOVER_DEVICE_TOKEN=local-demo-device python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In terminal 2:

```sh
cd dashboard
npm ci
cp .env.example .env.local
npm run dev
```

Open http://127.0.0.1:3000. The example token is for this loopback-only demo. Both servers must use the same staff token. Python does not automatically load `.env`; Next loads `.env.local`. No GPU, model weights, credentials, external service, or CV inference run is needed to browse seeded data.

`python -m api.seed` replaces seeded plates and servings for the configured hall; it preserves live records. Default: four weeks ending today in America/Los_Angeles. For reproducible judging evidence use `python -m api.seed --end 2026-11-15`, then visit `/?start=2026-10-19&end=2026-11-15&tab=menu`.

The frontend always calls FastAPI. API failures are displayed, never replaced with browser fixtures. The serving data, historical menu recurrence, plate counts, and trends are fictional and visibly labeled. Dish names originate from the Village November 14 menu. The seed's downward trend is invented, not a measured intervention outcome.

## Judge walkthrough (under three minutes)

1. Select **4 weeks**. Explore all 37 dots and rows; the largest masses emerge from the sorted table. Switch to **Compare with servings** and inspect uncertainty.
2. Search **Rice**. Select **Rice Pilaf**. Inspect grams, return share, and the separately explained reference-portion index.
3. Optionally record a dated portion-change note. It persists only in this browser, annotates trends, and never changes measurements.
4. Close the panel, clear the search, and switch to **Trends**. Compare mass with return share and inspect the full dish-by-week matrix. Missing periods are not filled with zero.
5. Export a CSV or preview the last completed week's report. The email preview copies text; it sends nothing.

## Architecture and sources of truth

Browser → allowlisted Next.js GET routes → FastAPI (server-only Bearer token) → SQLite plates, food regions and serving records. `api/metrics.py` performs the shared arithmetic used by the dashboard, detail, weekly series and report. No personal identifiers or plate images are returned by these endpoints.

The browser may write optional intervention notes to local storage. These are not shared, uploaded, or integrated with CBORD. Use **Data & methods → Reset local intervention notes** to clear them. The seed reset is a CLI operation, not an accidental browser action.

`src/lib/api.generated.ts` is generated, not hand-maintained. From the repository root:

```sh
curl -fsS http://localhost:8000/openapi.json -o dashboard/openapi.json
npm --prefix dashboard run generate:types
```

Offline alternative: `.venv/bin/python dashboard/scripts/export_openapi.py`, then the same generation command. The helper exports the actual FastAPI app schema without loading model weights or creating a database. Commit both the OpenAPI snapshot and generated TypeScript after schema changes.

## Metrics

| Metric | Definition and limit |
| --- | --- |
| Returned kg | Sum of all food-region grams / 1000, including unmatched food in hall totals. |
| Estimate range | Sum of individual lower/upper density-based mass estimates. Conservative range, not a statistical confidence interval or validated model accuracy. |
| Observed plates | Distinct random plate IDs. Hall totals include clean plates. Dish/station/search counts include only corresponding detections; counts overlap across dishes. |
| Grams per observed plate | Dish mass / distinct plates where that dish was detected. Not grams per diner and not a popularity measure. |
| Reference-portion index | Mean leftover volume against one reference portion per matched region. Existing `percent_left` / `waste_rate_percent` fields retain this meaning and can exceed 100. |
| Identified return share | Matched leftover mass / simulated served mass for the same hall, dish, dates and meal population. Unmatched food is excluded. Aggregate shares divide summed masses; they do not average dish percentages. |
| Serving comparison availability | Requires complete seeded serving records covering every matched observation group. Live observations, missing records, incomplete capture or zero served mass suppress the comparison. No live numerator is divided by a simulated denominator. |
| Estimated ingredient cost | Returned kg × configured assumed USD/kg factor. Not realized savings or USC procurement data. |
| Change in kg | Difference from the immediately preceding period of equal length, with the same filters. No prior observations means no baseline. Current-day and partial-week values are not extrapolated. |

The simulator chooses initial portions before leftovers, caps leftovers to physical serving limits, and records servings even for clean plates and unidentified leftovers. No student-level serving link is stored. `ServingRecord` is an additive table created on first database use; existing plate tables are unchanged. Reseed an older demo database to populate serving comparisons.

## Accessibility and responsive behavior

Local Source Sans 3; USC cardinal/gold only in branding; blue/teal/purple and supplementary data colors. Every chart has a numeric table with bounds. Keyboard users can navigate tabs with arrows, focus chart marks, open details with Enter/Space, and close dialogs with Escape. Native dialogs contain focus and restore it. Table sort state is announced. Inputs have explicit labels, and loading/error states are readable.

Layouts cover 1280 px desktop, 768 px tablet, and narrow reflow. Wide tables scroll within their own container. Coarse-pointer controls have 44 px targets. Reduced motion is honored. The accessible HTML report is primary: browser-generated PDFs are **not guaranteed tagged/PDF-UA** (the verified Chrome output is untagged).

## Verification

With both servers running and the frozen seed installed:

```sh
npm run typecheck
npm run lint
npm run build
npm test
```

Playwright uses installed Chrome by default. Set `PLAYWRIGHT_CHANNEL=chromium` after `npx playwright install chromium` for bundled Chromium. The tests cover all 37 dishes, linked detail, notes, URL filters, keyboard tabs/focus, CSV, report/email preview, failed requests, report loading/error accessibility, narrow reflow, axe checks, and Letter/A4 output. Recreate the fixed seed first if your database has changed.

Backend checks, from root:

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check api tests
.venv/bin/mypy api/metrics.py api/dashboard.py api/seed.py api/models.py api/schemas.py
```

See [verification notes](../docs/DASHBOARD_VERIFICATION.md) for actual results and limits.

## Institutional integration: implemented vs proposed

**Implemented:** read-only consumption of the team's authenticated FastAPI endpoints; server-side token handling; full-menu analysis; CSV compatible with Excel; browser PDF export; email preview. Launch scripts bind to loopback. Staff tokens never use `NEXT_PUBLIC_*`.

**Before production hosting:** put institutional staff authentication/authorization in front of the Next.js proxy, use HTTPS, protect staff report access according to institutional policy, configure the API URL, and replace local notes with an authenticated persistence service if shared notes are needed. The current aggregate `/report/weekly` endpoint remains public by the existing backend contract.

**Proposed vendor adapter:** ingest institution-approved exports or APIs with source system, stable recipe/item ID, hall, station, meal, date, initial served mass/quantity and unit, cost factor/version, and capture coverage. Maintain a reviewed source-ID-to-menu mapping. Reject unknown units and incompatible populations. Do not treat production forecasts, meal swipes, or kitchen output as the mass actually served to the captured cohort. Start with read-only imports; no automated recipe, inventory or procurement changes.

**Not verified:** USC's installed CBORD product/version, administrator screen configuration, licensed API/export access, and permitted embedding/SSO interface. No vendor API, background import, iframe integration or email delivery is claimed or simulated as connected.

### Workflow research

- [USC Chef de Cuisine, Campus Dining](https://usc.wd5.myworkdayjobs.com/en-US/ExternalUSCCareers/job/Chef-de-Cuisine--Campus-Dining_REQ20174140): CBORD for cost, inventory, waste and forecasting; Excel for recipe costing. This supports meal/station/dish controls and CSV export.
- [CBORD higher-education NetMenu guide](https://resources.cbord.com/hubfs/Collateral/HE-Collateral/HE-NetMenu-SellSheet.pdf): station-level menus, production planning, usage/waste tracking, portion analysis and summary/detail reports. This establishes vendor workflow conventions, not proof that USC uses this specific product configuration.
- [USC Hospitality sustainability](https://hospitality.usc.edu/sustainability/): existing back-of-house forecasting and prevention. Our incremental contribution is anonymous dish-return evidence.
- [USC colors](https://identity.usc.edu/identity/color/) and [typography](https://identity.usc.edu/identity/typography/): cardinal/gold branding and Source Sans 3 digital substitute.

Research accessed November 15, 2026. Proposed pilot success measures: returned kg per meal, matched return share with validated serving coverage, manager review time, and actual workflow adoption. No environmental or procurement savings have been measured by this demo.
