# Dashboard verification — November 15, 2026

## Reproducible setup

Python 3.13, Node 24.13, Next.js 16.4.6, system Chrome via Playwright.
Seed: `python -m api.seed --end 2026-11-15`, four weeks from October 19.
Result: **15,158 simulated plates and 37 distinct menu dishes**. All history,
serving quantities, and plate measurements in this demo are fictional.

Start both servers as documented in [dashboard/README.md](../dashboard/README.md).
Browser tests use this exact date range and expect the seed to be present.

## Evidence

- Full backend suite: **352 passed, 1 skipped**. The skip is the existing YuNet
  face-model check; its weights are not installed in this test environment.
- New full-menu metric tests verify range arithmetic, weighted serving share,
  complete dish coverage, clean plates, unknown-food reconciliation, filter
  consistency, report equality, mixed live/seeded rejection, incomplete serving
  records, partial periods, and empty periods.
- TypeScript typecheck, ESLint, and Next production build passed.
- Ruff checks pass for the repository. Modified backend files pass mypy with
  `--follow-imports=silent`. A full import-following check reports an existing
  `no-any-return` in `depth/plane.py:41` with this environment's NumPy types;
  the dashboard does not alter that measurement code.
- Browser suite: **5 passed**. Coverage: full menu, serving mode, numeric chart alternatives, detail,
  local notes, Escape/focus restoration, keyboard tabs, URL filters, CSV contents,
  weekly report/email preview, and explicit unavailable-API behavior.
- axe WCAG 2 A/AA and 2.1 AA checks cover desktop/tablet menu, trends, detail,
  report success and error states. Narrow reflow checks use 640 px (the layout
  equivalent of 200% zoom at 1280 px) and 320 px with reduced motion.
- Screenshots captured and visually inspected at 1280×1000 and 768×1024.
  Two-page Letter and A4 PDFs rendered with Poppler and inspected for table wrapping,
  repeated headings, complete rows and unclipped methods text.
- Independent Impeccable finish review found a missing accessible name for the
  report dialog during loading/error; fixed with a stable accessible label and
  a regression test. This is code-first work, with no reference-image fidelity
  claim. The Impeccable context engine was unavailable; no detector was run.

Automated checks and source review do not constitute a full screen-reader/user
study or a formal WCAG certification. Keyboard behavior is exercised by the
browser tests. The HTML report is the accessible alternative; the tested Chrome
PDF output is untagged, so PDF-UA conformance is not claimed.

## Judging evidence map

| Criterion | Inspectable evidence |
| --- | --- |
| Campus impact (30%) | Sourced USC staff workflow, full-menu evidence and proposed pilot measures in dashboard README. No measured savings claimed. |
| Technical and AI execution (25%) | Existing model/data cards and evaluation files, API integration, metric tests, uncertainty and provenance. Dashboard quantities remain separately labeled simulated. |
| Feasibility (25%) | Local reproducible seed/run path, read-only API boundary, Excel CSV, report export, documented CBORD/SSO dependencies. |
| UX/design (20%) | Two tabs, all37 dishes, responsive screenshots, keyboard tests, chart tables and uncertainty. |

Criteria source: [AI@USC rules](https://ai-usc.devpost.com/rules).

## Submission handoff

The dashboard is runnable locally. Public hosting, repository visibility, the
3-minute demo recording, slide deck and Devpost submission have not been
performed by this implementation. Institutional staff authentication is required
before exposing the local demo proxy. Email is a preview, not a delivery service.
