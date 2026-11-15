# Dining operations interface research

Reviewed November 15, 2026.

- USC chef responsibilities explicitly include CBORD forecasting, inventory, waste and food cost, plus Excel recipe costing: https://usc.wd5.myworkdayjobs.com/en-US/ExternalUSCCareers/job/Chef-de-Cuisine--Campus-Dining_REQ20174140
- CBORD NetMenu campus guide describes station-level menus, production forecasts, purchasing, inventory, summary/detail reports: https://resources.cbord.com/hubfs/Collateral/HE-Collateral/HE-NetMenu-SellSheet.pdf
- GFS NetMenu training organizes tasks into recipe management, menu planning and production. Linked written guides could not be retrieved in this session: https://sites.google.com/public.gfs.com/cmmcanada/english/training-modules/menu
- Epicure Digital shows an older reconstructed CBORD MenuDisplay Interface with location, dated meal context and a dense item grid. This is an integration vendor illustration, not USC’s installation or proof of current NetMenu styling: https://epicuredigital.com/cbord.html

## Applied design choices
Compact dark operations header; location/date/meal/station toolbar; two module tabs; restrained blue data marks; dense records with units and export actions. Estimate whiskers are centered over a saturated blue fill with dark end caps and no white outline. These are our design choices informed by workflow evidence, not a reproduction of an authenticated USC screen.

## Integration boundary
No CBORD login, data feed, purchase-order writeback or embedding connection is implemented. USC’s configured product/version and screen layout remain unverified. The companion module uses its own seeded API and exports CSV for an Excel workflow. A production adapter requires approved vendor identifiers, serving records and access arrangements.

## Purchasing recommendations and STARS
Email actions follow chef forecast → recipe yield → stock/open orders → vendor case packs and delivery → purchasing evidence. A 5% service trial is a proposal, not an inferred optimal order reduction. STARS 3.0 OP 7 covers qualifying food/beverage spend and social-impact supplier spend. Local origin alone is insufficient; certifications, ingredients and supplier attributes must be checked. Waste mass does not establish purchasing-credit eligibility. USC’s current reporting version is not verified. Source reviewed November 15, 2026: https://stars.aashe.org/resources-support/help-center/v3-operations/dining-service-procurement/
