import type { Report } from "./data";
const esc = (s: string) =>
  s.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const lb = (v: number) => v * 2.20462262185;
const num = (v: number) =>
  v.toLocaleString("en-US", { maximumFractionDigits: 1 });
export function procurementActions(report: Report): string[] {
  const rows = [...(report.rows || report.top_dishes)].sort(
    (a, b) => b.total_kg - a.total_kg,
  );
  const dish = rows[0];
  if (!dish)
    return ["Hold purchasing changes until dish measurements are available."];
  const served = dish.serving_comparison;
  const rising = rows
    .filter(
      (d) => d.previous_total_kg != null && d.total_kg > d.previous_total_kg,
    )
    .sort(
      (a, b) =>
        b.total_kg - b.previous_total_kg! - (a.total_kg - a.previous_total_kg!),
    )[0];
  const actions = [
    `${dish.dish_name}: ${num(lb(dish.total_kg))} lb came back${served ? ` (${num(served.returned_percent)}% of the served weight)` : ""}. Trial a 5% smaller production forecast for comparable attendance${served ? `—${num(lb(served.served_kg) * 0.95)} lb if the next forecast would otherwise match this week’s ${num(lb(served.served_kg))} lb served` : ""}. Keep a reserve batch available.`,
  ];
  if (rising && rising.dish_name !== dish.dish_name)
    actions.push(
      `${rising.dish_name}: returns rose from ${num(lb(rising.previous_total_kg!))} to ${num(lb(rising.total_kg))} lb. Check portions and meal attendance before increasing its next ingredient order; prepare smaller batches during the trial.`,
    );
  actions.push(
    `Next ${dish.dish_name} order: if the trial meets demand without shortages, scale its recipe to the lower forecast. Deduct usable inventory and incoming deliveries before rounding to vendor case packs. The returned weight is cooked food, not the amount of raw ingredients to remove from the order.`,
  );
  actions.push(
    `Keep the existing ingredient quality and sustainability specifications for ${dish.dish_name}. Reduce quantity before changing products; retain invoice amounts and any verified product certifications with the revised order.`,
  );
  return actions;
}

/** Email uses presentation tables and inline CSS for mail-client compatibility.
 * Seeded quantities remain disclosed once, near the heading; no synthetic claims. */
export function reportEmailHtml(
  report: Report,
  previous?: Report | null,
): string {
  const h = report.headline,
    rows = [...(report.rows || report.top_dishes)].sort(
      (a, b) => b.total_kg - a.total_kg,
    ),
    top = rows.slice(0, 5);
  const total = lb(h.kg_wasted),
    old = previous ? lb(previous.headline.kg_wasted) : null;
  const topShare = h.kg_wasted
    ? (top.reduce((s, d) => s + d.total_kg, 0) / h.kg_wasted) * 100
    : 0;
  const delta = old && old > 0 ? ((total - old) / old) * 100 : null;
  const bar = (
    label: string,
    value: number,
    max: number,
    detail: string,
    color = "#2463a0",
  ) =>
    `<tr><td style="padding:12px 0;border-bottom:1px solid #e1e6eb"><strong>${esc(label)}</strong><table role="presentation" width="100%"><tr><td style="padding:8px 0"><table role="presentation" width="${Math.max(1, (value / max) * 100).toFixed(1)}%"><tr><td height="14" bgcolor="${color}" style="height:14px;background:${color};font-size:1px">&nbsp;</td></tr></table></td></tr></table><span style="font-size:14px;color:#46535e">${detail}</span></td></tr>`;
  const heading = (t: string) =>
    `<h2 style="font-size:20px;line-height:1.3;margin:32px 0 12px;color:#243440">${t}</h2>`;
  return `<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weekly dining report</title></head><body style="margin:0;background:#f3f5f7;color:#243440;font:16px/1.5 Arial,sans-serif"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:24px 12px"><table role="presentation" width="640" style="width:100%;max-width:640px;background:white" cellpadding="0" cellspacing="0"><tr><td style="border-top:5px solid #990000;padding:32px 24px">
 <p style="margin:0 0 12px;font-size:13px;font-weight:bold;color:#990000">LEFTOVER LENS · WEEKLY DINING REPORT</p><h1 style="font-size:28px;line-height:1.2;margin:0 0 8px">USC Village Dining Hall</h1><p style="margin:0;color:#46535e">${esc(report.week_start)} – ${esc(report.week_end || "")}</p>
 ${report.data_source === "seeded" ? '<p style="font-size:13px;color:#46535e;margin:16px 0">Demo data · Assumed estimate range: ±5%.</p>' : ""}
 <h2 style="font-size:36px;line-height:1.2;margin:24px 0 4px">${num(total)} lb returned</h2><p style="margin:0;color:#46535e">Estimate range ${num(lb(h.kg_low ?? h.kg_wasted))}–${num(lb(h.kg_high ?? h.kg_wasted))} lb</p>
 <p style="font-size:18px;margin:16px 0">${delta == null ? "" : `<strong>${num(Math.abs(delta))}% ${delta < 0 ? "less" : "more"}</strong> returned food than the preceding week. `}The five largest contributors account for <strong>${num(topShare)}%</strong> of returned weight.</p>
 <table width="100%" style="border-top:1px solid #e1e6eb;border-bottom:1px solid #e1e6eb"><tr><td style="padding:16px 12px 16px 0;vertical-align:top;width:50%"><strong style="font-size:24px">${num(report.plates_seen || 0)}</strong><br>Observed plates</td><td style="padding:16px 12px 16px 0;vertical-align:top;width:50%"><strong style="font-size:24px">$${Math.round(h.est_cost_usd || 0).toLocaleString("en-US")}</strong><br>Estimated ingredient cost</td></tr></table>
 ${previous ? heading("Weekly totals") + `<table width="100%" role="presentation">${bar("Previous week", old!, Math.max(old!, total), `${num(old!)} lb · ${num(previous.plates_seen || 0)} observed plates`, "#677887")}${bar("This week", total, Math.max(old!, total), `${num(total)} lb · ${num(report.plates_seen || 0)} observed plates`)}</table>` : ""}
 ${heading("Most returned dishes")}<table role="presentation" width="100%">${top.map((d) => bar(d.dish_name, lb(d.total_kg), Math.max(1, lb(top[0].total_kg)), `<strong>${num(lb(d.total_kg))} lb</strong> · range ${num(lb(d.kg_low ?? d.total_kg))}–${num(lb(d.kg_high ?? d.total_kg))} lb`)).join("")}</table>
 ${heading("Procurement recommendation")}
 <ul style="padding-left:22px;margin:0">${procurementActions(report)
   .map((action) => `<li style="margin-bottom:12px">${esc(action)}</li>`)
   .join("")}</ul>
 <p style="font-size:13px;color:#46535e;margin:16px 0 0">5% is a proposed trial; no orders have been changed.</p>
 </td></tr></table></td></tr></table></body></html>`;
}
