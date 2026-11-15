"use client";
import { useId, useState } from "react";
import { ArrowUpRight, Table2, ChartNoAxesCombined } from "lucide-react";
import {
  type Dish,
  type Week,
  type Note,
  n,
  shortDate,
  stationColor,
  offset,
} from "@/lib/data";

export function ViewToggle({
  table,
  onChange,
}: {
  table: boolean;
  onChange: () => void;
}) {
  return (
    <button
      className="text-button chart-toggle"
      onClick={onChange}
      aria-pressed={table}
    >
      {table ? <ChartNoAxesCombined size={16} /> : <Table2 size={16} />}{" "}
      {table ? "View as chart" : "View as table"}
    </button>
  );
}

export function Scatter({
  rows,
  servings,
  onSelect,
}: {
  rows: Dish[];
  servings: boolean;
  onSelect: (d: Dish) => void;
}) {
  const [table, setTable] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);
  const id = useId();
  const eligible = rows.filter((d) =>
    servings ? d.serving_comparison != null : d.mean_grams != null,
  );
  const active = eligible.find((d) => d.dish_name === hovered) || eligible[0];
  const coords = (d: Dish) =>
    servings
      ? {
          x: d.serving_comparison!.served_kg,
          y: d.serving_comparison!.returned_percent,
          low: d.serving_comparison!.returned_percent_low,
          high: d.serving_comparison!.returned_percent_high,
        }
      : {
          x: d.plates_seen,
          y: d.mean_grams!,
          low: d.mean_grams_low ?? d.mean_grams!,
          high: d.mean_grams_high ?? d.mean_grams!,
        };
  const xMax = Math.max(1, ...eligible.map((d) => coords(d).x)) * 1.12;
  const yMax = Math.max(1, ...eligible.map((d) => coords(d).high)) * 1.18;
  const width = 1080,
    height = 340,
    left = 62,
    right = 28,
    top = 28,
    bottom = 53;
  const x = (v: number) => left + (v / xMax) * (width - left - right),
    y = (v: number) => height - bottom - (v / yMax) * (height - top - bottom);
  const xLabel = servings
    ? "Simulated food served (kg)"
    : "Plates with detected leftovers (count)";
  const yLabel = servings
    ? "Identified food returned (%)"
    : "Average leftovers (g / observed plate)";
  return (
    <section className="chart-section" aria-labelledby={id}>
      <div className="section-heading">
        <div>
          <h2 id={id}>The whole menu, at a glance</h2>
          <p>
            {servings
              ? "Compare identified leftovers with simulated serving quantities."
              : "See how often a dish comes back and how much is left."}
          </p>
        </div>
        <ViewToggle table={table} onChange={() => setTable(!table)} />
      </div>
      {eligible.length === 0 ? (
        <div className="empty-chart">
          No {servings ? "compatible serving records" : "dish observations"} in
          this selection. Try another date range.
        </div>
      ) : table ? (
        <div className="table-scroll">
          <table>
            <caption className="sr-only">
              Full-menu scatterplot data and estimate ranges
            </caption>
            <thead>
              <tr>
                <th>Dish</th>
                <th>{xLabel}</th>
                <th>{yLabel}</th>
                <th>Estimate range ({servings ? "%" : "g"})</th>
              </tr>
            </thead>
            <tbody>
              {eligible.map((d) => {
                const p = coords(d);
                return (
                  <tr key={d.dish_name}>
                    <th>
                      <button className="dish-link" onClick={() => onSelect(d)}>
                        {d.dish_name}
                      </button>
                    </th>
                    <td>{n(p.x, servings ? 1 : 0)}</td>
                    <td>{n(p.y)}</td>
                    <td>
                      {n(p.low)}–{n(p.high)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div className="plot-wrap">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="scatter"
              aria-labelledby={`${id}-title ${id}-desc`}
            >
              <title id={`${id}-title`}>
                All {eligible.length} dishes: {yLabel} against {xLabel}
              </title>
              <desc id={`${id}-desc`}>
                Each circle is a dish. Vertical lines show the estimate range.
                Tab to a dish and press Enter to inspect it. The View as table
                button provides equivalent data.
              </desc>
              <text x={left} y={15} className="axis-caption">
                {yLabel}
              </text>
              {[0, 1, 2, 3, 4].map((t) => (
                <g key={t}>
                  <line
                    x1={left}
                    x2={width - right}
                    y1={y((yMax * t) / 4)}
                    y2={y((yMax * t) / 4)}
                    className="grid-line"
                  />
                  <text
                    x={left - 12}
                    y={y((yMax * t) / 4) + 4}
                    textAnchor="end"
                    className="axis-label"
                  >
                    {n((yMax * t) / 4, 0)}
                  </text>
                  <text
                    x={x((xMax * t) / 4)}
                    y={height - bottom + 23}
                    textAnchor="middle"
                    className="axis-label"
                  >
                    {n((xMax * t) / 4, 0)}
                  </text>
                </g>
              ))}
              <text
                x={(width + left) / 2}
                y={height - 5}
                textAnchor="middle"
                className="axis-caption"
              >
                {xLabel}
              </text>
              {eligible.map((d) => {
                const p = coords(d),
                  focused = active?.dish_name === d.dish_name;
                return (
                  <g
                    key={d.dish_name}
                    className="plot-point"
                    role="button"
                    tabIndex={0}
                    aria-label={`${d.dish_name}: ${n(p.x, servings ? 1 : 0)} ${servings ? "kg served" : "observed plates"}, ${n(p.y)} ${servings ? "percent returned" : "grams left per plate"}, range ${n(p.low)} to ${n(p.high)}. Open details.`}
                    onMouseEnter={() => setHovered(d.dish_name)}
                    onFocus={() => setHovered(d.dish_name)}
                    onClick={() => onSelect(d)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelect(d);
                      }
                    }}
                  >
                    <line
                      x1={x(p.x)}
                      x2={x(p.x)}
                      y1={y(p.low)}
                      y2={y(p.high)}
                      stroke={stationColor(d.station)}
                      strokeWidth={focused ? 2 : 1}
                      opacity={focused ? 1 : 0.85}
                    />
                    <circle
                      cx={x(p.x)}
                      cy={y(p.y)}
                      r={focused ? 8 : 5.5}
                      fill={stationColor(d.station)}
                      stroke="white"
                      strokeWidth={2}
                    />
                    <circle
                      className="point-focus"
                      cx={x(p.x)}
                      cy={y(p.y)}
                      r={12}
                      fill="none"
                      stroke="#182b3a"
                      strokeWidth={2}
                    />
                    <title>{d.dish_name}</title>
                  </g>
                );
              })}
            </svg>
          </div>
          {active && (
            <div className="plot-readout">
              <span
                className="station-dot"
                style={{ background: stationColor(active.station) }}
              />
              <div>
                <strong>{active.dish_name}</strong>
                <span>
                  {active.station ?? "Station unavailable"} ·{" "}
                  {n(active.total_kg)} kg returned
                </span>
              </div>
              <div className="readout-number">
                <strong>
                  {n(coords(active).y)} {servings ? "%" : "g"}
                </strong>
                <span>
                  range {n(coords(active).low)}–{n(coords(active).high)}{" "}
                  {servings ? "%" : "g"}
                </span>
              </div>
              <button className="text-button" onClick={() => onSelect(active)}>
                Inspect dish <ArrowUpRight size={16} />
              </button>
            </div>
          )}
        </>
      )}
      <div className="chart-foot">
        <span>
          {eligible.length} dishes shown
          {rows.length !== eligible.length
            ? ` · ${rows.length - eligible.length} without comparable data`
            : ""}{" "}
          · Lines show estimate ranges, not confidence intervals.
        </span>
        <span>Observations are not popularity.</span>
      </div>
    </section>
  );
}

export function TrendChart({
  weeks,
  share = false,
  title,
  notes = [],
}: {
  weeks: Week[];
  share?: boolean;
  title: string;
  notes?: Note[];
}) {
  const [table, setTable] = useState(false);
  const id = useId();
  const points = weeks.map((w) => ({
    week: w,
    mid: share ? w.serving_comparison?.returned_percent : w.kg_wasted,
    low: share ? w.serving_comparison?.returned_percent_low : w.kg_low,
    high: share ? w.serving_comparison?.returned_percent_high : w.kg_high,
  }));
  const valid = points.filter((p) => p.mid != null);
  const max = Math.max(1, ...valid.map((p) => p.high ?? p.mid ?? 0)) * 1.2;
  const w = 560,
    h = 240,
    left = 48,
    right = 20,
    top = 22,
    bottom = 42;
  const first = weeks[0]?.week_start,
    last = weeks.at(-1)?.week_start;
  const time = (day: string) => new Date(`${day}T12:00:00Z`).getTime();
  const x = (day: string) =>
    first && last
      ? first === last
        ? w / 2
        : left +
          ((time(day) - time(first)) / Math.max(1, time(last) - time(first))) *
            (w - left - right)
      : left;
  const y = (v: number) => h - bottom - (v / max) * (h - top - bottom);
  const segments: (typeof points)[] = [];
  for (const p of points) {
    const prev = segments.at(-1)?.at(-1);
    if (p.mid == null) continue;
    if (!prev || offset(prev.week.week_start, 7) !== p.week.week_start) {
      segments.push([p]);
    } else {
      segments.at(-1)!.push(p);
    }
  }
  return (
    <section className="trend-chart" aria-labelledby={id}>
      <div className="section-heading">
        <h2 id={id}>{title}</h2>
        <ViewToggle table={table} onChange={() => setTable(!table)} />
      </div>
      {valid.length === 0 ? (
        <div className="empty-chart">
          No {share ? "compatible serving comparisons" : "observations"} for
          this period.
        </div>
      ) : table ? (
        <div className="table-scroll">
          <table>
            <caption className="sr-only">{title} with estimate ranges</caption>
            <thead>
              <tr>
                <th>Week starting</th>
                <th>{share ? "Returned (%)" : "Returned (kg)"}</th>
                <th>Range ({share ? "%" : "kg"})</th>
                <th>Period</th>
              </tr>
            </thead>
            <tbody>
              {points.map((p) => (
                <tr key={p.week.week_start}>
                  <th>{shortDate(p.week.week_start)}</th>
                  <td>{n(p.mid)}</td>
                  <td>
                    {p.low == null ? "Unavailable" : `${n(p.low)}–${n(p.high)}`}
                  </td>
                  <td>{p.week.partial ? "Partial week" : "Complete week"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${w} ${h}`}
          role="img"
          aria-labelledby={`${id}-title ${id}-desc`}
        >
          <title id={`${id}-title`}>{title}</title>
          <desc id={`${id}-desc`}>
            Weekly{" "}
            {share
              ? "identified returned percentage"
              : "estimated kilograms returned"}
            . Shaded bands show estimate ranges. Gaps mean no comparable data.
            Partial weeks have hollow markers. Use View as table for numeric
            values.
          </desc>
          {[0, 1, 2, 3].map((i) => (
            <g key={i}>
              <line
                x1={left}
                x2={w - right}
                y1={y((max * i) / 3)}
                y2={y((max * i) / 3)}
                className="grid-line"
              />
              <text
                x={left - 10}
                y={y((max * i) / 3) + 4}
                textAnchor="end"
                className="axis-label"
              >
                {n((max * i) / 3, 0)}
              </text>
            </g>
          ))}
          <text x={left} y={14} className="axis-caption">
            {share ? "Returned (%)" : "Returned (kg)"}
          </text>
          {segments.map((segment, i) => (
            <g key={i}>
              <path
                d={`M ${segment.map((p) => `${x(p.week.week_start)},${y(p.high ?? p.mid!)}`).join(" L ")} L ${segment
                  .toReversed()
                  .map((p) => `${x(p.week.week_start)},${y(p.low ?? p.mid!)}`)
                  .join(" L ")} Z`}
                fill={share ? "#e5ddf0" : "#ddeaf5"}
              />
              <path
                d={`M ${segment.map((p) => `${x(p.week.week_start)},${y(p.mid!)}`).join(" L ")}`}
                fill="none"
                stroke={share ? "#75549a" : "#2463a0"}
                strokeWidth={2.5}
              />
            </g>
          ))}
          {points.map((p) => (
            <g key={p.week.week_start}>
              {p.mid != null && (
                <circle
                  cx={x(p.week.week_start)}
                  cy={y(p.mid)}
                  r={5}
                  fill={
                    p.week.partial ? "white" : share ? "#75549a" : "#2463a0"
                  }
                  stroke={share ? "#75549a" : "#2463a0"}
                  strokeWidth={2}
                />
              )}
              <text
                x={x(p.week.week_start)}
                y={h - 17}
                textAnchor="middle"
                className="axis-label"
              >
                {shortDate(p.week.week_start)}
                {p.week.partial ? "*" : ""}
              </text>
            </g>
          ))}
          {first &&
            last &&
            notes
              .filter(
                (note) => note.date >= first && note.date <= offset(last, 6),
              )
              .map((note) => (
                <g key={note.id}>
                  <line
                    x1={Math.min(w - right, x(note.date))}
                    x2={Math.min(w - right, x(note.date))}
                    y1={top}
                    y2={h - bottom}
                    stroke="#48535d"
                    strokeDasharray="4 4"
                  />
                  <title>
                    {note.kind}: {note.text}
                  </title>
                </g>
              ))}
        </svg>
      )}
      <p className="chart-foot">
        Shading: estimate range. * Partial week. Missing periods are not zero.
      </p>
    </section>
  );
}
