"use client";
import { useId, useState } from "react";
import { Table2, ChartNoAxesCombined } from "lucide-react";
import {
  type Dish,
  type Week,
  type Note,
  n,
  shortDate,
  pounds,
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
  const id = useId();
  const values = rows
    .filter((d) => !servings || d.serving_comparison != null)
    .map((d) => ({
      dish: d,
      value: servings
        ? d.serving_comparison!.returned_percent
        : pounds(d.total_kg)!,
      low: servings
        ? d.serving_comparison!.returned_percent_low
        : pounds(d.kg_low)!,
      high: servings
        ? d.serving_comparison!.returned_percent_high
        : pounds(d.kg_high)!,
    }))
    .sort((a, b) => b.value - a.value);
  const maximum = Math.max(1, ...values.map((d) => d.high));
  const unit = servings ? "%" : "lb";
  return (
    <section className="chart-section" aria-labelledby={id}>
      <div className="section-heading">
        <h2 id={id}>
          {servings ? "Returned share by dish" : "Returned food by dish"}
        </h2>
        <ViewToggle table={table} onChange={() => setTable(!table)} />
      </div>
      {!values.length ? (
        <p className="empty-chart">No comparable data for this period.</p>
      ) : table ? (
        <div className="table-scroll">
          <table>
            <caption className="sr-only">
              Returned food and estimate ranges
            </caption>
            <thead>
              <tr>
                <th>Dish</th>
                <th>Returned ({unit})</th>
                <th>Estimate range ({unit})</th>
              </tr>
            </thead>
            <tbody>
              {values.map((v) => (
                <tr key={v.dish.dish_name}>
                  <th>
                    <button
                      className="dish-link"
                      onClick={() => onSelect(v.dish)}
                    >
                      {v.dish.dish_name}
                    </button>
                  </th>
                  <td>{n(v.value)}</td>
                  <td>
                    {n(v.low)}–{n(v.high)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="ranked-chart">
          <div className="ranking-heading">
            <span>Dish</span>
            <span>
              Estimated returned {servings ? "share (%)" : "weight (lb)"}
            </span>
            <span>Estimate range</span>
          </div>
          {values.map((v) => (
            <button
              className="ranking-row"
              key={v.dish.dish_name}
              onClick={() => onSelect(v.dish)}
              aria-label={`${v.dish.dish_name}: ${n(v.value)} ${unit} returned, estimate range ${n(v.low)} to ${n(v.high)} ${unit}. Open details.`}
            >
              <span className="ranking-name">{v.dish.dish_name}</span>
              <span className="ranking-track">
                <span
                  className="ranking-bar"
                  style={{ width: `${(v.value / maximum) * 100}%` }}
                />
                <span
                  className="ranking-range"
                  style={{
                    left: `${(v.low / maximum) * 100}%`,
                    width: `${((v.high - v.low) / maximum) * 100}%`,
                  }}
                />
              </span>
              <span className="ranking-value">
                <strong>
                  {n(v.value)} {unit}
                </strong>
                <small>
                  {n(v.low)}–{n(v.high)} {unit}
                </small>
              </span>
            </button>
          ))}
        </div>
      )}
      {servings && (
        <p className="chart-foot">
          Based on simulated serving quantities.{" "}
          {rows.length - values.length > 0
            ? `${rows.length - values.length} dishes have no compatible serving records.`
            : ""}
        </p>
      )}
    </section>
  );
}

export function TrendChart({
  weeks,
  share = false,
  title,
  notes = [],
  daily = false,
}: {
  weeks: Week[];
  share?: boolean;
  title: string;
  notes?: Note[];
  daily?: boolean;
}) {
  const [table, setTable] = useState(false);
  const id = useId();
  const points = weeks.map((w) => ({
    week: w,
    mid: share ? w.serving_comparison?.returned_percent : pounds(w.kg_wasted),
    low: share ? w.serving_comparison?.returned_percent_low : pounds(w.kg_low),
    high: share
      ? w.serving_comparison?.returned_percent_high
      : pounds(w.kg_high),
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
    if (
      !prev ||
      offset(prev.week.week_start, daily ? 1 : 7) !== p.week.week_start
    ) {
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
                <th>{daily ? "Date" : "Week starting"}</th>
                <th>{share ? "Returned (%)" : "Returned (lb)"}</th>
                <th>Range ({share ? "%" : "lb"})</th>
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
                  <td>
                    {p.week.partial ? "Partial period" : "Complete period"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : points.length === 1 ? (
        <div className="single-period">
          <strong>
            {n(points[0].mid)} {share ? "%" : "lb"}
          </strong>
          <span>
            {shortDate(points[0].week.week_start)}
            {points[0].week.partial ? " · In progress" : ""}
          </span>
          <span>
            Estimate range: {n(points[0].low)}–{n(points[0].high)}{" "}
            {share ? "%" : "lb"}
          </span>
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${w} ${h}`}
          role="img"
          aria-labelledby={`${id}-title ${id}-desc`}
        >
          <title id={`${id}-title`}>{title}</title>
          <desc id={`${id}-desc`}>
            {daily ? "Daily " : "Weekly "}
            {share
              ? "identified returned percentage"
              : "estimated pounds returned"}
            . Shaded bands show estimate ranges. Gaps mean no comparable data.
            Partial periods have hollow markers. Use View as table for numeric
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
            {share ? "Returned (%)" : "Returned (lb)"}
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
        Shading: estimate range. * Incomplete period.
      </p>
    </section>
  );
}
