"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowDownToLine,
  ArrowRight,
  ArrowUp,
  ArrowUpDown,
  ChartNoAxesCombined,
  ChevronDown,
  CircleHelp,
  FileSpreadsheet,
  FileText,
  Mail,
  RefreshCw,
  ScanLine,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import {
  api,
  type Dishes,
  type Dish,
  type Trends,
  type Filters,
  type Note,
  n,
  money,
  offset,
  query,
  shortDate,
  stationColor,
  downloadCsv,
  hallName,
  costAssumption,
} from "@/lib/data";
import { Scatter, TrendChart, ViewToggle } from "./charts";
import { DetailPanel, ReportPanel, Simulated, ErrorBox } from "./panels";

type SortKey =
  | "dish_name"
  | "plates_seen"
  | "total_kg"
  | "mean_grams"
  | "served"
  | "returned"
  | "change";
function sortedValue(d: Dish, key: SortKey): number | string | null {
  if (key === "served") return d.serving_comparison?.served_kg ?? null;
  if (key === "returned") return d.serving_comparison?.returned_percent ?? null;
  if (key === "change")
    return d.previous_total_kg == null
      ? null
      : d.total_kg - d.previous_total_kg;
  return d[key] ?? null;
}
function Delta({ dish }: { dish: Dish }) {
  const diff =
    dish.previous_total_kg == null
      ? null
      : dish.total_kg - dish.previous_total_kg;
  if (diff == null) return <span className="muted">No baseline</span>;
  return (
    <span className="delta">
      {diff > 0.05 ? (
        <ArrowUp size={13} />
      ) : diff < -0.05 ? (
        <ArrowDown size={13} />
      ) : (
        <ArrowRight size={13} />
      )}{" "}
      {n(Math.abs(diff))} kg{" "}
      {diff > 0.05 ? "more" : diff < -0.05 ? "less" : "change"}
    </span>
  );
}

export default function Dashboard({ today }: { today: string }) {
  const initial: Filters = {
    start: offset(today, -6),
    end: today,
    meal: "",
    station: "",
    search: "",
  };
  const [filters, setFilters] = useState<Filters>(initial),
    [tab, setTab] = useState("menu"),
    [hydrated, setHydrated] = useState(false);
  const [data, setData] = useState<Dishes | null>(null),
    [trends, setTrends] = useState<Trends | null>(null),
    [error, setError] = useState(""),
    [trendError, setTrendError] = useState(""),
    [loading, setLoading] = useState(true),
    [loadedKey, setLoadedKey] = useState(""),
    [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState<string | null>(null),
    [report, setReport] = useState<"report" | "email" | null>(null),
    [servings, setServings] = useState(false),
    [sort, setSort] = useState<SortKey>("total_kg"),
    [descending, setDescending] = useState(true),
    [matrixShare, setMatrixShare] = useState(false),
    [matrixPlain, setMatrixPlain] = useState(false),
    [notes, setNotes] = useState<Note[]>([]),
    [storageMessage, setStorageMessage] = useState("");
  const exportMenu = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    function restore() {
      const p = new URLSearchParams(window.location.search);
      setFilters({
        start: p.get("start") || initial.start,
        end: p.get("end") || initial.end,
        meal: p.get("meal") || "",
        station: p.get("station") || "",
        search: p.get("search") || "",
      });
      setTab(p.get("tab") === "trends" ? "trends" : "menu");
      setHydrated(true);
    }
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, [today]); // eslint-disable-line react-hooks/exhaustive-deps
  const filterKey = query(filters).toString();
  useEffect(() => {
    if (!hydrated) return;
    const p = new URLSearchParams(filterKey);
    p.set("tab", tab);
    window.history.replaceState(null, "", `?${p}`);
  }, [filterKey, tab, hydrated]);
  useEffect(() => {
    if (!hydrated) return;
    const abort = new AbortController();
    const timer = setTimeout(() => {
      setLoading(true);
      setError("");
      setTrendError("");
      setData(null);
      setTrends(null);
      api<Dishes>(`dashboard/dishes?${filterKey}`, abort.signal)
        .then((result) => {
          setData(result);
          setLoadedKey(filterKey);
        })
        .catch((e) => {
          if (!abort.signal.aborted) setError(e.message);
        })
        .finally(() => {
          if (!abort.signal.aborted) setLoading(false);
        });
      api<Trends>(`dashboard/trends?${filterKey}`, abort.signal)
        .then(setTrends)
        .catch((e) => {
          if (!abort.signal.aborted) setTrendError(e.message);
        });
    }, 200);
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [filterKey, revision, hydrated]);
  useEffect(() => {
    try {
      const raw = JSON.parse(
        localStorage.getItem("leftover-lens-notes") || "[]",
      );
      setNotes(Array.isArray(raw) ? raw : []);
    } catch {
      setNotes([]);
    }
  }, [selected]);
  const change = (key: keyof Filters, value: string) =>
    setFilters((f) => ({ ...f, [key]: value }));
  const preset = (days: number) =>
    setFilters((f) => ({ ...f, start: offset(today, 1 - days), end: today }));
  const rows = [...(data?.rows || [])].sort((a, b) => {
    const va = sortedValue(a, sort),
      vb = sortedValue(b, sort);
    if (va == null) return vb == null ? 0 : 1;
    if (vb == null) return -1;
    const cmp =
      typeof va === "string"
        ? va.localeCompare(String(vb))
        : Number(va) - Number(vb);
    return descending ? -cmp : cmp;
  });
  const maxKg = Math.max(1, ...rows.map((d) => d.total_kg));
  const seeded =
    data?.data_source === "seeded" || trends?.data_source === "seeded";
  const currentFilterData =
    data?.start === filters.start && data?.end === filters.end && !loading;
  function sortBy(key: SortKey) {
    if (sort === key) setDescending(!descending);
    else {
      setSort(key);
      setDescending(key !== "dish_name");
    }
  }
  function th(label: string, key: SortKey) {
    return (
      <th
        scope="col"
        aria-sort={
          sort === key ? (descending ? "descending" : "ascending") : "none"
        }
      >
        <button
          className="sort-button"
          onClick={() => sortBy(key)}
          aria-label={`Sort by ${label}`}
        >
          {label}
          {sort === key ? (
            descending ? (
              <ArrowDown size={13} />
            ) : (
              <ArrowUp size={13} />
            )
          ) : (
            <ArrowUpDown size={12} />
          )}
        </button>
      </th>
    );
  }
  const selectedNotes = notes.filter((note) =>
    rows.some((d) => d.dish_name === note.dish),
  );
  const clearNotes = () => {
    try {
      localStorage.removeItem("leftover-lens-notes");
      setNotes([]);
      setStorageMessage(
        "Local intervention notes reset. Plate data is unchanged.",
      );
    } catch {
      setStorageMessage("Could not reset local notes in this browser.");
    }
  };
  return (
    <>
      <div className="app-shell">
        <a className="skip-link" href="#main-content">
          Skip to dining data
        </a>
        <header className="masthead">
          <Link className="brand" href="/" aria-label="Leftover Lens home">
            <span className="brand-mark">
              <ScanLine size={25} />
            </span>
            <strong>Leftover Lens</strong>
          </Link>
          <div className="brand-context">
            <span className="usc-word">USC</span>
            <span>Dining operations</span>
          </div>
        </header>
        <main id="main-content">
          <div className="page-heading">
            <div>
              <div className="heading-line">
                <h1>Dining insights</h1>
                {data && <Simulated seeded={seeded} />}
              </div>
              <p>
                {hallName(data?.hall)} <span className="separator">/</span> A
                full-menu view of what comes back.
              </p>
            </div>
            <details ref={exportMenu} className="export-menu">
              <summary className="button primary">
                <ArrowDownToLine size={17} /> Export <ChevronDown size={15} />
              </summary>
              <div className="export-options">
                <button
                  disabled={!data || loading || loadedKey !== filterKey}
                  onClick={() => {
                    if (data) downloadCsv(data);
                    exportMenu.current!.open = false;
                  }}
                >
                  <FileSpreadsheet size={17} />
                  <span>
                    Filtered data <small>CSV for Excel</small>
                  </span>
                </button>
                <button
                  onClick={() => {
                    setReport("report");
                    exportMenu.current!.open = false;
                  }}
                >
                  <FileText size={17} />
                  <span>
                    Weekly report <small>Preview and save as PDF</small>
                  </span>
                </button>
                <button
                  onClick={() => {
                    setReport("email");
                    exportMenu.current!.open = false;
                  }}
                >
                  <Mail size={17} />
                  <span>
                    Email preview <small>Committee summary</small>
                  </span>
                </button>
              </div>
            </details>
          </div>
          <div
            className="tabs"
            role="tablist"
            aria-label="Dining analysis views"
            onKeyDown={(e) => {
              if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
                e.preventDefault();
                const next =
                  e.key === "Home"
                    ? "menu"
                    : e.key === "End"
                      ? "trends"
                      : tab === "menu"
                        ? "trends"
                        : "menu";
                setTab(next);
                document.getElementById(`tab-${next}`)?.focus();
              }
            }}
          >
            <button
              id="tab-menu"
              role="tab"
              aria-controls="panel-menu"
              aria-selected={tab === "menu"}
              tabIndex={tab === "menu" ? 0 : -1}
              onClick={() => setTab("menu")}
            >
              Menu performance
            </button>
            <button
              id="tab-trends"
              role="tab"
              aria-controls="panel-trends"
              aria-selected={tab === "trends"}
              tabIndex={tab === "trends" ? 0 : -1}
              onClick={() => setTab("trends")}
            >
              Trends
            </button>
            <span className="tab-note">Anonymous plates. Useful evidence.</span>
          </div>
          <section
            className="filter-bar"
            aria-label="Filter dining observations"
          >
            <div className="date-filter">
              <div className="presets" aria-label="Date presets">
                {[
                  [1, "Today"],
                  [7, "7 days"],
                  [28, "4 weeks"],
                ].map(([days, label]) => (
                  <button
                    key={days}
                    aria-pressed={
                      filters.start === offset(today, 1 - Number(days)) &&
                      filters.end === today
                    }
                    onClick={() => preset(Number(days))}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="date-inputs">
                <label>
                  <span className="sr-only">Start date</span>
                  <input
                    type="date"
                    value={filters.start}
                    max={filters.end}
                    onChange={(e) => change("start", e.target.value)}
                  />
                </label>
                <span>–</span>
                <label>
                  <span className="sr-only">End date</span>
                  <input
                    type="date"
                    value={filters.end}
                    min={filters.start}
                    max={today}
                    onChange={(e) => change("end", e.target.value)}
                  />
                </label>
              </div>
            </div>
            <label className="select-filter">
              <span>Meal</span>
              <select
                value={filters.meal}
                onChange={(e) => change("meal", e.target.value)}
              >
                <option value="">All meals</option>
                {["Breakfast", "Brunch", "Lunch", "Dinner"].map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </label>
            <label className="select-filter">
              <span>Station</span>
              <select
                value={filters.station}
                onChange={(e) => change("station", e.target.value)}
              >
                <option value="">All stations</option>
                {Array.from(
                  new Set([
                    ...(data?.stations || []),
                    ...(filters.station ? [filters.station] : []),
                  ]),
                )
                  .sort()
                  .map((s) => (
                    <option key={s}>{s}</option>
                  ))}
              </select>
            </label>
            <label className="search-filter">
              <Search size={16} />
              <span className="sr-only">Search dishes</span>
              <input
                type="search"
                placeholder="Find a dish…"
                value={filters.search}
                onChange={(e) => change("search", e.target.value)}
              />
            </label>
            <button
              className="icon-button refresh"
              onClick={() => setRevision((x) => x + 1)}
              aria-label="Refresh data"
            >
              <RefreshCw size={17} />
            </button>
          </section>
          {(filters.search || filters.station || filters.meal) && (
            <div className="active-filters">
              <SlidersHorizontal size={14} />
              <span>
                {filters.meal || "All meals"} ·{" "}
                {filters.station || "All stations"}
                {filters.search ? ` · “${filters.search}”` : ""}
              </span>
              <button
                className="text-button"
                onClick={() =>
                  setFilters((f) => ({
                    ...f,
                    meal: "",
                    station: "",
                    search: "",
                  }))
                }
              >
                Clear filters <X size={13} />
              </button>
            </div>
          )}
          {error ? (
            <ErrorBox message={error} retry={() => setRevision((x) => x + 1)} />
          ) : loading || !data ? (
            <div className="loading" role="status">
              <div />
              <div />
              <div />
              <span>Loading plate observations…</span>
            </div>
          ) : (
            <>
              <section
                className="summary-strip"
                aria-label="Selected period summary"
              >
                <div>
                  <span>Estimated returned food</span>
                  <strong>
                    {n(data.headline?.kg_wasted)} <small>kg</small>
                  </strong>
                  <p>
                    {n(data.headline?.kg_low)}–{n(data.headline?.kg_high)} kg
                    estimate range
                  </p>
                </div>
                <div>
                  <span>Observed plates</span>
                  <strong>{n(data.plates_seen, 0)}</strong>
                  <p>
                    {filters.station || filters.search
                      ? "Distinct plates in this selection"
                      : "Includes plates with no detected leftovers"}
                  </p>
                </div>
                <div>
                  <span>
                    Estimated ingredient cost{" "}
                    <details className="inline-help">
                      <summary aria-label="Cost assumption">
                        <CircleHelp size={14} />
                      </summary>
                      <p>{costAssumption(data.assumptions?.cost_usd_per_kg)}</p>
                    </details>
                  </span>
                  <strong>{money(data.headline?.est_cost_usd)}</strong>
                  <p>Assumed food cost · not realized savings</p>
                </div>
              </section>
              <div
                id="panel-menu"
                role="tabpanel"
                aria-labelledby="tab-menu"
                hidden={tab !== "menu"}
              >
                <div className="mode-line">
                  <div className="segmented" aria-label="Chart metric">
                    <button
                      aria-pressed={!servings}
                      onClick={() => setServings(false)}
                    >
                      Plate observations
                    </button>
                    <button
                      aria-pressed={servings}
                      onClick={() => setServings(true)}
                    >
                      Compare with servings
                    </button>
                  </div>
                  <span>
                    {servings
                      ? "Serving quantities are simulated."
                      : "Every eligible dish in the selected period."}
                  </span>
                </div>
                <Scatter
                  rows={data.rows}
                  servings={servings}
                  onSelect={(d) => setSelected(d.dish_name)}
                />
                <section className="menu-section" aria-labelledby="menu-title">
                  <div className="section-heading">
                    <div>
                      <h2 id="menu-title">
                        Menu breakdown{" "}
                        <span className="count">{rows.length}</span>
                      </h2>
                      <p>
                        Compare dishes, then select one to explore its evidence.
                      </p>
                    </div>
                    <span className="comparison-note">
                      Change vs. preceding{" "}
                      {Math.round(
                        (new Date(filters.end).getTime() -
                          new Date(filters.start).getTime()) /
                          86400000,
                      ) + 1}{" "}
                      days
                    </span>
                  </div>
                  {rows.length === 0 ? (
                    <div className="empty-state">
                      <Search size={28} />
                      <h3>No dish observations in this selection</h3>
                      <p>
                        Try a wider date range or clear the filters. No
                        observations does not mean zero waste.
                      </p>
                      <button onClick={() => setFilters(initial)}>
                        Reset filters
                      </button>
                    </div>
                  ) : (
                    <div className="table-scroll menu-table-wrap">
                      <table className="menu-table">
                        <caption className="sr-only">
                          All filtered menu dishes with mass estimates and
                          simulated serving comparisons
                        </caption>
                        <thead>
                          <tr>
                            {th("Dish / station", "dish_name")}
                            {th("Observed plates", "plates_seen")}
                            {th("Returned kg", "total_kg")}
                            {th("Grams / plate", "mean_grams")}
                            {th("Served kg¹", "served")}
                            {th("Returned %¹", "returned")}
                            {th("Change in kg", "change")}
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((d) => (
                            <tr key={d.dish_name}>
                              <th scope="row">
                                <button
                                  className="dish-link"
                                  onClick={() => setSelected(d.dish_name)}
                                >
                                  {d.dish_name}
                                  <ArrowRight size={14} />
                                </button>
                                <span className="station-name">
                                  <i
                                    style={{
                                      background: stationColor(d.station),
                                    }}
                                  />
                                  {d.station || "Station unavailable"}
                                </span>
                              </th>
                              <td>{n(d.plates_seen, 0)}</td>
                              <td className="mass-cell">
                                <strong>{n(d.total_kg)}</strong>
                                <span
                                  className="mass-bar"
                                  style={{
                                    width: `${(d.total_kg / maxKg) * 100}%`,
                                  }}
                                />
                                <small>
                                  {n(d.kg_low)}–{n(d.kg_high)}
                                </small>
                              </td>
                              <td>
                                {n(d.mean_grams)}
                                <small>
                                  {n(d.mean_grams_low)}–{n(d.mean_grams_high)} g
                                </small>
                              </td>
                              <td>{n(d.serving_comparison?.served_kg)}</td>
                              <td>
                                {n(d.serving_comparison?.returned_percent)}
                                {d.serving_comparison && (
                                  <small>
                                    {n(
                                      d.serving_comparison.returned_percent_low,
                                    )}
                                    –
                                    {n(
                                      d.serving_comparison
                                        .returned_percent_high,
                                    )}
                                    %
                                  </small>
                                )}
                              </td>
                              <td>
                                <Delta dish={d} />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <div className="table-notes">
                    <p>
                      ¹ Simulated servings; percentage = identified returned
                      mass ÷ corresponding served mass. Unavailable comparisons
                      appear as —.
                    </p>
                    <p>
                      {n(data.unmatched_kg)} kg of unmatched food is included in
                      the summary, but cannot be assigned to a dish. Dish
                      observation counts overlap; do not add them.
                    </p>
                  </div>
                </section>
              </div>
              <div
                id="panel-trends"
                role="tabpanel"
                aria-labelledby="tab-trends"
                hidden={tab !== "trends"}
              >
                <div className="trend-intro">
                  <div>
                    <h2>See what changes over time</h2>
                    <p>
                      Read mass and serving share together. Partial weeks are
                      marked, not projected.
                    </p>
                  </div>
                  <ChartNoAxesCombined size={25} />
                </div>
                {trendError ? (
                  <ErrorBox
                    message={trendError}
                    retry={() => setRevision((x) => x + 1)}
                  />
                ) : !trends ? (
                  <p role="status">Loading weekly comparisons…</p>
                ) : (
                  <>
                    <div className="trend-grid">
                      <TrendChart
                        weeks={trends.weeks}
                        title="Estimated returned food"
                        notes={selectedNotes}
                      />
                      <TrendChart
                        weeks={trends.weeks}
                        title="Identified share of servings"
                        share
                        notes={selectedNotes}
                      />
                    </div>
                    <section className="matrix-section">
                      <div className="section-heading">
                        <div>
                          <h2>Every dish, week by week</h2>
                          <p>
                            Numeric values remain visible at every intensity.
                          </p>
                        </div>
                        <div className="matrix-controls">
                          <label>
                            <span className="sr-only">
                              Weekly matrix metric
                            </span>
                            <select
                              value={matrixShare ? "share" : "mass"}
                              onChange={(e) =>
                                setMatrixShare(e.target.value === "share")
                              }
                            >
                              <option value="mass">Returned kilograms</option>
                              <option value="share">Returned share (%)</option>
                            </select>
                          </label>
                          <ViewToggle
                            table={matrixPlain}
                            onChange={() => setMatrixPlain(!matrixPlain)}
                          />
                        </div>
                      </div>
                      <div className="table-scroll">
                        <table className="matrix">
                          <caption className="sr-only">
                            Dish by week,{" "}
                            {matrixShare
                              ? "returned share percent"
                              : "returned kilograms"}{" "}
                            with estimate ranges
                          </caption>
                          <thead>
                            <tr>
                              <th>Dish</th>
                              {trends.weeks.map((w) => (
                                <th key={w.week_start}>
                                  {shortDate(w.week_start)}
                                  <small>
                                    week starting{w.partial ? " · partial" : ""}
                                  </small>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {trends.dishes?.map((d) => (
                              <tr key={d.dish_name}>
                                <th>
                                  <button
                                    className="dish-link"
                                    onClick={() => setSelected(d.dish_name)}
                                  >
                                    {d.dish_name}
                                  </button>
                                </th>
                                {trends.weeks.map((w) => {
                                  const p = d.weeks.find(
                                      (p) => p.week_start === w.week_start,
                                    ),
                                    v = matrixShare
                                      ? p?.serving_comparison?.returned_percent
                                      : p?.kg_wasted,
                                    low = matrixShare
                                      ? p?.serving_comparison
                                          ?.returned_percent_low
                                      : p?.kg_low,
                                    high = matrixShare
                                      ? p?.serving_comparison
                                          ?.returned_percent_high
                                      : p?.kg_high;
                                  return (
                                    <td
                                      key={w.week_start}
                                      style={{
                                        background:
                                          matrixPlain || v == null
                                            ? "transparent"
                                            : `rgba(36,99,160,${Math.min(0.2, 0.025 + (v / (matrixShare ? 100 : 50)) * 0.18)})`,
                                      }}
                                    >
                                      {v == null
                                        ? "Unavailable"
                                        : `${n(v)} ${matrixShare ? "%" : "kg"}`}
                                      <small>
                                        {low == null
                                          ? ""
                                          : `${n(low)}–${n(high)} ${matrixShare ? "%" : "kg"}`}
                                      </small>
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {trends.dishes?.length === 0 && (
                        <p className="empty-note">
                          No weekly dish observations in this period.
                        </p>
                      )}
                    </section>
                  </>
                )}
              </div>
              <div className="data-footer">
                <span>
                  <span className="status-dot" />{" "}
                  {currentFilterData
                    ? "Connected to plate-data service"
                    : "Updating"}{" "}
                  · {shortDate(data.start)}–{shortDate(data.end)} ·
                  America/Los_Angeles
                </span>
                <details>
                  <summary>Data & methods</summary>
                  <div>
                    <h3>What these numbers mean</h3>
                    <p>
                      Mass is estimated from food volume and category density.
                      Bounds are summed estimate ranges, not statistical
                      confidence intervals. Observed plates are distinct
                      returns, not diners or servings. The serving comparison is
                      available only for compatible simulated cohorts.
                    </p>
                    <p>{costAssumption(data.assumptions?.cost_usd_per_kg)}</p>
                    <p>
                      This demo uses the USC Village menu catalog with fictional
                      historical quantities. It is not connected to CBORD.
                      Existing recipe and purchasing workflows stay in that
                      system.
                    </p>
                    <button onClick={clearNotes}>
                      Reset local intervention notes
                    </button>
                    <p role="status">{storageMessage}</p>
                  </div>
                </details>
              </div>
            </>
          )}
        </main>
        <footer className="site-footer">
          <span>Leftover Lens</span>
          <span>Built for better dining decisions.</span>
          <span>Plate estimates, never student identities.</span>
        </footer>
      </div>
      {selected && (
        <DetailPanel
          key={selected}
          dish={selected}
          filters={filters}
          today={today}
          onClose={() => setSelected(null)}
        />
      )}
      {report && (
        <ReportPanel
          filters={filters}
          today={today}
          emailInitially={report === "email"}
          onClose={() => setReport(null)}
        />
      )}
    </>
  );
}
