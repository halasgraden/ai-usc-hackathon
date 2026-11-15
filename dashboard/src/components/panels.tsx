"use client";
import { useEffect, useRef, useState } from "react";
import {
  X,
  FlaskConical,
  Printer,
  Mail,
  Copy,
  Plus,
  Trash2,
  AlertCircle,
} from "lucide-react";
import {
  api,
  type Detail,
  type Filters,
  type Note,
  type Report,
  money,
  n,
  mass,
  portion,
  shortDate,
  query,
  monday,
  offset,
  hallName,
  costAssumption,
} from "@/lib/data";
import { reportEmailHtml, procurementActions } from "@/lib/email-report";
import { TrendChart } from "./charts";

export function Simulated({ seeded }: { seeded: boolean }) {
  return seeded ? (
    <span
      className="badge"
      title="Simulated plate history and serving quantities; real menu dish names."
    >
      <FlaskConical size={13} /> Demo data
    </span>
  ) : (
    <span className="badge live">Plate observations</span>
  );
}
export function ErrorBox({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="error-box" role="alert">
      <AlertCircle size={20} />
      <div>
        <strong>We couldn’t load this data</strong>
        <p>{message}</p>
        {retry && <button onClick={retry}>Retry</button>}
      </div>
    </div>
  );
}
function useData<T>(path: string) {
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState(""),
    [retry, setRetry] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    setData(null);
    setError("");
    api<T>(path, abort.signal)
      .then(setData)
      .catch((e) => {
        if (!abort.signal.aborted) setError(e.message);
      });
    return () => abort.abort();
  }, [path, retry]);
  return { data, error, retry: () => setRetry((x) => x + 1) };
}

export function DetailPanel({
  dish,
  filters,
  today,
  onClose,
}: {
  dish: string;
  filters: Filters;
  today: string;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const detailFilters = new URLSearchParams({
    start: filters.start,
    end: filters.end,
    ...(filters.meal ? { meal: filters.meal } : {}),
  });
  const { data, error, retry } = useData<Detail>(
    `dashboard/dishes/${encodeURIComponent(dish)}?${detailFilters}`,
  );
  const [notes, setNotes] = useState<Note[]>([]),
    [text, setText] = useState(""),
    [kind, setKind] = useState("Recipe change"),
    [date, setDate] = useState(today),
    [noteError, setNoteError] = useState(""),
    [adding, setAdding] = useState(false);
  useEffect(() => {
    dialog.current?.showModal();
    try {
      const raw = JSON.parse(
        localStorage.getItem("leftover-lens-notes") || "[]",
      );
      setNotes(
        Array.isArray(raw)
          ? raw.filter(
              (x: Note) =>
                typeof x.id === "string" &&
                typeof x.dish === "string" &&
                typeof x.date === "string" &&
                typeof x.text === "string",
            )
          : [],
      );
    } catch {
      setNoteError("Local notes could not be loaded.");
    }
  }, []);
  const mine = notes.filter((note) => note.dish === dish);
  function save(next: Note[]) {
    try {
      localStorage.setItem("leftover-lens-notes", JSON.stringify(next));
      setNotes(next);
      setNoteError("");
      return true;
    } catch {
      setNoteError(
        "This browser could not save notes. Try enabling local storage.",
      );
      return false;
    }
  }
  const d = data?.dish;
  return (
    <dialog
      ref={dialog}
      className="detail-dialog"
      onClose={onClose}
      aria-labelledby="detail-title"
    >
      <div className="panel-head">
        <span>Dish details</span>
        <button
          className="icon-button"
          onClick={() => dialog.current?.close()}
          aria-label="Close dish details"
        >
          <X size={22} />
        </button>
      </div>
      <div className="panel-body">
        <h2 id="detail-title">{dish}</h2>
        <p className="muted">
          {d?.station} · {shortDate(filters.start)}–{shortDate(filters.end)}
        </p>

        {error ? (
          <ErrorBox message={error} retry={retry} />
        ) : !d ? (
          <p role="status">Loading dish evidence…</p>
        ) : (
          <>
            <div className="detail-numbers">
              <div>
                <span>Estimated returned</span>
                <strong>
                  {mass(d.total_kg)} <small>lb</small>
                </strong>
                <span>
                  {mass(d.kg_low)}–{mass(d.kg_high)} lb estimate range
                </span>
              </div>
              <div>
                <span>Left per observed plate</span>
                <strong>
                  {portion(d.mean_grams)} <small>lb</small>
                </strong>
                <span>
                  {n(d.plates_seen, 0)} plates with this dish detected
                </span>
              </div>
            </div>
            <section className="evidence-note">
              <h3>How to read this dish</h3>
              <p>
                {d.serving_comparison
                  ? `${n(d.serving_comparison.returned_percent)}% of the simulated served mass was identified on returned plates (${n(d.serving_comparison.returned_percent_low)}–${n(d.serving_comparison.returned_percent_high)}%). The denominator is ${mass(d.serving_comparison.served_kg)} lb across ${n(d.serving_comparison.servings, 0)} simulated servings.`
                  : "Serving comparison unavailable: there is no compatible, complete serving record for this observation population."}
              </p>
              <p>
                The reference-portion index is{" "}
                <strong>{n(d.mean_percent_left)}%</strong>. This compares
                leftover volume with a reference portion; it is not the
                percentage a student left.
              </p>
            </section>
            <TrendChart
              weeks={data!.weekly}
              title="Returned mass over time"
              notes={mine}
            />
            <TrendChart
              weeks={data!.weekly}
              share
              title="Compared with simulated servings"
              notes={mine}
            />
            <section className="notes-section">
              <div className="section-heading">
                <h3>Intervention notes</h3>
                <button
                  className="text-button"
                  onClick={() => setAdding(!adding)}
                  aria-expanded={adding}
                >
                  <Plus size={16} /> Add note
                </button>
              </div>
              <p className="muted">
                Saved on this browser. Notes mark timing, not proof of cause.
              </p>
              {adding && (
                <form
                  className="note-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (
                      save([
                        ...notes,
                        {
                          id: crypto.randomUUID(),
                          dish,
                          date,
                          kind,
                          text: text.trim(),
                        },
                      ])
                    ) {
                      setText("");
                      setAdding(false);
                    }
                  }}
                >
                  <label>
                    Change type
                    <select
                      value={kind}
                      onChange={(e) => setKind(e.target.value)}
                    >
                      <option>Recipe change</option>
                      <option>Portion change</option>
                      <option>Menu change</option>
                      <option>Observation</option>
                    </select>
                  </label>
                  <label>
                    Effective date
                    <input
                      type="date"
                      value={date}
                      max={today}
                      required
                      onChange={(e) => setDate(e.target.value)}
                    />
                  </label>
                  <label className="full">
                    What changed?
                    <textarea
                      value={text}
                      maxLength={1000}
                      required
                      onChange={(e) => setText(e.target.value)}
                    />
                  </label>
                  <button className="primary" disabled={!text.trim()}>
                    Save note
                  </button>
                </form>
              )}
              {noteError && <p role="alert">{noteError}</p>}
              {mine.length === 0 ? (
                <p className="empty-note">No changes recorded for this dish.</p>
              ) : (
                <ul className="notes-list">
                  {mine.toReversed().map((note) => (
                    <li key={note.id}>
                      <div>
                        <strong>{note.kind}</strong>
                        <span>{shortDate(note.date)}</span>
                        <p>{note.text}</p>
                      </div>
                      <button
                        className="icon-button"
                        aria-label={`Delete note: ${note.text}`}
                        onClick={() =>
                          save(notes.filter((x) => x.id !== note.id))
                        }
                      >
                        <Trash2 size={16} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </>
        )}
      </div>
    </dialog>
  );
}

export function ReportPanel({
  filters,
  today,
  emailInitially,
  onClose,
}: {
  filters: Filters;
  today: string;
  emailInitially: boolean;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [week, setWeek] = useState(offset(monday(today), -7)),
    [email, setEmail] = useState(emailInitially),
    [copied, setCopied] = useState("");
  const params = query({ ...filters, start: "", end: "" });
  params.set("week_start", week);
  const { data, error, retry } = useData<Report>(`report/weekly?${params}`);
  const previousParams = new URLSearchParams(params);
  previousParams.set("week_start", offset(week, -7));
  const { data: previousReport } = useData<Report>(
    `report/weekly?${previousParams}`,
  );
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  const emailText = data
    ? `Subject: Leftover Lens — dining report, ${shortDate(data.week_start)}–${shortDate(data.week_end!)}\n\n${data.data_source === "seeded" ? "SIMULATED DATA — demo quantities, not measured results.\n\n" : ""}${hallName(data.hall)}\n${filters.meal || "All meals"} · ${filters.station || "All stations"}${filters.search ? ` · Search: ${filters.search}` : ""}\n\nEstimated returned food: ${mass(data.headline.kg_wasted)} lb (${mass(data.headline.kg_low)}–${mass(data.headline.kg_high)} lb).\nObserved plates: ${n(data.plates_seen, 0)}.\nEstimated ingredient cost: ${money(data.headline.est_cost_usd)} (assumed factor).\n\nLargest returned quantities:\n${data.top_dishes.map((d) => `• ${d.dish_name}: ${mass(d.total_kg)} lb`).join("\n")}\n\n${data.headline.serving_comparison ? `Identified returns / simulated servings: ${n(data.headline.serving_comparison.returned_percent)}%. Unmatched food is excluded from this percentage.\n\n` : ""}Please review the attached report before changing production or recipes. These observations do not establish student preference or prove the effect of an intervention.\n\n${costAssumption(data.assumptions.cost_usd_per_kg)}\n\nProcurement recommendations:\n${procurementActions(
        data,
      )
        .map((action) => `• ${action}`)
        .join(
          "\n",
        )}\n\nPrepared by Leftover Lens. Email preview only; nothing has been sent.`
    : "";
  return (
    <dialog
      ref={dialog}
      className="report-dialog"
      aria-label="Weekly report preview"
      onClose={onClose}
    >
      <div className="panel-head no-print">
        <span>Report export</span>
        <button
          className="icon-button"
          aria-label="Close report preview"
          onClick={() => dialog.current?.close()}
        >
          <X size={22} />
        </button>
      </div>
      <div className="report-controls no-print">
        <label>
          Week containing
          <input
            type="date"
            max={offset(monday(today), -1)}
            value={week}
            onChange={(e) => e.target.value && setWeek(e.target.value)}
          />
        </label>
        <button aria-pressed={email} onClick={() => setEmail(!email)}>
          <Mail size={16} />
          {email ? "Report preview" : "Email preview"}
        </button>
        <button
          className="primary"
          disabled={!data || !!error}
          onClick={() => window.print()}
        >
          <Printer size={16} /> Print / Save as PDF
        </button>
      </div>
      {error ? (
        <ErrorBox message={error} retry={retry} />
      ) : !data ? (
        <p role="status">Preparing report…</p>
      ) : (
        <>
          {email && (
            <section className="email-preview no-print">
              <div className="section-heading">
                <h3>Email preview</h3>
                <button
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(emailText);
                      setCopied("Copied to clipboard.");
                    } catch {
                      setCopied("Select and copy the text below.");
                    }
                  }}
                >
                  <Copy size={16} /> Copy text
                </button>
              </div>
              <iframe
                title="Formatted weekly email"
                srcDoc={reportEmailHtml(data, previousReport)}
                sandbox=""
                style={{ width: "100%", height: 1100, border: 0 }}
              />
              <label className="sr-only" htmlFor="email-text">
                Email draft
              </label>
              <textarea id="email-text" readOnly value={emailText} />
              <p role="status">
                {copied || "Preview only. No email will be sent."}
              </p>
            </section>
          )}
          <article id="weekly-report" hidden={email}>
            <div className="report-brand">
              Leftover Lens <span>USC dining operations</span>
            </div>
            <h2 id="report-title">Weekly plate-waste report</h2>
            <p>
              {shortDate(data.week_start)}–{shortDate(data.week_end!)} ·{" "}
              {hallName(data.hall)}
              {data.partial ? " · Partial week" : ""}
            </p>
            <Simulated seeded={data.data_source === "seeded"} />
            <p className="report-scope">
              {filters.meal || "All meals"} ·{" "}
              {filters.station || "All stations"}
              {filters.search ? ` · Search: ${filters.search}` : ""}
            </p>
            <div className="report-summary">
              <div>
                <strong>{mass(data.headline.kg_wasted)} lb</strong>
                <span>Estimated returned food</span>
                <small>
                  {mass(data.headline.kg_low)}–{mass(data.headline.kg_high)} lb
                </small>
              </div>
              <div>
                <strong>{n(data.plates_seen, 0)}</strong>
                <span>Observed plates</span>
              </div>
              <div>
                <strong>{money(data.headline.est_cost_usd)}</strong>
                <span>Estimated ingredient cost</span>
                <small>Based on an assumed factor</small>
              </div>
            </div>
            <h3>Menu items by returned mass</h3>
            <table className="report-table">
              <thead>
                <tr>
                  <th>Dish</th>
                  <th>Returned (lb)</th>
                  <th>Range (lb)</th>
                  <th>Identified return share (%)</th>
                </tr>
              </thead>
              <tbody>
                {data.rows?.map((d) => (
                  <tr key={d.dish_name}>
                    <th>{d.dish_name}</th>
                    <td>{mass(d.total_kg)}</td>
                    <td>
                      {mass(d.kg_low)}–{mass(d.kg_high)}
                    </td>
                    <td>{n(d.serving_comparison?.returned_percent)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>
              {mass(data.unmatched_kg)} lb of unmatched food is included in the
              overall mass, but cannot be attributed to a dish.
            </p>
            <h3>Interpretation and methods</h3>
            <p>{data.methods_note}</p>
            <p>
              Serving quantities are simulated for the same captured meal
              population. Return share includes identified dish leftovers only.
              Ranges sum the density-based lower and upper mass estimates; they
              are not statistical confidence intervals. Missing serving
              comparisons are not zero waste.
            </p>
            <p>{costAssumption(data.assumptions.cost_usd_per_kg)}</p>
            <p>
              Use repeated comparable meal periods to assess a change. This
              report does not establish causation, popularity, procurement
              savings, or model accuracy.
            </p>
            <footer>
              Leftover Lens · Anonymous plates, useful evidence · Prepared{" "}
              {today}
            </footer>
          </article>
        </>
      )}
    </dialog>
  );
}
