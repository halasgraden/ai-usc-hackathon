import type { components } from "./api.generated";
export type Dish = components["schemas"]["DishRow"];
export type Dishes = components["schemas"]["DishesOut"];
export type Trends = components["schemas"]["TrendsOut"];
export type Week = components["schemas"]["TrendPoint"];
export type Detail = components["schemas"]["DishDetailOut"];
export type Report = components["schemas"]["ReportOut"];
export type Filters = {
  start: string;
  end: string;
  meal: string;
  station: string;
  search: string;
};
export type Note = {
  id: string;
  dish: string;
  date: string;
  kind: string;
  text: string;
};
export const n = (value: number | null | undefined, digits = 1) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en-US", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      }).format(value);
export const money = (value: number | null | undefined) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(value);
export const shortDate = (value: string) =>
  new Date(`${value}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
export function offset(day: string, days: number) {
  const d = new Date(`${day}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}
export function monday(day: string) {
  const weekday = new Date(`${day}T12:00:00Z`).getUTCDay();
  return offset(day, -((weekday + 6) % 7));
}
export function query(filters: Filters) {
  return new URLSearchParams(Object.entries(filters).filter(([, v]) => v));
}
export async function api<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/backend/${path}`, { signal });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      response.status === 401
        ? "The API rejected the staff token. Check that both servers use the same LEFTOVER_STAFF_TOKEN."
        : typeof data.detail === "string"
          ? data.detail
          : "The requested filters are invalid. Check the date range and retry.",
    );
  return data as T;
}
export const STATIONS: Record<string, string> = {
  Grill: "#2463a0",
  Expo: "#087f77",
  "Salad Bar": "#75549a",
  Flexitarian: "#a25321",
  "Sweets and Treats": "#476c33",
  Crepes: "#78555c",
};
export const stationColor = (station?: string | null) =>
  STATIONS[station || ""] || "#52606c";
export function csvText(data: Dishes): string {
  const headers = [
    "Dish",
    "Station",
    "Observed plates",
    "Returned kg",
    "Returned kg low",
    "Returned kg high",
    "Mean leftover g",
    "Mean leftover g low",
    "Mean leftover g high",
    "Simulated served kg",
    "Servings",
    "Identified returned %",
    "Returned % low",
    "Returned % high",
    "Previous returned kg",
    "Reference portion index %",
    "Data source",
    "Hall",
    "Start",
    "End",
  ];
  const escape = (value: unknown) => {
    let str = value == null ? "" : String(value);
    if (/^[=+@\-\t\r]/.test(str)) str = `'${str}`;
    return `"${str.replaceAll('"', '""')}"`;
  };
  const rows = data.rows.map((d) => [
    d.dish_name,
    d.station,
    d.plates_seen,
    d.total_kg,
    d.kg_low,
    d.kg_high,
    d.mean_grams,
    d.mean_grams_low,
    d.mean_grams_high,
    d.serving_comparison?.served_kg,
    d.serving_comparison?.servings,
    d.serving_comparison?.returned_percent,
    d.serving_comparison?.returned_percent_low,
    d.serving_comparison?.returned_percent_high,
    d.previous_total_kg,
    d.mean_percent_left,
    data.data_source,
    data.hall,
    data.start,
    data.end,
  ]);
  return (
    "\ufeff" +
    [headers, ...rows].map((r) => r.map(escape).join(",")).join("\r\n")
  );
}
export function downloadCsv(data: Dishes) {
  const url = URL.createObjectURL(
    new Blob([csvText(data)], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `leftover-lens-${data.start}-${data.end}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export const hallName = (hall?: string) =>
  !hall
    ? "Dining hall"
    : hall === "university-village"
      ? "USC Village Dining Hall"
      : hall.replaceAll("-", " ").replace(/\b\w/g, (c) => c.toUpperCase());

export const costAssumption = (text?: string) =>
  text?.split("; set ")[0] ||
  "Ingredient cost uses an assumed per-kilogram factor.";
