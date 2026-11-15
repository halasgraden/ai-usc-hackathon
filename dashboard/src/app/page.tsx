import Dashboard from "@/components/dashboard";
export const dynamic = "force-dynamic";
export default function Page() {
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Los_Angeles",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  return <Dashboard today={today} />;
}
