import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  // This is a local-demo BFF. Bind Next to loopback; add institutional auth before deployment.
  const { path } = await context.params;
  const allowed =
    (path[0] === "dashboard" &&
      ["today", "dishes", "trends"].includes(path[1])) ||
    (path.length === 2 && path[0] === "report" && path[1] === "weekly");
  if (!allowed || path.some((p) => p === ".." || p === ".")) {
    return NextResponse.json({ detail: "Unknown endpoint." }, { status: 404 });
  }
  const token = process.env.LEFTOVER_STAFF_TOKEN;
  if (!token)
    return NextResponse.json(
      {
        detail:
          "Set LEFTOVER_STAFF_TOKEN on the Next.js server, then restart it.",
      },
      { status: 503 },
    );
  try {
    const base = (
      process.env.LEFTOVER_API_URL || "http://127.0.0.1:8000"
    ).replace(/\/$/, "");
    const response = await fetch(
      `${base}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`,
      {
        headers: { Authorization: `Bearer ${token}` },
        cache: "no-store",
        signal: AbortSignal.timeout(15000),
      },
    );
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "The plate-data service is unavailable. Start FastAPI on port 8000, then retry.",
      },
      { status: 502 },
    );
  }
}
