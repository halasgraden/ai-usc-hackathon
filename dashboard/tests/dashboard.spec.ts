import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs/promises";
const url = "/?start=2026-10-19&end=2026-11-15&tab=menu";

test("full menu, linked detail, keyboard tabs, URL and exports", async ({
  page,
}) => {
  await page.goto(url);
  await expect(page.locator(".menu-table tbody tr")).toHaveCount(20);
  await expect(
    page.getByText("Demo data", { exact: true }).first(),
  ).toBeVisible();
  await expect(page.locator(".ranking-row")).toHaveCount(20);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.locator(".menu-table tbody tr")).toHaveCount(17);
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await page
    .getByRole("button", { name: "Compare with servings", exact: true })
    .click();
  await expect(page.locator(".ranking-row")).toHaveCount(20);
  await page
    .getByRole("button", { name: "View as table", exact: true })
    .first()
    .click();
  await expect(
    page.getByRole("columnheader", { name: "Returned (%)" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "View as chart", exact: true })
    .first()
    .click();
  await page.getByRole("searchbox", { name: "Search dishes" }).fill("Rice");
  await expect(page.locator(".menu-table tbody tr")).toHaveCount(1);
  await expect(page).toHaveURL(/search=Rice/);
  await page
    .locator(".menu-table")
    .getByRole("button", { name: "Rice Pilaf", exact: true })
    .click();
  await expect(page.getByRole("dialog", { name: "Rice Pilaf" })).toBeVisible();
  await page.getByRole("button", { name: "Add note", exact: true }).click();
  await page
    .getByLabel("What changed?")
    .fill("Reduced portion for a controlled demo comparison.");
  await page.getByRole("button", { name: "Save note", exact: true }).click();
  await expect(
    page.getByText("Reduced portion for a controlled demo comparison.", {
      exact: true,
    }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(
    page
      .locator(".menu-table")
      .getByRole("button", { name: "Rice Pilaf", exact: true }),
  ).toBeFocused();
  await page.getByRole("tab", { name: "Menu performance" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Trends" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.locator(".matrix tbody tr")).toHaveCount(1);
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(page.locator(".matrix tbody tr")).toHaveCount(37);
  await page.locator(".export-menu summary").click();
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Filtered data CSV for Excel" })
    .click();
  const file = await download;
  const content = await fs.readFile((await file.path())!, "utf8");
  expect(content).toContain("Returned lb low");
  expect(content).toContain("Rice Pilaf");
  expect(content.split("\r\n")).toHaveLength(38);
  await page.locator(".export-menu summary").click();
  await page
    .getByRole("button", { name: "Weekly report Preview and save as PDF" })
    .click();
  await page.getByLabel("Week containing").fill("2026-11-02");
  await expect(
    page.getByRole("heading", { name: "Weekly plate-waste report" }),
  ).toBeVisible();
  await expect(
    page.locator("#weekly-report .report-table tbody tr"),
  ).toHaveCount(37);
  await page
    .getByRole("button", { name: "Email preview", exact: true })
    .click();
  await expect(page.locator("#email-text")).toContainText("SIMULATED DATA");
  await page.keyboard.press("Escape");
});

test("API failure is explicit and never replaced by fictional frontend data", async ({
  page,
}) => {
  await page.route("**/api/backend/dashboard/**", (route) =>
    route.fulfill({
      status: 502,
      contentType: "application/json",
      body: JSON.stringify({
        detail:
          "The plate-data service is unavailable. Start FastAPI on port 8000, then retry.",
      }),
    }),
  );
  await page.goto(url);
  await expect(page.locator(".error-box")).toContainText("unavailable");
  await expect(page.locator(".ranking-row")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Retry", exact: true }),
  ).toBeVisible();
});

test("desktop and tablet accessibility and evidence captures", async ({
  page,
}) => {
  await fs.mkdir("../.impeccable/review", { recursive: true });
  for (const [name, width, height] of [
    ["desktop", 1280, 1000],
    ["tablet", 768, 1024],
  ] as const) {
    await page.setViewportSize({ width, height });
    await page.goto(url);
    await expect(page.locator(".menu-table tbody tr")).toHaveCount(20);
    await page.evaluate(() => document.fonts.ready);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    let results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);
    await page.screenshot({
      path: `../.impeccable/review/${name}.png`,
      fullPage: false,
    });
    await page.getByRole("tab", { name: "Trends" }).click();
    await expect(page.locator(".matrix tbody tr")).toHaveCount(37);
    results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);
    await page.screenshot({
      path: `../.impeccable/review/${name}-trends.png`,
      fullPage: false,
    });
  }
  await page
    .locator(".matrix")
    .getByRole("button", { name: "Rice Pilaf", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "How to read this dish" }),
  ).toBeVisible();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.screenshot({ path: "../.impeccable/review/tablet-detail.png" });
  await page.keyboard.press("Escape");
  await page.locator(".export-menu summary").click();
  await page
    .getByRole("button", { name: "Weekly report Preview and save as PDF" })
    .click();
  await page.getByLabel("Week containing").fill("2026-11-02");
  await expect(page.locator(".report-table tbody tr")).toHaveCount(37);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: "../.impeccable/review/report-letter.pdf",
    format: "Letter",
    printBackground: true,
  });
  await page.pdf({
    path: "../.impeccable/review/report-a4.pdf",
    format: "A4",
    printBackground: true,
  });
});

test("report modal remains named during loading and failure", async ({
  page,
}) => {
  await page.goto(url);
  await expect(page.locator(".menu-table tbody tr")).toHaveCount(20);
  await page.route("**/api/backend/report/**", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 400));
    await route.fulfill({
      status: 502,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Report service unavailable." }),
    });
  });
  await page.locator(".export-menu summary").click();
  await page
    .getByRole("button", { name: "Weekly report Preview and save as PDF" })
    .click();
  await expect(
    page.getByRole("dialog", { name: "Weekly report preview" }),
  ).toBeVisible();
  await expect(page.locator(".report-dialog .error-box")).toContainText(
    "unavailable",
  );
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
});

test("narrow reflow, zoom equivalent and reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const width of [640, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(url);
    await expect(page.locator(".menu-table tbody tr")).toHaveCount(20);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
  }
});
