import { test, expect } from "@playwright/test";

test.describe("PaladinMedia Phase 13.1 Full-Stack Integration E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to frontend application
    await page.goto("/");
  });

  test("1. Application mounts and preloads capabilities and baseline fixtures", async ({ page }) => {
    // Verify document root
    const root = page.locator("#root");
    await expect(root).toBeVisible();

    // Verify navigation tabs
    const experimentTab = page.locator("button", { hasText: "Live Steganography Experiment" });
    const standaloneTab = page.locator("button", { hasText: "Standalone Carrier Extraction" });
    await expect(experimentTab).toBeVisible();
    await expect(standaloneTab).toBeVisible();

    // Verify operational status badge
    const statusBadge = page.locator("span", { hasText: "STANDBY" });
    await expect(statusBadge).toBeVisible();
  });

  test("2. End-to-end embedding pipeline execution and recovery", async ({ page }) => {
    // Click RUN STEGANOGRAPHY PIPELINE
    const runBtn = page.locator("button", { hasText: "RUN STEGANOGRAPHY PIPELINE" });
    await expect(runBtn).toBeEnabled();
    await runBtn.click();

    // Wait for pipeline completion
    await expect(page.locator("span", { hasText: "COMPLETED" })).toBeVisible({ timeout: 15000 });

    // Verify image fidelity metrics are reported
    const psnrText = page.locator("text=/\\d+\\.\\d+\\s*dB/");
    await expect(psnrText).toBeVisible();

    // Verify exact byte equality
    const equalityBadge = page.locator("text=100% BYTE EQUALITY");
    await expect(equalityBadge).toBeVisible();
  });

  test("3. Tab navigation to Standalone Carrier Extraction", async ({ page }) => {
    const standaloneTab = page.locator("button", { hasText: "Standalone Carrier Extraction" });
    await standaloneTab.click();

    // Verify Standalone Extraction UI controls are displayed
    await expect(page.locator("text=Standalone Carrier Extraction & Verification")).toBeVisible();
    await expect(page.locator("text=STEGO CARRIER (PNG)")).toBeVisible();
    await expect(page.locator("text=AES-256 Key")).toBeVisible();
  });

  test("4. Fail-closed security verification on corrupted carrier", async ({ page }) => {
    const standaloneTab = page.locator("button", { hasText: "Standalone Carrier Extraction" });
    await standaloneTab.click();

    // Attach corrupted carrier payload via file input
    const fileInput = page.locator('input[type="file"][accept="image/png"]');
    await fileInput.setInputFiles({
      name: "corrupted_stego.png",
      mimeType: "image/png",
      buffer: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x00]),
    });

    // Click extraction button
    const extractBtn = page.locator("button", { hasText: "EXTRACT HIDDEN SECRET PAYLOAD" });
    await expect(extractBtn).toBeEnabled();
    await extractBtn.click();

    // Verify fail-closed error container appears and app does not crash
    await expect(page.locator("text=EXTRACTION FAILED CLOSED")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("#root")).toBeVisible();
  });
});
