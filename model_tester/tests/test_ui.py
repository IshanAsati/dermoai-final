import json
import os
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


TEST_ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = TEST_ROOT / "browser-sample.png"
SCREENSHOT_PATH = TEST_ROOT / "browser-result.png"
TEST_URL = os.environ.get("DERMOAI_TEST_URL", "http://127.0.0.1:7861/")


def main():
    Image.new("RGB", (640, 480), (157, 113, 94)).save(SAMPLE_PATH)
    console_errors = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.goto(TEST_URL, wait_until="networkidle", timeout=60_000)

        assert page.title() == "DermoAI V2 — Local Model Tester"
        page.locator("#imageInput").set_input_files(str(SAMPLE_PATH))
        page.locator("#analyzeButton").click()
        page.locator("#resultContent").wait_for(state="visible", timeout=60_000)

        rows = page.locator("#predictionList .prediction-row")
        assert rows.count() == 12
        assert page.locator(".signal-strip").count() == 0
        assert page.locator("#topCode").inner_text().strip()
        assert "%" in page.locator("#topConfidence").inner_text()
        assert page.locator("#resultState").inner_text() == "Analysis complete"
        assert not console_errors, console_errors

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        print(
            json.dumps(
                {
                    "rows": rows.count(),
                    "top_class": page.locator("#topCode").inner_text(),
                    "top_score": page.locator("#topConfidence").inner_text(),
                    "screenshot": str(SCREENSHOT_PATH),
                }
            )
        )
        browser.close()


if __name__ == "__main__":
    main()
