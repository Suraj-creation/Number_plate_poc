"""Optional Chromium test for a running UI with the real pretrained stack."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright


def fixtures(root):
    source = Path(__file__).resolve().parents[1] / "input/videos/india_traffic_acceptance_2m14s_3fps.mp4"
    ffmpeg = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    for name, count in (("traffic", 4), ("cancel", 45)):
        subprocess.run([*ffmpeg, "-frames:v", str(count), "-c:v", "libx264", "-threads", "2",
                        str(root / f"{name}.mp4")], check=True)
    subprocess.run([*ffmpeg, "-frames:v", "2", "-threads", "2", str(root / "day-%d.png")], check=True)
    (root / "corrupt.png").write_bytes(b"unreadable file")


def check_ui(browser, url, root, screenshots=None):
    page = browser.new_page(viewport={"width": 1400, "height": 960}, accept_downloads=True)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("https://**/*", lambda route: route.abort())
    page.goto(url, wait_until="domcontentloaded")
    page.get_by_role("button", name="Process images").wait_for()
    if screenshots:
        screenshots.mkdir(exist_ok=True, parents=True)
        page.screenshot(path=screenshots / "03-workspace-desktop.png")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth") <= 392
    layout = page.locator(".work-area").first
    assert layout.evaluate("(element) => getComputedStyle(element).flexDirection") == "column"
    if screenshots:
        page.screenshot(path=screenshots / "04-workspace-mobile.png")
    page.set_viewport_size({"width": 1400, "height": 960})
    page.get_by_role("button", name="Process images").click()
    page.get_by_text("Choose input media to start.", exact=True).wait_for()
    page.locator('input[type=file]').first.set_input_files([root / "day-1.png", root / "day-2.png", root / "corrupt.png"])
    page.get_by_text("day-1.png", exact=True).first.wait_for()
    page.get_by_role("button", name="Process images").click()
    page.get_by_text("2 of 3 images processed.", exact=False).wait_for(timeout=120000)
    page.get_by_text("corrupt.png: Cannot decode image.", exact=True).wait_for()
    page.get_by_role("textbox", name="Search").fill("no_match_for_this_query")
    page.get_by_text("0 visible of", exact=False).first.wait_for()
    page.get_by_role("textbox", name="Search").fill("")
    page.get_by_text("Download results", exact=True).first.click()
    archive_link = page.locator('a[href*="batch.zip"]').first
    with page.expect_download() as download:
        archive_link.click()
    assert download.value.suggested_filename == "batch.zip"
    print("Image validation, partial batch, filter and download passed.")

    def video_panel():
        page.get_by_role("tab", name="Recorded video", exact=True).click()
        return page.get_by_role("tabpanel").filter(has=page.get_by_role("button", name="Process video", exact=True))

    panel = video_panel()
    panel.locator("input[type=file]").first.set_input_files(root / "traffic.mp4")
    panel.locator("video").first.wait_for(state="attached")
    panel.get_by_role("button", name="Process video").click()
    panel.get_by_text("4 decoded · 4 analyzed", exact=False).wait_for(timeout=120000)
    panel.get_by_role("tab", name="Vehicle tracks").click()
    panel.get_by_text("Track summaries", exact=True).first.wait_for()
    print("Video preview and vehicle tracks passed.")

    page.reload(wait_until="domcontentloaded")
    panel = video_panel()
    panel.locator("input[type=file]").first.set_input_files(root / "cancel.mp4")
    panel.locator("video").first.wait_for(state="attached")
    panel.get_by_role("button", name="Process video").click()
    panel.get_by_text("Starting analysis.", exact=True).wait_for(timeout=120000)
    page.wait_for_timeout(6000)
    panel.get_by_role("button", name="Stop", exact=True).click()
    panel.get_by_text("Cancelled", exact=True).wait_for(timeout=120000)
    assert not errors, errors
    print("Active cancellation passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--browser-exe", type=Path)
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temporary, sync_playwright() as playwright:
        root = Path(temporary)
        fixtures(root)
        options = {"executable_path": args.browser_exe} if args.browser_exe else {}
        with playwright.chromium.launch(**options) as browser:
            check_ui(browser, args.url, root, args.screenshots)


if __name__ == "__main__":
    main()
