from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "ui_screenshots"
VIDEO_PATH = PROJECT_ROOT / "docs" / "jury_demo.mp4"


def capture_screenshots(base_url: str) -> list[Path]:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    shots = [
        ("01_home.png", "/"),
        ("02_demo_ready.png", "/demo"),
        ("04_methodology.png", "/methodology"),
        ("05_counter_uav.png", "/counter_uav"),
        ("06_roadmap.png", "/roadmap"),
        ("07_about.png", "/about"),
    ]
    created: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1.25)

        for filename, route in shots:
            page.goto(f"{base_url}{route}", wait_until="networkidle")
            page.screenshot(path=str(SCREENSHOTS_DIR / filename), full_page=True)
            created.append(SCREENSHOTS_DIR / filename)

        page.goto(f"{base_url}/demo", wait_until="networkidle")
        page.get_by_role("button", name=re.compile(r"Run cascade|Запустить каскад")).click()
        try:
            page.wait_for_function(
                "() => document.body.innerText.includes('Done in') || document.body.innerText.includes('Готово:') || document.body.innerText.includes('Error:') || document.body.innerText.includes('Ошибка:')",
                timeout=90_000,
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError("Timed out while waiting for demo inference result") from exc
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_demo_result.png"), full_page=True)
        created.append(SCREENSHOTS_DIR / "03_demo_result.png")
        browser.close()

    return sorted(created)


def build_video(images: list[Path], seconds_per_slide: int = 3) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not available on PATH")

    list_file = SCREENSHOTS_DIR / "jury_demo_slides.txt"
    lines: list[str] = []
    for image in images:
        lines.append(f"file '{image.as_posix()}'")
        lines.append(f"duration {seconds_per_slide}")
    lines.append(f"file '{images[-1].as_posix()}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=25,format=yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_PATH),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        list_file.unlink(missing_ok=True)
    return VIDEO_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture UI screenshots and build jury demo backup video.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Base URL of the running UI server")
    parser.add_argument("--skip-video", action="store_true", help="Only capture screenshots")
    args = parser.parse_args()

    images = capture_screenshots(args.base_url)
    print(f"Captured {len(images)} screenshots in {SCREENSHOTS_DIR}")
    if not args.skip_video:
        video = build_video(images)
        print(f"Built backup video: {video}")


if __name__ == "__main__":
    main()
