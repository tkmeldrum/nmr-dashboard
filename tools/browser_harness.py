"""A small harness for checking the built site in a real (headless) browser, with no local web server and no network.

    pip install playwright && playwright install chromium        # once, in any environment (not a project dependency)
    python tools/fetch_vendor.py                                  # once: the libraries the site serves itself

    from browser_harness import browser_page
    with browser_page() as (pg, blocked, errors):
        pg.click("#sample-btn"); ...                              # drive the page like a user
    assert not blocked and not errors

It builds the site into a temporary folder, answers every request for https://nmrdash.test/ from it, and REFUSES every other request (recorded in `blocked`),
so a stray outside connection fails loudly. Page errors and console errors are collected in `errors`. "Explore test data" loads the bundled samples; for a
folder-write path use an Origin Private File System directory as a stand-in for a data folder (navigator.storage.getDirectory()).
Notes: the Results sections start collapsed, so wait with state="attached" for things inside them; wait_for_function returns at once if its condition already holds.
"""
import contextlib
import mimetypes
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


@contextlib.contextmanager
def browser_page(width=1300, height=1000, timeout_ms=120000, accept_downloads=True, wait_ready=True):
    from build_site import build
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        site = build(Path(tmp) / "site", "harness")
        blocked, errors = [], []

        def handle(route):
            url = route.request.url
            if url.startswith("https://nmrdash.test/"):
                rel = url.split("nmrdash.test/", 1)[1].split("?")[0] or "index.html"
                f = site / rel
                if not f.is_file():
                    return route.fulfill(status=404, body=b"")
                ct = mimetypes.guess_type(str(f))[0] or ("application/wasm" if rel.endswith(".wasm") else "application/octet-stream")
                return route.fulfill(status=200, body=f.read_bytes(), content_type=ct)
            blocked.append(url)
            return route.abort()

        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": width, "height": height}, accept_downloads=accept_downloads)
        ctx.route("**/*", handle)
        pg = ctx.new_page()
        pg.set_default_timeout(timeout_ms)
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("https://nmrdash.test/")
        if wait_ready:
            pg.wait_for_function("document.getElementById('status').textContent === 'Python ready.'", timeout=300000)
        try:
            yield pg, blocked, errors
        finally:
            browser.close()
