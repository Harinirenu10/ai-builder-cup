"""The customer /app PWA, in a real browser, against the real built frontend and a real running
API -- the counterpart to test_widget_e2e.py's store/widget coverage, which never drives /app
itself (see the audit report this fixes).

What is REAL here: the Vite production build of web/, served with SPA routing (so client-side
routes like /app resolve), a real `uvicorn` server running the actual services/api/main.py app in
this same process, real HTTP requests from the browser (Authorization headers, CORS, 401/403
handling), the real demo sign-in flow (web/src/lib/auth.tsx), and the real authenticated-Blob GLB
fetch (web/src/lib/api.ts::loadObjectUrl) rendering a real, validly-exported GLB
(services/geometry/exporter.py::export_twin_glb) in <model-viewer>.

What is MOCKED: services/geometry/worker.py::process_twin_job (the MediaPipe pose extraction +
SMPL optimization pipeline) is replaced with a fast, deterministic fake that writes a twin
document shaped like the real one. This proves the real browser UI drives a real end-to-end HTTP
request and handles the real response shape correctly; it does NOT claim the geometry pipeline
itself was exercised live from a browser -- that is services/geometry's own test suite's job, and
running the real pipeline from a browser test would make this slow and environment-dependent
(MediaPipe/SMPL availability) for no additional UI coverage.

Runs in Chromium and WebKit; needs `playwright install chromium webkit` and `npm install` in
web/. Skips what is missing, same as test_widget_e2e.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import API_DIR, ROOT
from tests.test_widget_e2e import WEB, browser_installed, free_port, serve

TWIN_ZONES = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]

# A minimal valid JPEG (same bytes already used by services/api/tests/test_api_twin.py), written
# to real temp files so Playwright can upload them through the real file inputs.
_DUMMY_JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01,
                     0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9])


@pytest.fixture(scope="module")
def app_stack(tmp_path_factory):
    if not shutil.which("node") or not (WEB / "node_modules").is_dir():
        pytest.skip("Node and web/node_modules are needed to build /app (run npm install in web/)")

    import uvicorn

    from services.geometry.cloud_client import InMemoryFirestoreClient, InMemoryStorageClient, set_cloud_clients
    from services.geometry.exporter import export_twin_glb
    from services.geometry.loader import load_smpl

    api_port, app_port = free_port(), free_port()
    api_url, app_url = f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{app_port}"
    tmp = tmp_path_factory.mktemp("app_e2e")

    # 1. The real customer app, built with dev sign-in and this test's API address -- the same
    # build test_widget_e2e.py's `stack` fixture already proves works for /app/connect.
    dist = tmp / "app"
    env = {**os.environ, "VITE_AUTH_MODE": "dev", "VITE_API_URL": api_url}
    npx = shutil.which("npx") or "npx"
    build = subprocess.run(
        [npx, "vite", "build", "--outDir", str(dist), "--emptyOutDir"],
        cwd=WEB, env=env, capture_output=True, text=True, timeout=300,
    )
    assert build.returncode == 0, build.stdout[-800:] + build.stderr[-800:]

    # 2. Cloud clients: in-memory, shared with the fake process_twin_job below.
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient()
    set_cloud_clients(firestore_client=fs, storage_client=st)

    # A real, validly-exported GLB (not placeholder bytes) via the real exporter on the real
    # neutral SMPL template, so <model-viewer> has something genuinely parseable to render.
    model = load_smpl("neutral")
    glb_bytes = export_twin_glb(vertices=model().vertices[0].detach().numpy(), faces=model.faces)
    glb_url = st.upload_bytes("e2e/twin.glb", glb_bytes, content_type="model/gltf-binary")

    # A twin that already exists for the demo shopper, for the twin-viewer/GLB test below (so
    # that test doesn't have to go through the capture form first).
    fs.create_twin("twin_preexisting", {
        "id": "twin_preexisting", "user_id": "demo-shopper", "gender": "neutral", "height_cm": 175.0,
        "betas": [0.0] * 10, "measurements": {"zones": TWIN_ZONES}, "glb_storage_url": glb_url,
        "final_mean_iou": 0.91, "created_at": "2026-01-01T00:00:00Z",
    })

    # 3. The real API, with process_twin_job mocked (see module docstring for why).
    os.environ.update({"AUTH_MODE": "dev", "CONSENT_JWT_SECRET": "e2e-secret-" + "z" * 24, "API_CORS_ORIGINS": app_url})
    for v in ("GOOGLE_CLOUD_PROJECT", "LLM_BACKEND"):
        os.environ.pop(v, None)
    sys.path.insert(0, str(API_DIR))
    spec = importlib.util.spec_from_file_location("api_main_app_e2e", API_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def fake_process_twin_job(job_id, firestore_client):
        job = firestore_client.get_job(job_id)
        twin_id = f"twin_{job_id}"
        firestore_client.create_twin(twin_id, {
            "id": twin_id, "user_id": job["user_id"], "gender": job.get("gender", "neutral"),
            "height_cm": job.get("height_cm", 175.0), "betas": [0.0] * 10,
            "measurements": {"zones": TWIN_ZONES}, "glb_storage_url": glb_url,
            "final_mean_iou": 0.91, "created_at": "2026-01-01T00:00:00Z",
        })
        firestore_client.update_job(job_id, {"status": "done", "twin_id": twin_id})
        return {"status": "done", "job_id": job_id, "twin_id": twin_id, "final_mean_iou": 0.91}

    mod.process_twin_job = fake_process_twin_job

    server = uvicorn.Server(uvicorn.Config(mod.app, host="127.0.0.1", port=api_port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(api_url + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("API did not start")

    app_srv = serve(dist, app_port, spa=True)
    yield {"api": api_url, "app": app_url, "fs": fs, "st": st}
    server.should_exit = True
    app_srv.shutdown()


@pytest.fixture(params=["chromium", "webkit"])
def page(request, app_stack, monkeypatch):
    from playwright.sync_api import sync_playwright

    monkeypatch.setenv("API_CORS_ORIGINS", app_stack["app"])
    name = request.param
    if not browser_installed(name):
        pytest.skip(f"Playwright {name} is not installed")
    with sync_playwright() as p:
        browser = getattr(p, name).launch()
        ctx = browser.new_context()
        pg = ctx.new_page()
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on(
            "response",
            lambda r: pg.errors.append(f"{r.status} {r.url}")
            if r.status >= 400 and "favicon" not in r.url and "googleapis.com" not in r.url
            else None,
        )
        yield pg
        browser.close()


def sign_in_as_demo_shopper(page) -> None:
    # The Navbar (web/src/components/Navbar.tsx, data-testid="auth-controls") also shows a
    # "Demo (demo-shopper)" badge once signed in, in addition to CustomerApp.tsx's own "User:
    # Demo (demo-shopper)" badge -- matching the bare text is ambiguous, so wait for the one
    # specific to CustomerApp.tsx's own UI.
    page.get_by_role("button", name="Sign in as Demo Shopper").click()
    page.get_by_text("User: Demo (demo-shopper)").wait_for(timeout=10000)


def test_demo_sign_in_shows_signed_in_state_and_persists_across_reload(app_stack, page):
    page.goto(f"{app_stack['app']}/app")
    page.get_by_role("button", name="Sign in as Demo Shopper").wait_for()
    sign_in_as_demo_shopper(page)

    page.reload()
    page.get_by_text("User: Demo (demo-shopper)").wait_for(timeout=10000)
    # Signing in again is not offered once already signed in.
    assert page.get_by_role("button", name="Sign in as Demo Shopper").count() == 0


def test_twin_viewer_loads_and_renders_a_real_glb_for_the_signed_in_user(app_stack, page):
    page.goto(f"{app_stack['app']}/app")
    sign_in_as_demo_shopper(page)

    model_viewer = page.locator("model-viewer")
    model_viewer.wait_for(timeout=15000)
    src = model_viewer.get_attribute("src")
    assert src is not None and src.startswith("blob:"), (
        f"model-viewer's src must be an authenticated in-memory blob URL (loadObjectUrl), not a bare "
        f"API URL model-viewer would fetch itself with no Authorization header. Got: {src!r}"
    )
    assert "3D Model Unavailable" not in page.content()


def test_four_photo_capture_submits_and_shows_the_mocked_pipelines_result(app_stack, page, tmp_path):
    page.goto(f"{app_stack['app']}/app")
    sign_in_as_demo_shopper(page)

    page.get_by_role("button", name="4-Photo Capture").click()
    page.locator("#height-cm").fill("168.0")
    page.get_by_text("I confirm that I am 18 years of age or older").click()

    photo_path = tmp_path / "view.jpg"
    photo_path.write_bytes(_DUMMY_JPEG)
    file_inputs = page.locator('input[type="file"]')
    assert file_inputs.count() == 4
    for i in range(4):
        file_inputs.nth(i).set_input_files(str(photo_path))

    page.get_by_role("button", name="Process 3D Twin via Geometry Service").click()
    page.get_by_text("3D Body Twin created successfully!").wait_for(timeout=15000)

    page.get_by_role("button", name="My Measurements").click()
    # TWIN_ZONES includes chest; the mocked pipeline's twin carries it through to the UI.
    page.get_by_text("Chest").first.wait_for(timeout=5000)

    assert not page.errors, f"Unexpected browser errors: {page.errors}"


def test_twin_viewer_shows_the_fresh_twin_right_after_capture_no_reload(app_stack, page, tmp_path):
    """Regression test for a real reported bug: after a fresh 4-photo capture completes, the UI
    showed '3D Body Twin created successfully!' and updated measurements, but the 3D Twin Viewer
    tab still showed 'No 3D Twin Yet' (or, as reproduced here, the WRONG -- stale, pre-existing --
    twin) without a page reload.

    Root cause confirmed by this exact test before the fix: app_stack seeds a pre-existing twin
    for demo-shopper (for the separate test above) with a hardcoded created_at. The mocked
    process_twin_job used the same hardcoded created_at string for the newly-captured twin.
    InMemoryFirestoreClient.get_latest_twin_for_user's `max(..., key=lambda t: t['created_at'])`
    broke that tie by returning whichever twin happened to come first in dict iteration (the
    OLD, pre-existing one) -- so GET /v1/me/twin, and therefore the Twin Viewer, kept showing the
    old twin's data even though the capture genuinely produced a new one. The frontend's refetch
    logic (CustomerApp.tsx) was already correct; this was a backend tie-break bug.
    """
    page.goto(f"{app_stack['app']}/app")
    sign_in_as_demo_shopper(page)

    # Record whichever twin is shown BEFORE this test's own capture (the module-scoped app_stack
    # fixture is shared with other tests in this file, each of which may have already captured
    # their own twin for demo-shopper, so this is not necessarily twin_preexisting specifically --
    # the point is only that it must DIFFER from the twin this test is about to create itself).
    page.get_by_role("button", name="3D Twin Viewer").click()
    page.locator("model-viewer").wait_for(timeout=15000)
    twin_label = page.get_by_text("Twin:", exact=False)
    twin_label.wait_for(timeout=5000)
    twin_id_before = twin_label.inner_text()

    page.get_by_role("button", name="4-Photo Capture").click()
    page.locator("#height-cm").fill("168.0")
    page.get_by_text("I confirm that I am 18 years of age or older").click()

    photo_path = tmp_path / "view.jpg"
    photo_path.write_bytes(_DUMMY_JPEG)
    file_inputs = page.locator('input[type="file"]')
    for i in range(4):
        file_inputs.nth(i).set_input_files(str(photo_path))

    page.get_by_role("button", name="Process 3D Twin via Geometry Service").click()
    page.get_by_text("3D Body Twin created successfully!").wait_for(timeout=15000)

    # The actual regression: switch to the Twin Viewer WITHOUT reloading the page. It must show
    # the model-viewer with the NEW twin -- not "No 3D Twin Yet", and not the twin that was
    # showing before this capture.
    page.get_by_role("button", name="3D Twin Viewer").click()
    page.locator("model-viewer").wait_for(timeout=15000)
    assert "No 3D Twin Yet" not in page.content()
    twin_id_after = page.get_by_text("Twin:", exact=False).inner_text()
    assert twin_id_after != twin_id_before, (
        f"Twin Viewer still shows the twin from before this capture ({twin_id_before!r}), "
        "not the one just created"
    )
    assert not page.errors, f"Unexpected browser errors: {page.errors}"
