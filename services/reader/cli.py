"""Reader CLI. Runs on saved pages in samples/products/ with recorded LLM responses, no cloud needed.

    python cli.py --compare                                  # all samples: extracted vs true chart values (offline)
    python cli.py samples/products/demo_threads_tee.html     # one saved page (offline if a fixture exists)
    python cli.py --record [--only NAME]                     # call the live LLM and save fixtures (LLM_BACKEND from .env)
    python cli.py --store --compare | --record               # the same for the six mock storefront pages in web/store
    python cli.py --url https://... --save-html samples/products/name.html   # visit one public page once, save rendered HTML
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fitcore.read.normalize import assemble  # noqa: E402
from readerlib import capture as cap_mod  # noqa: E402
from readerlib.cache import url_key  # noqa: E402
from readerlib.compare import SAMPLES, STORE, compare_chart, format_table, load_truth, page_for  # noqa: E402
from readerlib.pipeline import _extract_from_capture  # noqa: E402
from readerlib.replay import ReplayClient, fixture_path, use_offline_env  # noqa: E402


def read_file(path: Path, live: bool, label: str | None = None):
    """Capture a saved page through Chromium, extract (recorded or live), assemble."""
    url = path.resolve().as_uri()
    cap = cap_mod.capture_page(url)
    if cap.status != "ok":
        raise SystemExit(f"{path.name}: {cap.status} {cap.reason}")
    client = None
    if not live:
        fx = fixture_path(label or path.stem)
        if not fx.is_file():
            raise SystemExit(f"No recorded response for {path.stem}. Run with --record first, or pass --live.")
        use_offline_env()
        client = ReplayClient(fx)
    ext, _chart_from_html = _extract_from_capture(cap, url, client, None)
    return ext, assemble(ext, url=url, product_id=url_key(url)[:16], fallback_brand=path.stem, image_urls=cap.image_urls)


def cmd_compare(only: str | None, directory: Path) -> int:
    truth = load_truth(directory)
    bad = 0
    for name, t in truth.items():
        if only and name != only:
            continue
        ext, out = read_file(page_for(directory, name), live=False, label=name)
        if out.status != "ok" or out.product is None:
            print(f"== {name} ==\nstatus {out.status}")
            bad += 1
            continue
        rows = compare_chart(out.product, t)
        print(format_table(name, out.product, t, rows), end="\n\n")
        bad += sum(not r["ok"] for r in rows)
        rv = out.reviews
        if rv:
            signals = ", ".join(f"{s.zone.value}/{s.direction} x{s.mention_count}" for s in rv.zone_signals) or "none"
            print(f"reviews read {rv.reviews_read} (true {t['reviews_read']}); signals: {signals}; "
                  f"shrinkage {rv.shrinkage_mentions}, colour {rv.color_mismatch_mentions}, sheer {rv.sheerness_mentions}")
            print(f"true review counts: {t['review_counts']}\n")
    return 1 if bad else 0


def cmd_record(only: str | None, directory: Path) -> None:
    for name in load_truth(directory):
        if only and name != only:
            continue
        ext, out = read_file(page_for(directory, name), live=True, label=name)
        fixture_path(name).write_text(ext.model_dump_json(indent=2), encoding="utf-8")
        print(f"recorded {fixture_path(name).name}: status {out.status}")


def cmd_url(url: str, save_html: str | None) -> None:
    cap = cap_mod.capture_page(url)
    print(f"status {cap.status} {cap.reason or ''} (http {cap.http_status}), chart opened: {cap.chart_opened}, text chars: {len(cap.text)}")
    if cap.status == "ok" and save_html:
        html = re.sub(r"(?is)<script(?![^>]*ld\+json).*?</script>", "", cap.html)
        Path(save_html).write_text(html, encoding="utf-8")
        print(f"saved rendered HTML (scripts removed, JSON-LD kept) to {save_html}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--store", action="store_true", help="use the mock storefront in web/store instead of samples/products")
    ap.add_argument("--live", action="store_true", help="use the live LLM instead of the recorded response")
    ap.add_argument("--url")
    ap.add_argument("--save-html")
    a = ap.parse_args()
    if a.compare:
        raise SystemExit(cmd_compare(a.only, STORE if a.store else SAMPLES))
    if a.record:
        return cmd_record(a.only, STORE if a.store else SAMPLES)
    if a.url:
        return cmd_url(a.url, a.save_html)
    for f in a.files:
        _, out = read_file(Path(f), a.live)
        print(json.dumps({"status": out.status, "flags": out.flags,
                          "product": out.product.model_dump(mode="json") if out.product else None,
                          "reviews": out.reviews.model_dump(mode="json") if out.reviews else None},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
