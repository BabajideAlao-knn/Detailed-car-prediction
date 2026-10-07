"""
Autochek Nigeria car-listings scraper
https://autochek.africa/ng/cars-for-sale

Each listing page is a Next.js page that embeds its data as JSON in
<script id="__NEXT_DATA__">. The scraper reads that JSON (much more reliable than
parsing the HTML cards) and walks the pages with ?page_number=N (23 cars per page).

Install:  pip install requests
Run:      python scraper/autochek_scraper.py                 # all pages -> data/autochek_cars_ng.csv
          python scraper/autochek_scraper.py --max-pages 5   # quick test
          python scraper/autochek_scraper.py --start-url https://autochek.africa/ng/cars-for-sale/toyota

Be polite: the default delay is ~1.5 s between requests. The listing pages are allowed
by the site's robots.txt; its separate API host is not, so this scraper only reads the public pages.
Check the site's terms of service before any commercial use.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

BASE = "https://autochek.africa"
DEFAULT_START = f"{BASE}/ng/cars-for-sale"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "autochek_cars_ng.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>', re.S)

COLUMNS = ["id", "title", "year", "make", "model", "price_ngn", "old_price_ngn", "monthly_installment_ngn",
           "loan_value_ngn", "has_financing", "condition", "mileage", "mileage_unit", "transmission", "fuel_type",
           "engine_type", "engine_cc", "body_type_id", "state", "city", "grade_score", "listing_score", "accidented",
           "inspected", "has_warranty", "is_featured", "sold", "listed_date", "url", "image_url"]


def page_url(start_url: str, page: int) -> str:
    parts = urlparse(start_url)
    qs = parse_qs(parts.query)
    qs["page_number"] = [str(page)]
    return urlunparse(parts._replace(query=urlencode(qs, doseq=True)))


def parse_page(html: str) -> dict:
    """Return the `cars` object ({result: [...], pagination: {...}}) from a listing page."""
    m = NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("__NEXT_DATA__ block not found (site layout may have changed)")
    return json.loads(m.group(1))["props"]["pageProps"]["cars"]


def split_title(title: str, known_makes: list[str]) -> tuple[str, str]:
    """'Mercedes-Benz GLE-Class' -> ('Mercedes-Benz', 'GLE-Class') using the site's list of makes."""
    t = (title or "").strip()
    for name in known_makes:  # longest names first
        if t.lower() == name.lower() or t.lower().startswith(name.lower() + " "):
            return t[:len(name)], t[len(name):].strip()
    first, _, rest = t.partition(" ")
    return first, rest


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def to_row(car: dict, known_makes: list[str]) -> dict:
    make, model = split_title(car.get("title"), known_makes)
    return {
        "id": car.get("id"), "title": car.get("title"), "year": car.get("year"), "make": make, "model": model,
        "price_ngn": car.get("marketplacePrice"), "old_price_ngn": car.get("marketplaceOldPrice"),
        "monthly_installment_ngn": car.get("installment"), "loan_value_ngn": car.get("loanValue"),
        "has_financing": car.get("hasFinancing"), "condition": car.get("sellingCondition"),
        "mileage": car.get("mileage"), "mileage_unit": car.get("mileageUnit"),
        "transmission": car.get("transmission"), "fuel_type": car.get("fuelType"),
        "engine_type": car.get("engineType"), "engine_cc": car.get("ccMeasurement") or car.get("engineDisplacement"),
        "body_type_id": car.get("bodyTypeId"), "state": car.get("state"), "city": car.get("city"),
        "grade_score": car.get("gradeScore"), "listing_score": car.get("listingScore"),
        "accidented": car.get("accidented"), "inspected": car.get("inspected"),
        "has_warranty": car.get("hasWarranty"), "is_featured": car.get("isFeatured"), "sold": car.get("sold"),
        "listed_date": car.get("marketplaceVisibleDate"),
        "url": f"{BASE}/ng/cars-for-sale/{slug(make)}/{slug(model)}/{car.get('id')}",
        "image_url": car.get("imageUrl"),
    }


class Client:
    def __init__(self, delay: float = 1.5, timeout: int = 30):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.delay, self.timeout = delay, timeout

    def get(self, url: str, retries: int = 3) -> str:
        for attempt in range(1, retries + 1):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.text
            except Exception as e:  # noqa: BLE001
                wait = self.delay * 2 * attempt
                print(f"  ! {e} (attempt {attempt}/{retries}); retrying in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url}")

    def pause(self):
        time.sleep(self.delay + random.uniform(0, self.delay / 2))


def scrape(start_url: str = DEFAULT_START, max_pages: int = 0, delay: float = 1.5) -> list[dict]:
    client = Client(delay)
    html = client.get(page_url(start_url, 1))
    first = parse_page(html)
    pp = json.loads(NEXT_DATA_RE.search(html).group(1))["props"]["pageProps"]
    makes = pp.get("makes") or []
    makes = makes if isinstance(makes, list) else (makes.get("makeList") or makes.get("result") or [])
    known_makes = sorted({m.get("name") for m in makes if m.get("name")}, key=len, reverse=True)

    pag = first["pagination"]
    pages = -(-pag["total"] // pag["pageSize"])
    if max_pages:
        pages = min(pages, max_pages)
    print(f"{pag['total']:,} listings across {pages} pages")

    rows = {c["id"]: to_row(c, known_makes) for c in first.get("result", [])}
    failed = []
    for p in range(2, pages + 1):
        client.pause()
        try:
            for c in parse_page(client.get(page_url(start_url, p))).get("result", []):
                rows[c["id"]] = to_row(c, known_makes)
        except Exception as e:  # noqa: BLE001
            failed.append(p)
            print(f"  ! page {p}: {e}", file=sys.stderr)
        if p % 25 == 0 or p == pages:
            print(f"page {p}/{pages}: {len(rows):,} listings")
    if failed:
        print(f"{len(failed)} pages failed: {failed}", file=sys.stderr)
    return list(rows.values())


def save(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows):,} rows to {path}")


def main():
    ap = argparse.ArgumentParser(description="Scrape Autochek Nigeria car listings")
    ap.add_argument("--start-url", default=DEFAULT_START, help="Listing URL, can be filtered (e.g. .../cars-for-sale/toyota)")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    ap.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    save(scrape(args.start_url, args.max_pages, args.delay), args.out)


if __name__ == "__main__":
    main()
