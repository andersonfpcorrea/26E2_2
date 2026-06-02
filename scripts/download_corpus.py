"""Download the BBC News corpus and consolidate it into a single CSV.

Source: D. Greene and P. Cunningham, "Practical Solutions to the Problem of
Diagonal Dominance in Kernel Document Clustering", ICML 2006.
Dataset page: http://mlg.ucd.ie/datasets/bbc.html

The raw distribution is a zip of one folder per category, each holding plain
.txt articles. This script downloads it, parses every article, and writes
``data/bbc_news.csv`` with columns ``id``, ``category``, ``title``, ``text``.
The CSV is committed so the notebook needs no network access.
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

CORPUS_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "bbc_news.csv"


def fetch_zip(url: str) -> zipfile.ZipFile:
    """Download the corpus archive into memory and return it as a ZipFile."""
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    return zipfile.ZipFile(io.BytesIO(payload))


def parse_article(raw: str) -> tuple[str, str]:
    """Split a raw article into (title, body).

    BBC articles store the headline on the first non-empty line and the body in
    the remaining paragraphs.
    """
    lines = [line.strip() for line in raw.splitlines()]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return "", ""
    title = non_empty[0]
    body = " ".join(non_empty[1:])
    return title, body


def build_csv() -> None:
    """Download, parse, and write the consolidated corpus CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = fetch_zip(CORPUS_URL)

    rows: list[tuple[int, str, str, str]] = []
    next_id = 0
    for name in sorted(archive.namelist()):
        # Entries look like "bbc/business/001.txt"; skip dirs and the README.
        if not name.endswith(".txt"):
            continue
        parts = name.split("/")
        if len(parts) != 3:
            continue
        category = parts[1]
        raw_bytes = archive.read(name)
        # The corpus files are UTF-8; decode tolerantly to absorb stray bytes.
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        title, body = parse_article(raw_text)
        if not body:
            continue
        full_text = f"{title}. {body}" if title else body
        rows.append((next_id, category, title, full_text))
        next_id += 1

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "title", "text"])
        writer.writerows(rows)

    categories: dict[str, int] = {}
    for _, category, _, _ in rows:
        categories[category] = categories.get(category, 0) + 1

    print(f"Wrote {len(rows)} documents to {OUTPUT_CSV}")
    for category in sorted(categories):
        print(f"  {category}: {categories[category]}")


if __name__ == "__main__":
    try:
        build_csv()
    except Exception as error:  # noqa: BLE001 - surface any failure to the CLI
        print(f"Failed to build corpus CSV: {error}", file=sys.stderr)
        sys.exit(1)
