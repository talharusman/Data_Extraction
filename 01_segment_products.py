"""
Step 1: Segment the merged PDF into individual product chunks.

Detects product titles using font metadata (PyMuPDF): in this document,
product titles are set in bold ~14pt font, while body text is ~12pt and
section sub-headers (e.g. "Eligibility Criteria") are bold ~12pt or ~11pt.
This is far more reliable than text heuristics on a messy multi-bank PDF.

Output: products/<NNN>_<slug>.txt  (raw text per product)
        products_index.json        (title, start/end page)
"""
from __future__ import annotations

import json
import os
import re

import fitz

from pipeline_config import INDEX_PATH, PDF_PATH, PRODUCTS_DIR

TITLE_MIN_SIZE = 13.5
BOLD_FLAG = 16


def slugify(text, max_len=60):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return s[:max_len] or "product"


def find_titles(doc):
    """Return list of (page_index_0based, title_text) for each detected title."""
    titles = []
    for pno in range(len(doc)):
        page = doc[pno]
        d = page.get_text("dict")
        for block in d["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                title_spans = [
                    s["text"]
                    for s in line["spans"]
                    if s["size"] >= TITLE_MIN_SIZE and (s["flags"] & BOLD_FLAG)
                ]
                if title_spans:
                    text = "".join(title_spans).strip()
                    if text and len(text) > 2:
                        titles.append((pno, text))

    merged = []
    for pno, text in titles:
        if merged and merged[-1][0] == pno and merged[-1][1] == text:
            continue
        merged.append((pno, text))
    return merged


def main():
    if not PDF_PATH.exists():
        raise SystemExit(
            f"PDF not found: {PDF_PATH}\n"
            "Set PDF_PATH in .env or copy final_report_merged.pdf into the project folder."
        )

    os.makedirs(PRODUCTS_DIR, exist_ok=True)
    doc = fitz.open(str(PDF_PATH))
    titles = find_titles(doc)
    print(f"Detected {len(titles)} product titles across {len(doc)} pages")

    index = []
    for i, (pno, title) in enumerate(titles):
        start_page = pno
        end_page = (titles[i + 1][0] - 1) if i + 1 < len(titles) else (len(doc) - 1)
        index.append(
            {
                "product_no": i + 1,
                "title": title,
                "start_page": start_page + 1,
                "end_page": end_page + 1,
            }
        )

    for entry in index:
        text_parts = []
        for pno in range(entry["start_page"] - 1, entry["end_page"]):
            text_parts.append(doc[pno].get_text("text"))
        full_text = "\n".join(text_parts).strip()
        fname = f"{entry['product_no']:03d}_{slugify(entry['title'])}.txt"
        entry["file"] = fname
        with open(PRODUCTS_DIR / fname, "w", encoding="utf-8") as f:
            f.write(full_text)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(index)} product files to {PRODUCTS_DIR}")
    print(f"Index written to {INDEX_PATH}")


if __name__ == "__main__":
    main()
