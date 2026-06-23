from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_config import (
    INDEX_PATH,
    OUT_JSONL,
    OUT_XLSX,
    PDF_PATH,
    PRODUCTS_DIR,
    REGENERATE_STAGE1,
)


BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    script_path = BASE_DIR / script_name
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main():
    print("Bank extractor starting...")
    print(f"Project dir: {BASE_DIR}")
    print(f"PDF_PATH: {PDF_PATH}")
    print(f"PRODUCTS_DIR: {PRODUCTS_DIR}")
    print(f"INDEX_PATH: {INDEX_PATH}")
    print(f"OUT_JSONL: {OUT_JSONL}")
    print(f"OUT_XLSX: {OUT_XLSX}")

    if REGENERATE_STAGE1:
        print("Stage 1 regeneration requested via REGENERATE_STAGE1=true.")
        if PDF_PATH.exists():
            print("Stage 1: PDF found, segmenting products.")
            run_step("01_segment_products.py")
        else:
            print(f"Skipping 01_segment_products.py because PDF was not found at {PDF_PATH}")
            print("Stage 1 skipped because the PDF is missing.")
            print("If you want to regenerate the product chunks, set PDF_PATH in .env to the real PDF location and add the PDF there.")
            print("If you already have products/ and products_index.json, the pipeline can continue from stage 2.")
    else:
        print("Stage 1 skipped because REGENERATE_STAGE1 is false.")
        print("Using existing products/ and products_index.json.")

    if not INDEX_PATH.exists():
        raise SystemExit(
            f"products_index.json not found at {INDEX_PATH}. Run stage 1 first or restore the file."
        )

    if not PRODUCTS_DIR.exists():
        raise SystemExit(
            f"products folder not found at {PRODUCTS_DIR}. Run stage 1 first or restore the files."
        )

    print("Stage 2: extracting fields with the local Hugging Face model.")
    run_step("02_extract_fields.py")
    print("Stage 3: building the Excel workbook.")
    run_step("03_build_excel.py")

    print(f"\nDone.\nJSONL: {OUT_JSONL}\nXLSX: {OUT_XLSX}")


if __name__ == "__main__":
    main()
