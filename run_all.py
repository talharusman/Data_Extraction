from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_config import INDEX_PATH, OUT_JSONL, OUT_XLSX, PDF_PATH, PRODUCTS_DIR


BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    script_path = BASE_DIR / script_name
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main():
    if PDF_PATH.exists():
        run_step("01_segment_products.py")
    else:
        print(f"Skipping 01_segment_products.py because PDF was not found at {PDF_PATH}")
        print("If you want to regenerate the product chunks, set PDF_PATH in .env and add the PDF.")

    if not INDEX_PATH.exists():
        raise SystemExit(
            f"products_index.json not found at {INDEX_PATH}. Run stage 1 first or restore the file."
        )

    if not PRODUCTS_DIR.exists():
        raise SystemExit(
            f"products folder not found at {PRODUCTS_DIR}. Run stage 1 first or restore the files."
        )

    run_step("02_extract_fields.py")
    run_step("03_build_excel.py")

    print(f"\nDone.\nJSONL: {OUT_JSONL}\nXLSX: {OUT_XLSX}")


if __name__ == "__main__":
    main()
