from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.cli import prompt_selection
from app.config import load_config
from app.logging_config import setup_logging
from app.pipeline import BankingExtractionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process all supported banking product documents in a root folder. "
            "Files in the root folder and all nested folders are included automatically."
        )
    )
    parser.add_argument(
        "root_folder",
        nargs="?",
        help="Root folder address. Example: C:\\BankDocuments",
    )
    parser.add_argument("--folder", help="Root folder containing nested banking product documents")
    parser.add_argument("--file", help="Single product document to process")
    args = parser.parse_args()

    mode = "folder"
    target_path = args.folder or args.root_folder or args.file
    if args.file:
        mode = "file"
    elif args.folder or args.root_folder:
        mode = "nested_folder"
    else:
        mode, target_path = prompt_selection()

    if not target_path:
        parser.error("Provide a file or folder path")

    target = Path(target_path)
    if mode == "file" and not target.is_file():
        parser.error(f"File does not exist: {target_path}")
    if mode != "file" and not target.is_dir():
        parser.error(f"Folder does not exist: {target_path}")

    config = load_config()
    setup_logging(config.get("app", "log_file"))
    try:
        pipeline = BankingExtractionPipeline(config)
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    if mode == "file":
        product = pipeline.process_file(target)
        print(f"Done. Extracted 1 product row from {product.source_file}.")
        print(f"Products Excel: {config.get('outputs', 'products_xlsx')}")
        print(f"Review Excel: {config.get('outputs', 'review_xlsx')}")
        return

    if mode == "folder":
        print(f"Scanning folder: {target_path}")
    else:
        print(f"Scanning root folder and all nested folders: {target_path}")
    products = pipeline.process_folder(target)
    print(f"Done. Extracted {len(products)} product rows.")
    print(f"Products Excel: {config.get('outputs', 'products_xlsx')}")
    print(f"Review Excel: {config.get('outputs', 'review_xlsx')}")


if __name__ == "__main__":
    main()
