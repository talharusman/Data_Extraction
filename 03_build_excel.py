"""
Step 3: Assemble extracted_data.jsonl into a single Excel workbook with the
required 41 columns, one row per product.
"""
from __future__ import annotations

import json

import pandas as pd

from pipeline_config import COLUMNS, OUT_JSONL, OUT_XLSX


def main():
    rows = []
    with open(OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({col: rec.get(col, "N/A") for col in COLUMNS})

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_excel(OUT_XLSX, index=False, sheet_name="Products")
    print(f"Wrote {len(df)} rows to {OUT_XLSX}")


if __name__ == "__main__":
    main()
