# Bank Products Extraction Pipeline

Extracts structured data for bank and insurance products into a single Excel
file with your 41 required columns.

## How it works

1. `01_segment_products.py` splits the merged PDF into one text file per
   product and writes `products_index.json`.
2. `02_extract_fields.py` uses a local Hugging Face model to turn each
   product text file into one JSON record.
3. `03_build_excel.py` merges the JSONL file into `bank_products_extracted.xlsx`.

## Local model setup

This version does not use Claude or Anthropic APIs. Stage 2 runs a local
Hugging Face model that you choose in `.env`.

Create a `.env` file from `.env.example` and set the model you want to test:

```powershell
copy .env.example .env
```

Then edit `HF_MODEL_NAME_OR_PATH` to point to either:

1. A cached Hugging Face model ID, such as `Qwen/Qwen2.5-3B-Instruct`
2. A local model folder on disk

If you want offline-only loading, keep `HF_LOCAL_FILES_ONLY=true`.

## Install and run

```powershell
pip install -r requirements.txt
python run_all.py
```

`run_all.py` will:

1. Regenerate `products/` and `products_index.json` if `final_report_merged.pdf`
   is present in the project folder
2. Run field extraction with the configured local model
3. Build the final Excel workbook

If the PDF is missing, it skips stage 1 and uses the existing `products/`
and `products_index.json` files already in the folder.

## Colab

If you want to run this in Google Colab, use `colab_setup.py` and
`COLAB_QUICKSTART.md`.

The shortest path is:

1. Mount Drive in Colab
2. Clone or open the repo
3. Run `python colab_setup.py`
4. Edit `.env` with your model and file paths
5. Run `python run_all.py`

## Output files

```text
products/
products_index.json
extracted_data.jsonl
bank_products_extracted.xlsx
```

## Notes

- `02_extract_fields.py` is resumable and skips products already written to
  `extracted_data.jsonl`.
- The extractor writes `"N/A"` for fields that are not present or not
  applicable to the product.
- You can compare different models by changing `HF_MODEL_NAME_OR_PATH` in
  `.env` and re-running stage 2 or `run_all.py`.
