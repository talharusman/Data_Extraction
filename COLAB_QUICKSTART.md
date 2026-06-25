# Colab Quickstart

Use these cells in a new Google Colab notebook.

## 1) Mount Drive and clone the repo

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone <YOUR_REPO_URL> bank_extractor
%cd /content/bank_extractor
```

If the repo is already in Google Drive, just `cd` into that folder instead.

## 2) Set up the project

```bash
!python colab_setup.py
```

Open `.env` and set:

```text
HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
HF_MODEL_CLASS=causal
HF_LOCAL_FILES_ONLY=false
```

If the model is already in your Colab cache or mounted Drive folder, you can
keep `HF_LOCAL_FILES_ONLY=true`.

Avoid reasoning-first models like `DeepSeek-R1-*` for this pipeline unless you
have to. Instruct models are much better at returning strict JSON.

## 3) Install dependencies

```bash
!pip install -r requirements.txt
```

If you hit a memory issue with `torch`, install a Colab-friendly build instead:

```bash
!pip install --upgrade torch transformers accelerate
```

## 4) Run the pipeline

```bash
!python run_all.py
```

## 5) Get the output

The final workbook will be written to:

```text
bank_products_extracted.xlsx
```

If you want it saved directly into Drive, set `OUT_XLSX` in `.env` to a path
like:

```text
OUT_XLSX=/content/drive/MyDrive/bank_extractor/bank_products_extracted.xlsx
```

You can also point `PDF_PATH` to a file in Drive if you want to regenerate the
product text files from the PDF inside Colab.
