# Simple Python RAG Data Extraction

Command-line Python RAG pipeline for extracting banking product fields from documents into Excel.

## What It Does

- Recursively scans a document folder
- Parses `.docx`, `.pdf`, `.json`, and `.txt` files
- Splits parsed content into chunks
- Creates local hashing embeddings by default
- Stores and searches chunks with Qdrant client fallback memory storage
- Uses a local GGUF Qwen 2.5 model for JSON field extraction and validation
- Exports results to Excel

## Kept Project Structure

```text
main.py                 CLI entry point
app/                    pipeline, config, chunking, Excel export
agents/                 local GGUF extraction, validation, confidence scoring
embeddings/             local hashing embeddings
parsers/                document parsers
retrievers/             semantic retrieval
vectorstore/            Qdrant vector store wrapper
prompts/                extraction and validation prompts
config/config.yaml      schema and runtime config
Documents/              source document corpus
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your local GGUF model path:

```env
LLM_MODEL_PATH=models/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLM_N_CTX=4096
LLM_N_THREADS=8
LLM_N_GPU_LAYERS=0
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=1024
QDRANT_URL=http://localhost:6333
```

If Qdrant is not running, the app falls back to in-memory vector storage for the current run.

## Local Model Setup

1. Download a Qwen 2.5 GGUF file, such as `Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
2. Put it somewhere on disk, for example `models/Qwen2.5-7B-Instruct-GGUF/`.
3. Set `LLM_MODEL_PATH` in `.env` to the full file path.
4. Install the Python dependency:

```bash
pip install -r requirements.txt
```

If `llama-cpp-python` fails to install on Windows, you may need a prebuilt wheel or C++ build tools.

## Run

Process the included document folder:

```bash
python main.py "C:\Users\Dell\Desktop\BankAlfala\dataextrction\Documents"
```

Or run interactively:

```bash
python main.py
```

You can also process one file:

```bash
python main.py --file "C:\path\to\product.docx"
```

Outputs are created when the pipeline runs:

- `output/products.xlsx`
- `output/review_required.xlsx`
- `output/extraction_trace.jsonl`
- `logs/system.log`

## Configure

Edit `config/config.yaml` to change supported file types, chunk size, retrieval count, confidence threshold, output paths, Excel columns, or extraction groups.
