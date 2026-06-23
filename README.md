# Simple Python RAG Data Extraction

Command-line Python RAG pipeline for extracting banking product fields from documents into Excel.
The current setup is designed for Google Colab with a Transformers-based Qwen3-14B model.

## What It Does

- Recursively scans a document folder
- Parses `.docx`, `.pdf`, `.json`, and `.txt` files
- Splits parsed content into chunks
- Creates local hashing embeddings by default
- Stores and searches chunks with Qdrant client fallback memory storage
- Uses a Transformers-loaded Qwen3-14B model for JSON field extraction and validation
- Exports results to Excel

## Kept Project Structure

```text
main.py                 CLI entry point
app/                    pipeline, config, chunking, Excel export
agents/                 Transformers extraction, validation, confidence scoring
embeddings/             local hashing embeddings
parsers/                document parsers
retrievers/             semantic retrieval
vectorstore/            Qdrant vector store wrapper
prompts/                extraction and validation prompts
config/config.yaml      schema and output config
COLAB_TRANSFORMERS_GUIDE.md  step-by-step Colab setup guide
Documents/              source document corpus
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your Transformers model name or path:

```env
LLM_MODEL_NAME_OR_PATH=Qwen/Qwen3-14B-Instruct
LLM_USE_4BIT=true
LLM_TEMPERATURE=0
LLM_TOP_P=0.9
LLM_REPETITION_PENALTY=1.05
LLM_MAX_NEW_TOKENS=1024
QDRANT_URL=http://localhost:6333
```

If Qdrant is not running, the app falls back to in-memory vector storage for the current run.

## Model Setup

1. Use the Hugging Face model id `Qwen/Qwen3-14B-Instruct` or a local path that contains the model files.
2. Keep `LLM_USE_4BIT=true` for Colab unless you know you have enough VRAM for full precision.
3. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If the model is gated on Hugging Face, log in with `huggingface-cli login` or provide a token in Colab before loading it.

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
