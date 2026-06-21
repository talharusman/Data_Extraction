# Simple Python RAG Data Extraction

Command-line Python RAG pipeline for extracting banking product fields from documents into Excel.

## What It Does

- Recursively scans a document folder
- Parses `.docx`, `.pdf`, `.json`, and `.txt` files
- Splits parsed content into chunks
- Creates local hashing embeddings by default
- Stores and searches chunks with Qdrant client fallback memory storage
- Uses Groq for JSON field extraction and validation
- Exports results to Excel

## Kept Project Structure

```text
main.py                 CLI entry point
app/                    pipeline, config, chunking, Excel export
agents/                 Groq extraction, validation, confidence scoring
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

Edit `.env` and set your Groq key:

```env
GROQ_API_KEY=your-key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-20b
QDRANT_URL=http://localhost:6333
```

If Qdrant is not running, the app falls back to in-memory vector storage for the current run.

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
