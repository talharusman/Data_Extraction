"""
Step 2: Extract the fixed 56-column schema as JSON from product files.
UPGRADED: Hybrid RAG Pipeline with Token-Based Chunking, FAISS, BM25,
Static Field Groups, Group-Specific Prompts, Confidence Scoring,
Evidence Tracking, and Improved Merge Strategy.

Backward-compatible with the original pipeline. All external behavior
(JSONL output, field schema, normalization) is preserved.

SYSTEM PROMPT IS READ FROM: EXTRACTION_SYSTEM_PROMPT.txt

Input modes (same as before):
  1) Single file
  2) Folder (non-recursive)
  3) Nested folder (recursive)

Supported file types: .txt, .pdf, .docx, .doc, .csv, .json, .xlsx, .xls

Required packages (original):
  pip install pdfplumber python-docx openpyxl
  # for .doc: sudo apt install antiword

NEW required packages for Hybrid RAG:
  pip install sentence-transformers faiss-cpu rank_bm25 tiktoken

Configure the model through .env:
  HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
  HF_MODEL_CLASS=causal
  HF_LOCAL_FILES_ONLY=true
  EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
  HYBRID_TOP_K=4
  TOKEN_CHUNK_SIZE=850
  TOKEN_CHUNK_OVERLAP=150

Resumable: already-extracted products are skipped on re-run.
"""
from __future__ import annotations

import ast
import gc
import json
import os
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub.errors import RepositoryNotFoundError
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from pipeline_config import (
    COLUMNS,
    OUT_JSONL,
    env_bool,
    env_float,
    env_int,
    load_local_env,
)

load_local_env()

# ---------------------------------------------------------------------------
# Inject PREMIUM_PAYMENT_FREQUENCY if pipeline_config was not yet updated.
# ---------------------------------------------------------------------------
_NEW_COLUMNS = ["PREMIUM_PAYMENT_FREQUENCY"]
for _col in _NEW_COLUMNS:
    if _col not in COLUMNS:
        COLUMNS = list(COLUMNS) + [_col]

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = os.environ.get("HF_MODEL_NAME_OR_PATH", "").strip()
MODEL_CLASS = os.environ.get("HF_MODEL_CLASS", "causal").strip().lower()
LOCAL_FILES_ONLY = env_bool("HF_LOCAL_FILES_ONLY", True)
TRUST_REMOTE_CODE = env_bool("HF_TRUST_REMOTE_CODE", False)
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 1500)
TEMPERATURE = env_float("HF_TEMPERATURE", 0.0)
TOP_P = env_float("HF_TOP_P", 1.0)
REPETITION_PENALTY = env_float("HF_REPETITION_PENALTY", 1.03)

# Legacy character chunking (kept for fallback / ENABLE_CHUNKING=False path)
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", 6000)
TEXT_CHUNK_OVERLAP = env_int("TEXT_CHUNK_OVERLAP", 300)

# Hybrid RAG settings
TOKEN_CHUNK_SIZE = env_int("TOKEN_CHUNK_SIZE", 850)
TOKEN_CHUNK_OVERLAP = env_int("TOKEN_CHUNK_OVERLAP", 150)
HYBRID_TOP_K = env_int("HYBRID_TOP_K", 4)
EMBED_MODEL_NAME = os.environ.get(
    "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
).strip()
ENABLE_HYBRID_RAG = env_bool("ENABLE_HYBRID_RAG", True)

LOAD_IN_4BIT = env_bool("HF_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = env_bool("HF_LOAD_IN_8BIT", False)
DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
TORCH_DTYPE = os.environ.get("HF_TORCH_DTYPE", "auto").strip().lower()
GPU_RESERVE_GIB = env_float("HF_GPU_RESERVE_GIB", 3.0)
ENABLE_CHUNKING = env_bool("ENABLE_CHUNKING", True)

DEFAULT_VALUE = "N/A"
NUMERIC_COLUMNS = {
    "MIN_AGE", "MAX_AGE", "IS_BANK_OFFERED", "MIN_BALANCE",
    "AVG_BALANCE_REQUIREMENT", "MIN_INCOME", "MIN_INCOME_USD",
    "MIN_INVESTMENT", "MIN_CONTRIBUTION", "MIN_TERM_YEARS",
    "MAX_TERM_YEARS", "FREE_LOOK_PERIOD_DAYS",
}

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc", ".csv", ".json", ".xlsx", ".xls"}

# ============================================================================
# STATIC FIELD GROUPS
# Defined once at development time. Never generated at runtime by the LLM.
# Grouped by semantic relatedness to maximize focused extraction accuracy.
# ============================================================================

FIELD_GROUPS: dict[str, list[str]] = {
    "identity": [
        "PRODUCT_NAME", "LEAD_MARKER", "SOURCE_FILE_PRODUCT",
        "PLAN_TYPE", "TARGET_GOAL", "PRODUCT_DESCRIPTION",
        "PROVIDER_NAME", "PRODUCT_VARIANT_TIER",
    ],
    "customer_profile": [
        "CUSTOMER_TYPE", "EMPLOYMENT_TYPE", "CUSTOMER_SEGMENT",
        "TARGET_SEGMENT", "SEGMENT_TIER", "MIN_AGE", "MAX_AGE",
        "GENDER", "IS_BANK_OFFERED", "ELIGIBILITY_TYPE",
        "ACCOUNT_TYPE", "CARD_TYPE", "CHANNEL",
    ],
    "financial_structure": [
        "CURRENCY", "CURRENCY_TYPE", "MIN_BALANCE", "AVG_BALANCE_REQUIREMENT",
        "MIN_INCOME", "MIN_INCOME_USD", "MIN_INVESTMENT", "MIN_CONTRIBUTION",
        "LOAN_AMOUNT_RANGE", "FINANCING_TYPE", "DEPOSIT_PROFIT_TYPE",
        "DEPOSIT_PROFIT_FREQUENCY", "PRICING_RATE", "FEES_AND_CHARGES",
        "TRANSACTION_LIMIT",
    ],
    "coverage_and_tenure": [
        "COVERAGE_AMOUNT", "TENURE", "TENURE_OPTIONS",
        "MIN_TERM_YEARS", "MAX_TERM_YEARS", "BUSINESS_TENURE",
        "COLLATERAL_TYPE", "EQUITY_REQUIREMENT", "DBR_LIMIT",
        "PREMIUM_PAYMENT_FREQUENCY",
    ],
    "benefits_and_conditions": [
        "SERVICE_TYPE", "REWARD_TYPE", "KEY_BENEFITS", "OPTIONAL_RIDERS",
        "FREE_LOOK_PERIOD_DAYS", "REQUIRED_DOCUMENTS", "CLAIMS_SERVICE_CONTACT",
        "KEY_EXCLUSIONS", "TAX_ZAKAT_TREATMENT", "SPECIAL_CONDITIONS",
    ],
}

# Retrieval queries auto-derived from field names + descriptions.
# Built once at module load; never generated at runtime by the LLM.
FIELD_GROUP_QUERIES: dict[str, str] = {
    "identity": (
        "product name plan type description provider underwriter bank insurance takaful"
    ),
    "customer_profile": (
        "customer eligibility age gender employment income segment "
        "account card channel branch salaried retail corporate"
    ),
    "financial_structure": (
        "currency balance income contribution premium investment "
        "profit rate deposit financing conventional Islamic takaful fees charges"
    ),
    "coverage_and_tenure": (
        "coverage amount term years tenure duration collateral DBR equity "
        "loan premium payment frequency annual quarterly"
    ),
    "benefits_and_conditions": (
        "benefits riders free look period documents claims exclusions tax zakat "
        "special conditions restrictions reporting"
    ),
}

# ============================================================================
# BASE SYSTEM PROMPT (global extraction rules only, no field definitions)
# ============================================================================

BASE_SYSTEM_PROMPT = """You are a precision data extraction engine for banking and insurance product documents.

GLOBAL RULES — always apply:
- Extract ONLY information explicitly stated in the provided text.
- Never infer, hallucinate, or use outside knowledge.
- Ignore marketing language and promotional wording.
- Ignore examples and worked scenarios.
- Preserve original numeric values (do not estimate).
- Use "N/A" for every missing field. Never use null/None/NaN/empty string.
- Return ONLY valid JSON. No markdown fences. No explanations outside JSON.
- Extract from the provided chunks ONLY. Do not fill from general knowledge."""


# ============================================================================
# LOAD SYSTEM PROMPT FROM FILE
# ============================================================================

def load_system_prompt(prompt_file: str = "EXTRACTION_SYSTEM_PROMPT.txt") -> str:
    search_paths = [
        Path(prompt_file),
        Path(__file__).parent / prompt_file,
        Path(__file__).parent.parent / prompt_file,
    ]
    for prompt_path in search_paths:
        if prompt_path.exists() and prompt_path.is_file():
            print(f"✓ Loaded system prompt from: {prompt_path.resolve()}")
            return prompt_path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"\n{'='*70}\n"
        f"ERROR: System prompt file not found!\n"
        f"Expected file: {prompt_file}\n\n"
        f"Searched in:\n"
        + "\n".join(f"  - {p.resolve()}" for p in search_paths) +
        f"\n\nSolution:\n"
        f"1. Make sure '{prompt_file}' is in the same directory as this script\n"
        f"2. Or in the parent directory of this script\n"
        f"3. Or in the current working directory\n"
        f"{'='*70}\n"
    )


# Load the full system prompt at module level (reused as base for all group prompts)
SYSTEM_PROMPT = load_system_prompt()


# ============================================================================
# PROMPT BUILDER
# Transforms the existing SYSTEM_PROMPT into a group-specific prompt.
# Removes all field definitions not in the active group; preserves all rules.
# ============================================================================

def _extract_columns_section(prompt: str) -> tuple[str, str, str]:
    """
    Split the full system prompt into three parts:
      (before_columns, columns_block, after_columns)
    where columns_block contains the COLUMNS: ... list and rules.
    """
    # Find the COLUMNS: line
    col_match = re.search(r"^(COLUMNS:.*?)$", prompt, re.MULTILINE)
    if not col_match:
        return prompt, "", ""

    col_start = col_match.start()

    # Find the ===GLOBAL RULES=== section that follows
    rules_match = re.search(r"^===", prompt[col_start:], re.MULTILINE)
    if rules_match:
        col_end = col_start + rules_match.start()
    else:
        col_end = len(prompt)

    before = prompt[:col_start]
    columns_block = prompt[col_start:col_end]
    after = prompt[col_end:]
    return before, columns_block, after


def _filter_column_rules(after_section: str, group_fields: list[str]) -> str:
    """
    From the ===COLUMN RULES=== section, keep only the rules for fields
    that belong to the active group. All other field-specific rules are removed.
    """
    # Find ===COLUMN RULES=== block
    col_rules_match = re.search(r"(===COLUMN RULES===.*?)(?====|\Z)", after_section, re.DOTALL)
    if not col_rules_match:
        return after_section

    col_rules_text = col_rules_match.group(1)
    rest = after_section[col_rules_match.end():]

    # Split column rules by field entries (FIELDNAME: rule text)
    field_set = set(group_fields)
    kept_rules = ["===COLUMN RULES==="]

    # Parse each field rule block
    # Pattern: field name followed by colon at start of a line (or after newline)
    entries = re.split(r"\n(?=[A-Z_]+:)", col_rules_text)
    for entry in entries:
        stripped = entry.strip()
        if not stripped or stripped.startswith("===COLUMN RULES==="):
            continue
        # Extract field name
        field_match = re.match(r"^([A-Z_/]+):", stripped)
        if field_match:
            field_name = field_match.group(1)
            # Check if any group field matches (handle compound like MIN_AGE/MAX_AGE)
            parts = re.split(r"[/]", field_name)
            if any(p.strip() in field_set for p in parts):
                kept_rules.append(stripped)
        else:
            # Non-field header lines (keep)
            kept_rules.append(stripped)

    return "\n".join(kept_rules) + "\n" + rest


def build_group_prompt(group_name: str, group_fields: list[str], entry: dict) -> str:
    """
    Build a group-specific extraction prompt by:
    1. Keeping all global rules from the original SYSTEM_PROMPT.
    2. Replacing the full COLUMNS list with only this group's fields.
    3. Filtering column rules to only this group's fields.
    4. Appending a compact user message for the chunk (caller injects chunk text).

    The LLM never sees all 56 fields unless in the final validation pass.
    """
    before, _columns_block, after = _extract_columns_section(SYSTEM_PROMPT)

    # Build replacement columns block with only this group's fields
    fields_str = ", ".join(group_fields)
    new_columns_block = f"COLUMNS: {fields_str}\n\n"

    # Filter column rules to this group only
    filtered_after = _filter_column_rules(after, group_fields)

    # Remove ===WORKED EXAMPLES=== and ===DO NOT=== sections to save tokens
    # (they add ~800 tokens and are less critical for focused group extraction)
    filtered_after = re.sub(
        r"===WORKED EXAMPLES.*?(?====DO NOT===|$)", "", filtered_after, flags=re.DOTALL
    )
    filtered_after = re.sub(
        r"===DO NOT===.*", "", filtered_after, flags=re.DOTALL
    )

    group_system_prompt = (
        BASE_SYSTEM_PROMPT
        + "\n\n"
        + before.strip()
        + "\n\n"
        + new_columns_block
        + filtered_after.strip()
        + f"\n\nExtract ONLY these {len(group_fields)} fields: {fields_str}"
        + f"\nReturn ONE valid JSON object with exactly these keys."
    )

    return group_system_prompt


# ============================================================================
# File reading functions (unchanged from original)
# ============================================================================

def read_file_text(path: Path) -> str:
    """Extract plain text from any supported file type."""
    ext = path.suffix.lower()

    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    elif ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise SystemExit(
                "pdfplumber is required to read PDF files.\n"
                "Install it with:  pip install pdfplumber"
            )
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            raise SystemExit(
                "python-docx is required to read DOCX files.\n"
                "Install it with:  pip install python-docx"
            )
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = "\t".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)

    elif ext == ".doc":
        try:
            result = subprocess.run(
                ["antiword", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            return result.stdout
        except FileNotFoundError:
            raise SystemExit(
                "antiword is required to read old .doc files.\n"
                "Install it with:  sudo apt install antiword  (Linux)\n"
                "or download from http://www.winfield.demon.nl/ (Windows/Mac)"
            )

    elif ext == ".csv":
        return path.read_text(encoding="utf-8", errors="replace")

    elif ext == ".json":
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return raw

    elif ext in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            raise SystemExit(
                "openpyxl is required to read Excel files.\n"
                "Install it with:  pip install openpyxl"
            )
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = []
        for sheet in wb.worksheets:
            rows.append(f"[Sheet: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join(
                    str(cell) if cell is not None else "" for cell in row
                )
                if row_text.strip():
                    rows.append(row_text)
        wb.close()
        return "\n".join(rows)

    else:
        return path.read_text(encoding="utf-8", errors="replace")


# ============================================================================
# Interactive input selection (unchanged from original)
# ============================================================================

def _prompt_choice(prompt: str, choices: list[str]) -> str:
    while True:
        print(prompt)
        for i, choice in enumerate(choices, 1):
            print(f"  {i}) {choice}")
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(choices)}.\n")


def _collect_files(path: Path, recursive: bool) -> list[Path]:
    if recursive:
        all_files = path.rglob("*")
    else:
        all_files = path.glob("*")
    return sorted(
        f for f in all_files
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def select_input_files() -> list[Path]:
    supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    print("\n" + "=" * 60)
    print("  Product Text Extractor – Input Selection")
    print(f"  Supported formats: {supported_str}")
    print("=" * 60)

    mode = _prompt_choice(
        "\nHow do you want to supply input files?",
        ["Single file", "Folder (non-recursive)", "Nested folder (recursive)"],
    )

    if mode == "Single file":
        while True:
            raw = input("\nEnter the full path to the file: ").strip()
            p = Path(raw)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                return [p]
            if p.is_file():
                print(f"  Unsupported file type '{p.suffix}'. Supported: {supported_str}\n")
            else:
                print(f"  File not found: {raw}\n")

    elif mode == "Folder (non-recursive)":
        while True:
            raw = input("\nEnter the folder path: ").strip()
            p = Path(raw)
            if p.is_dir():
                files = _collect_files(p, recursive=False)
                if files:
                    print(f"  Found {len(files)} supported file(s) directly in '{p}'.")
                    return files
                print(f"  No supported files found directly in '{p}'.\n  Supported formats: {supported_str}\n")
            else:
                print(f"  Not a valid directory: {raw}\n")

    else:
        while True:
            raw = input("\nEnter the root folder path: ").strip()
            p = Path(raw)
            if p.is_dir():
                files = _collect_files(p, recursive=True)
                if files:
                    print(f"  Found {len(files)} supported file(s) under '{p}' (all sub-folders included).")
                    return files
                print(f"  No supported files found anywhere under '{p}'.\n  Supported formats: {supported_str}\n")
            else:
                print(f"  Not a valid directory: {raw}\n")


def build_index_from_files(files: list[Path]) -> list[dict]:
    index = []
    for i, path in enumerate(files, start=1):
        index.append({
            "product_no": i,
            "title": path.stem.replace("_", " ").replace("-", " "),
            "file": str(path),
            "filename": path.name,
            "folder": path.parent.name,
            "_abs_path": path.resolve(),
        })
    return index


# ============================================================================
# TOKEN-BASED CHUNKING (replaces character chunking)
# ============================================================================

def _get_tiktoken_encoder():
    """Lazy-load tiktoken cl100k_base encoder (compatible with most LLMs)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def chunk_text_tokens(
    text: str,
    chunk_size: int = TOKEN_CHUNK_SIZE,
    overlap: int = TOKEN_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into token-based chunks with overlap.
    Returns list of dicts with: chunk_id, token_count, text.

    Falls back to character chunking if tiktoken is unavailable.
    """
    enc = _get_tiktoken_encoder()

    if enc is None:
        # Fallback to character-based chunking
        char_chunks = chunk_text(text, chunk_size * 4, overlap * 4)
        return [
            {"chunk_id": i, "token_count": len(c) // 4, "text": c}
            for i, c in enumerate(char_chunks)
        ]

    tokens = enc.encode(text)
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        return [{"chunk_id": 0, "token_count": total_tokens, "text": text}]

    chunks = []
    start = 0
    chunk_id = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)

        # Try to break at a paragraph/sentence boundary in the decoded text
        if end < total_tokens:
            nl_idx = chunk_text_str.rfind("\n", len(chunk_text_str) // 2)
            period_idx = chunk_text_str.rfind(". ", len(chunk_text_str) // 2)
            boundary = max(nl_idx, period_idx)
            if boundary > len(chunk_text_str) // 2:
                chunk_text_str = chunk_text_str[:boundary + 1]
                # Re-encode to get actual token count after boundary trim
                chunk_tokens = enc.encode(chunk_text_str)
                end = start + len(chunk_tokens)

        chunks.append({
            "chunk_id": chunk_id,
            "token_count": len(chunk_tokens),
            "text": chunk_text_str.strip(),
        })

        if end >= total_tokens:
            break

        start = max(end - overlap, start + 1)
        chunk_id += 1

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Legacy character-based chunking (kept for ENABLE_CHUNKING=False path
    and as fallback when tiktoken is not installed).
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            search_from = max(start, end - 300)
            newline_idx = text.rfind("\n", search_from, end)
            period_idx = text.rfind(". ", search_from, end)
            boundary = max(newline_idx, period_idx)
            if boundary > search_from:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


# ============================================================================
# EMBEDDINGS (generate once, cache for document lifetime)
# ============================================================================

_embed_model = None


def _get_embed_model():
    """Lazy-load the sentence-transformer embedding model."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading embedding model: {EMBED_MODEL_NAME}")
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except ImportError:
            print(
                "  WARNING: sentence-transformers not installed. "
                "Falling back to BM25-only retrieval.\n"
                "  Install with: pip install sentence-transformers"
            )
    return _embed_model


def generate_embeddings(chunks: list[dict]) -> "np.ndarray | None":
    """
    Generate embeddings for all chunks. Returns numpy array shape (N, dim)
    or None if embedding model unavailable.
    """
    embed_model = _get_embed_model()
    if embed_model is None:
        return None

    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


# ============================================================================
# FAISS INDEX (built per document, destroyed after)
# ============================================================================

def build_faiss_index(embeddings: "np.ndarray") -> "faiss.IndexFlatIP | None":
    """Build a FAISS inner-product index from chunk embeddings."""
    try:
        import faiss
        import numpy as np
        dim = embeddings.shape[1]
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = (embeddings / norms).astype("float32")
        index = faiss.IndexFlatIP(dim)
        index.add(normed)
        return index
    except ImportError:
        print(
            "  WARNING: faiss-cpu not installed. Falling back to BM25-only retrieval.\n"
            "  Install with: pip install faiss-cpu"
        )
        return None


def faiss_search(
    index: "faiss.IndexFlatIP",
    query: str,
    k: int = HYBRID_TOP_K,
) -> list[tuple[int, float]]:
    """Search FAISS index for query. Returns [(chunk_idx, score), ...]."""
    import numpy as np
    embed_model = _get_embed_model()
    if embed_model is None or index is None:
        return []

    q_embed = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    norm = float(q_embed[0] @ q_embed[0]) ** 0.5
    if norm > 0:
        q_embed = q_embed / norm

    scores, indices = index.search(q_embed, k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


# ============================================================================
# BM25 INDEX (built per document, destroyed after)
# ============================================================================

def build_bm25_index(chunks: list[dict]) -> "BM25Okapi | None":
    """Build BM25 index from chunk texts."""
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in chunks]
        return BM25Okapi(tokenized)
    except ImportError:
        print(
            "  WARNING: rank_bm25 not installed. Falling back to FAISS-only retrieval.\n"
            "  Install with: pip install rank_bm25"
        )
        return None


def bm25_search(
    bm25_index: "BM25Okapi",
    query: str,
    k: int = HYBRID_TOP_K,
) -> list[tuple[int, float]]:
    """Search BM25 index. Returns [(chunk_idx, score), ...]."""
    if bm25_index is None:
        return []
    import numpy as np
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    top_k = int(min(k, len(scores)))
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, float(scores[idx])) for idx in indices]


# ============================================================================
# HYBRID RETRIEVAL
# ============================================================================

def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    faiss_index,
    bm25_index,
    k: int = HYBRID_TOP_K,
) -> list[dict]:
    """
    Retrieve top-K most relevant chunks using hybrid FAISS + BM25 scoring.
    Normalises each score set to [0,1] and averages them.
    Falls back gracefully if either index is unavailable.
    """
    faiss_results = faiss_search(faiss_index, query, k=k * 2) if faiss_index else []
    bm25_results = bm25_search(bm25_index, query, k=k * 2) if bm25_index else []

    scores: dict[int, float] = {}

    # Normalise FAISS scores
    if faiss_results:
        max_f = max(s for _, s in faiss_results) or 1e-10
        for idx, score in faiss_results:
            scores[idx] = scores.get(idx, 0.0) + (score / max_f) * 0.5

    # Normalise BM25 scores
    if bm25_results:
        max_b = max(s for _, s in bm25_results) or 1e-10
        for idx, score in bm25_results:
            scores[idx] = scores.get(idx, 0.0) + (score / max_b) * 0.5

    if not scores:
        # No retrieval available — return all chunks truncated to k
        return chunks[:k]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [chunks[idx] for idx, _ in ranked if idx < len(chunks)]


# ============================================================================
# DOCUMENT-LEVEL INDEX (temporary, destroyed after each product)
# ============================================================================

class DocumentIndex:
    """
    Temporary per-document index: token chunks + embeddings + FAISS + BM25.
    Must be explicitly destroyed after processing each product file.
    """

    def __init__(self, text: str):
        self.chunks: list[dict] = chunk_text_tokens(text)
        self.embeddings = None
        self.faiss_index = None
        self.bm25_index = None
        self._built = False

    def build(self):
        if self._built:
            return
        print(f"    Building indexes for {len(self.chunks)} token chunks…")
        self.embeddings = generate_embeddings(self.chunks)
        if self.embeddings is not None:
            self.faiss_index = build_faiss_index(self.embeddings)
        self.bm25_index = build_bm25_index(self.chunks)
        self._built = True

    def retrieve(self, query: str, k: int = HYBRID_TOP_K) -> list[dict]:
        return hybrid_retrieve(
            query, self.chunks, self.faiss_index, self.bm25_index, k=k
        )

    def destroy(self):
        """Release all memory. Must be called after processing each product."""
        self.chunks = []
        self.embeddings = None
        if self.faiss_index is not None:
            try:
                # FAISS indices don't have a close() but freeing the ref is enough
                del self.faiss_index
            except Exception:
                pass
            self.faiss_index = None
        self.bm25_index = None
        self._built = False
        gc.collect()


# ============================================================================
# JSON parsing helpers (unchanged from original)
# ============================================================================

def _strip_wrappers(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?is)</?think>", "", text)
    return text.strip()


def _parse_candidate(candidate):
    candidate = candidate.strip()
    if not candidate:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _iter_json_candidates(text):
    text = _strip_wrappers(text)
    seen: set[str] = set()

    def emit(candidate):
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced:
        yield from emit(block)

    yield from emit(text)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        yield from emit(text[first: last + 1])

    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for brace_start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(brace_start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield from emit(text[brace_start: idx + 1])
                        break


def parse_json_blob(raw):
    for candidate in _iter_json_candidates(raw):
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = _parse_candidate(cleaned)
        if parsed is not None:
            return parsed
    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


# ============================================================================
# Record helpers with normalization (unchanged from original)
# ============================================================================

def blank_record(entry):
    record = {col: DEFAULT_VALUE for col in COLUMNS}
    record["PRODUCT_NAME"] = entry["title"]
    record["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)
    return record


def get_source_filename(entry) -> str:
    for key in ("filename", "file"):
        raw = entry.get(key)
        if raw:
            return Path(str(raw)).name
    abs_path = entry.get("_abs_path")
    if abs_path:
        return Path(str(abs_path)).name
    return entry.get("title", DEFAULT_VALUE)


field_max_lengths = {
    "PRODUCT_NAME": 50,
    "PRODUCT_DESCRIPTION": 250,
    "PROVIDER_NAME": 100,
    "PRODUCT_VARIANT_TIER": 50,
    "PRICING_RATE": 400,
    "FEES_AND_CHARGES": 300,
    "KEY_BENEFITS": 280,
    "OPTIONAL_RIDERS": 300,
    "REQUIRED_DOCUMENTS": 220,
    "CLAIMS_SERVICE_CONTACT": 200,
    "KEY_EXCLUSIONS": 200,
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 30,
    "TARGET_GOAL": 50,
    "CUSTOMER_TYPE": 25,
    "EMPLOYMENT_TYPE": 500,
    "ACCOUNT_TYPE": 20,
    "CARD_TYPE": 25,
    "CHANNEL": 30,
    "ELIGIBILITY_TYPE": 100,
    "SERVICE_TYPE": 25,
    "REWARD_TYPE": 25,
    "CURRENCY": 30,
    "CURRENCY_TYPE": 15,
    "LOAN_AMOUNT_RANGE": 50,
    "COVERAGE_AMOUNT": 300,
    "FINANCING_TYPE": 50,
    "DEPOSIT_PROFIT_TYPE": 30,
    "DEPOSIT_PROFIT_FREQUENCY": 20,
    "TENURE": 50,
    "TENURE_OPTIONS": 50,
    "BUSINESS_TENURE": 30,
    "COLLATERAL_TYPE": 50,
    "EQUITY_REQUIREMENT": 20,
    "DBR_LIMIT": 20,
    "TRANSACTION_LIMIT": 50,
    "SPECIAL_CONDITIONS": 250,
    "PREMIUM_PAYMENT_FREQUENCY": 50,
}


def truncate_to_boundary(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    cut = value[:max_len].rstrip()
    boundary = max(cut.rfind(" "), cut.rfind(","), cut.rfind(";"), cut.rfind(":"), cut.rfind("-"))
    if boundary > 0:
        return cut[:boundary].rstrip(" ,;:-/")
    return cut


def _normalize_plan_type(value: str) -> str:
    valid_types = {
        "Loan", "Deposit", "Savings", "Card", "Investment",
        "Insurance", "Service", "Loyalty", "Protection", "Health",
        "Savings & Protection Insurance",
        "Insurance (Hospitalization)",
    }
    stripped = value.strip()
    if stripped in valid_types:
        return stripped
    for vt in valid_types:
        if stripped.lower() == vt.lower():
            return vt
    for word in re.split(r"[\s,;/]+", stripped):
        word_clean = word.strip(".,;:()")
        if word_clean in valid_types:
            return word_clean
        for vt in valid_types:
            if word_clean.lower() == vt.lower():
                return vt
    return DEFAULT_VALUE


def _normalize_financing_type(value: str) -> str:
    stripped = value.strip()
    lower = stripped.lower()
    if "hybrid" in lower or ("bonus" in lower and "unit" in lower):
        return "Hybrid (Bonus Based and Unit Linked)"
    if "unit linked" in lower or "unit-linked" in lower:
        return "Unit Linked"
    canonical_map = {
        "conventional": "Conventional",
        "islamic": "Islamic",
        "takaful": "Takaful",
        "mudarabah": "Mudarabah",
    }
    for key, canonical in canonical_map.items():
        if key in lower:
            return canonical
    known_exact = {
        "Conventional", "Islamic", "Takaful", "Mudarabah",
        "Unit Linked", "Hybrid (Bonus Based and Unit Linked)", DEFAULT_VALUE,
    }
    if stripped in known_exact:
        return stripped
    return stripped


def normalize_record(record, entry):
    """
    Normalize an extracted record. Unchanged from original — all existing
    business logic preserved.
    """
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}

    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)

        if value in (None, "", []):
            value = DEFAULT_VALUE

        if col in NUMERIC_COLUMNS and isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.upper() == DEFAULT_VALUE:
                value = DEFAULT_VALUE
            else:
                match = re.match(r'^(\d+(?:\.\d+)?)', stripped.replace(",", ""))
                if match:
                    num_str = match.group(1)
                    if "." in num_str:
                        try:
                            value = str(int(float(num_str)))
                        except ValueError:
                            value = DEFAULT_VALUE
                    else:
                        value = num_str
                else:
                    value = DEFAULT_VALUE

        if col == "PRODUCT_NAME" and isinstance(value, str):
            stripped_name = value.strip()
            if (stripped_name
                    and stripped_name == stripped_name.upper()
                    and len(stripped_name.split()) > 1
                    and len(stripped_name) > 5):
                value = stripped_name.title()

        if col == "PLAN_TYPE" and isinstance(value, str):
            value = _normalize_plan_type(value)

        if col == "FINANCING_TYPE" and isinstance(value, str):
            if value not in (DEFAULT_VALUE, ""):
                value = _normalize_financing_type(value)

        if col == "GENDER" and isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ("male", "m"):
                value = "Male"
            elif value_lower in ("female", "f"):
                value = "Female"
            elif value_lower in ("all", "both", "all genders", "all customers"):
                value = "All"
            elif value_lower in ("n/a", "na", ""):
                value = DEFAULT_VALUE
            else:
                value = DEFAULT_VALUE

        if col == "CUSTOMER_TYPE" and isinstance(value, str):
            allowed = {
                "Salaried", "Self-Employed", "SME",
                "Corporate", "Retail", "Government",
            }
            stripped_ct = value.strip()
            if stripped_ct in allowed:
                pass
            elif " | " in stripped_ct:
                # Multiple pipe-separated values — validate each token
                tokens = [t.strip() for t in stripped_ct.split(" | ")]
                valid_tokens = [t for t in tokens if t in allowed]
                value = " | ".join(valid_tokens) if valid_tokens else DEFAULT_VALUE
            elif "," in stripped_ct:
                first = stripped_ct.split(",")[0].strip()
                value = first if first in allowed else DEFAULT_VALUE
            elif stripped_ct.lower() == "n/a":
                value = DEFAULT_VALUE

        if col == "OPTIONAL_RIDERS" and isinstance(value, str):
            if value != DEFAULT_VALUE:
                if "." not in value:
                    value = re.sub(r"\s*;\s*", " | ", value).strip().strip("|").strip()

        if col in field_max_lengths and isinstance(value, str):
            max_len = field_max_lengths[col]
            if len(value) > max_len:
                value = truncate_to_boundary(value, max_len)

        normalized[col] = value

    raw_name = record.get("PRODUCT_NAME") or entry["title"]
    if isinstance(raw_name, str):
        if (raw_name.strip()
                and raw_name.strip() == raw_name.strip().upper()
                and len(raw_name.strip().split()) > 1):
            raw_name = raw_name.strip().title()
    normalized["PRODUCT_NAME"] = raw_name
    normalized["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)

    return normalized


# ============================================================================
# CONFIDENCE + EVIDENCE TRACKING (internal only)
# ============================================================================

def _extract_with_evidence(raw_json: dict, retrieved_chunks: list[dict]) -> dict:
    """
    For each extracted field value, try to locate the supporting evidence
    in the retrieved chunks. Returns an internal evidence dict (not written
    to final JSONL output unless DEBUG_MODE=true).
    """
    evidence: dict[str, dict] = {}
    chunk_texts = [c["text"] for c in retrieved_chunks]
    all_text = " ".join(chunk_texts).lower()

    for field, value in raw_json.items():
        if value in (None, "", DEFAULT_VALUE, "N/A"):
            evidence[field] = {"value": value, "confidence": 0.0, "evidence": "", "chunk_id": -1}
            continue

        value_str = str(value).lower()
        # Simple heuristic: does the value (or a key substring) appear in retrieved chunks?
        found_in = -1
        best_conf = 0.3  # base confidence for present-but-unverified value

        # Search each chunk for the value
        for chunk in retrieved_chunks:
            chunk_lower = chunk["text"].lower()
            if value_str in chunk_lower:
                found_in = chunk.get("chunk_id", -1)
                best_conf = 0.95
                break
            # Partial match: check if 40%+ of words appear
            value_words = [w for w in value_str.split() if len(w) > 3]
            if value_words:
                matches = sum(1 for w in value_words if w in chunk_lower)
                ratio = matches / len(value_words)
                if ratio > 0.5 and ratio * 0.9 > best_conf:
                    best_conf = ratio * 0.9
                    found_in = chunk.get("chunk_id", -1)

        # Determine evidence snippet
        if found_in >= 0:
            src_text = next(
                (c["text"] for c in retrieved_chunks if c.get("chunk_id") == found_in),
                ""
            )
            # Find surrounding sentence
            idx = src_text.lower().find(value_str[:20]) if len(value_str) >= 4 else -1
            if idx >= 0:
                start = max(0, idx - 60)
                end = min(len(src_text), idx + 120)
                snippet = src_text[start:end].replace("\n", " ").strip()
            else:
                snippet = src_text[:120].replace("\n", " ").strip()
        else:
            snippet = ""
            best_conf = 0.2  # value not found in any retrieved chunk

        evidence[field] = {
            "value": value,
            "confidence": round(best_conf, 2),
            "evidence": snippet,
            "chunk_id": found_in,
        }

    return evidence


def _confidence_merge(
    accumulated: dict,
    new_evidence: dict[str, dict],
    columns: list[str],
) -> dict:
    """
    Improved merge strategy (replaces "first non-null wins"):
    1. Highest confidence wins
    2. Explicit evidence (found in chunk) preferred over unverified
    3. Majority agreement (tracked via confidence accumulation)
    4. N/A is always overridable

    `accumulated` maps column → {"value": ..., "confidence": float}
    Returns updated accumulated dict.
    """
    for col in columns:
        new_ev = new_evidence.get(col, {})
        new_val = new_ev.get("value", DEFAULT_VALUE)
        new_conf = new_ev.get("confidence", 0.0)

        if new_val in (None, "", DEFAULT_VALUE):
            continue

        old = accumulated.get(col, {})
        old_val = old.get("value", DEFAULT_VALUE)
        old_conf = old.get("confidence", 0.0)

        if old_val in (None, "", DEFAULT_VALUE):
            # First real value found
            accumulated[col] = {"value": new_val, "confidence": new_conf}
        elif new_conf > old_conf:
            # Higher confidence replaces lower
            accumulated[col] = {"value": new_val, "confidence": new_conf}
        elif new_val == old_val:
            # Agreement boosts confidence (majority vote signal)
            accumulated[col] = {
                "value": old_val,
                "confidence": min(1.0, old_conf + 0.05),
            }

    return accumulated


def _flatten_accumulated(accumulated: dict, columns: list[str]) -> dict:
    """Extract just the values from the confidence-tracking dict."""
    result = {}
    for col in columns:
        entry = accumulated.get(col)
        if isinstance(entry, dict):
            result[col] = entry.get("value", DEFAULT_VALUE)
        else:
            result[col] = entry if entry is not None else DEFAULT_VALUE
    return result


# ============================================================================
# Prompt builders
# ============================================================================

def build_prompt(entry, chunk, tokenizer, chunk_idx=1, chunk_total=1):
    """
    Build the extraction prompt for ONE chunk (legacy path / validation pass).
    Uses the full SYSTEM_PROMPT with all 56 fields.
    """
    if chunk_total > 1:
        chunk_note = (
            f"\nNOTE: This is PART {chunk_idx} of {chunk_total} of a single, longer "
            f"product document (split only because of length). Extract whatever fields "
            f"you can find in THIS part only. Set missing fields to \"N/A\".\n"
        )
    else:
        chunk_note = ""

    user_msg = (
        f"Product title: {entry['title']}\n"
        f"Source file: {get_source_filename(entry)}\n"
        f"{chunk_note}"
        f"--- PRODUCT TEXT START ---\n"
        f"{chunk}\n"
        f"--- PRODUCT TEXT END ---\n\n"
        f"Return only one JSON object with all 56 fields as defined in the system prompt."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            pass

    return f"{SYSTEM_PROMPT}\n\n{user_msg}"


def build_group_extraction_prompt(
    group_name: str,
    group_fields: list[str],
    entry: dict,
    chunks: list[dict],
    tokenizer,
) -> str:
    """
    Build a focused group-specific extraction prompt.
    Only includes fields for the active group.
    """
    group_system = build_group_prompt(group_name, group_fields, entry)

    combined_text = "\n\n---\n\n".join(c["text"] for c in chunks)

    user_msg = (
        f"Product title: {entry['title']}\n"
        f"Source file: {get_source_filename(entry)}\n"
        f"--- PRODUCT TEXT START ---\n"
        f"{combined_text}\n"
        f"--- PRODUCT TEXT END ---\n\n"
        f"Extract ONLY these fields: {', '.join(group_fields)}\n"
        f"Return ONE valid JSON object with exactly these {len(group_fields)} keys."
    )

    messages = [
        {"role": "system", "content": group_system},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            pass

    return f"{group_system}\n\n{user_msg}"


def build_repair_prompt(entry, raw_text):
    """Compact repair prompt for malformed JSON output."""
    repair_instructions = f"""You are repairing broken JSON from a data extraction task.
Product: {entry['title']}
Source file: {get_source_filename(entry)}

BROKEN OUTPUT TO REPAIR:
{raw_text[:3000]}

REPAIR RULES — apply all of these:
- Return exactly ONE valid JSON object, nothing else
- Use "N/A" for every missing or unparseable field (never null/None/NaN/"")
- PRODUCT_NAME: Title Case (never ALL CAPS)
- LEAD_MARKER: exactly "IBG" or "BNK"
- PLAN_TYPE: one of Insurance|Savings & Protection Insurance|Insurance (Hospitalization)|Protection|Health|Savings|Deposit|Loan|Card|Investment|Service|Loyalty
- CUSTOMER_TYPE: exactly one of Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A (pipe-separated if multiple)
- GENDER: exactly one of Male|Female|All|N/A
- FINANCING_TYPE: one of Conventional|Islamic|Takaful|Mudarabah|Unit Linked|Hybrid (Bonus Based and Unit Linked)|N/A
- Numeric fields (MIN_AGE, MAX_AGE, MIN_BALANCE, MIN_INCOME, MIN_INCOME_USD,
  MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS, MAX_TERM_YEARS,
  FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED): integers only, no units, no .0
- TENURE_OPTIONS: plan duration choices only, NOT payment frequency
- PREMIUM_PAYMENT_FREQUENCY: how customer pays (Annual/Quarterly/etc.) or "N/A"
- OPTIONAL_RIDERS: pipe-separated (" | "), not semicolons
- SPECIAL_CONDITIONS: max 250 chars
- SOURCE_FILE_PRODUCT: filename only, no path
- No markdown fences, no explanations outside the JSON

Return the repaired JSON object now."""

    return repair_instructions


def build_validation_prompt(entry, extracted_json):
    """Validation and correction prompt using the full system context."""
    validation_instructions = f"""You extracted this JSON. Validate and fix any issues:

EXTRACTED JSON:
{json.dumps(extracted_json, indent=2)}

VALIDATION RULES — check each and FIX if violated:

1. PRODUCT_NAME: Is it Title Case? Not ALL CAPS?

2. PLAN_TYPE: Does it contain the word "Insurance" for insurer/takaful products?
   Accident/theft protection → "Insurance"
   Savings + protection endowment → "Savings & Protection Insurance"
   Hospitalization → "Insurance (Hospitalization)"
   Bank-only product → one of Deposit|Loan|Card|Service|Loyalty|Investment|Savings

3. CUSTOMER_TYPE: Exactly one of Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A
   (pipe-separated " | " if multiple apply). Never free text.

4. GENDER: Is it exactly one of Male|Female|All|N/A?
   "N/A" if gender is not mentioned. "All" ONLY if explicitly stated.

5. FINANCING_TYPE: For unit-linked plans (PIA, fund allocation) → "Unit Linked"
   Hybrid (bonus + unit-linked) → "Hybrid (Bonus Based and Unit Linked)"

6. DEPOSIT_PROFIT_TYPE / DEPOSIT_PROFIT_FREQUENCY: Unit-linked plans → both "N/A"
   Health/protection plans → both "N/A"

7. TENURE_OPTIONS: ONLY plan duration choices (e.g. "10 | 15 | 20 years")
   Payment frequencies → PREMIUM_PAYMENT_FREQUENCY instead.
   If no distinct plan duration menu → "N/A"

8. PREMIUM_PAYMENT_FREQUENCY: Payment frequency (Annual/Semi-Annual/Quarterly/Monthly)
   Pipe-separated " | " if multiple.

9. OPTIONAL_RIDERS: Pipe-separated " | " (not semicolons, not commas)

10. Numeric fields: integers ONLY (no PKR, no commas, no .0):
    MIN_AGE, MAX_AGE, MIN_BALANCE, AVG_BALANCE_REQUIREMENT, MIN_INCOME,
    MIN_INCOME_USD, MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS,
    MAX_TERM_YEARS, FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED

11. FREE_LOOK_PERIOD_DAYS: ONLY if THIS product explicitly states it. Never default to 14.

12. SEGMENT_TIER / SERVICE_TYPE / CUSTOMER_SEGMENT / TARGET_SEGMENT:
    "N/A" unless explicitly stated.

13. SOURCE_FILE_PRODUCT: Filename only (no folder path).

14. All 56 fields present? No null/None/NaN/empty string → "N/A"
    Columns: PRODUCT_NAME, LEAD_MARKER, SOURCE_FILE_PRODUCT, PLAN_TYPE, TARGET_GOAL,
    CUSTOMER_TYPE, EMPLOYMENT_TYPE, CUSTOMER_SEGMENT, TARGET_SEGMENT, SEGMENT_TIER,
    MIN_AGE, MAX_AGE, GENDER, IS_BANK_OFFERED, ACCOUNT_TYPE, CARD_TYPE, CHANNEL,
    ELIGIBILITY_TYPE, SERVICE_TYPE, REWARD_TYPE, CURRENCY, CURRENCY_TYPE,
    MIN_BALANCE, AVG_BALANCE_REQUIREMENT, MIN_INCOME, MIN_INCOME_USD,
    MIN_INVESTMENT, MIN_CONTRIBUTION, LOAN_AMOUNT_RANGE, COVERAGE_AMOUNT,
    FINANCING_TYPE, DEPOSIT_PROFIT_TYPE, DEPOSIT_PROFIT_FREQUENCY,
    TENURE, TENURE_OPTIONS, MIN_TERM_YEARS, MAX_TERM_YEARS, BUSINESS_TENURE,
    COLLATERAL_TYPE, EQUITY_REQUIREMENT, DBR_LIMIT, TRANSACTION_LIMIT,
    SPECIAL_CONDITIONS, PRODUCT_DESCRIPTION, PROVIDER_NAME, PRODUCT_VARIANT_TIER,
    PRICING_RATE, FEES_AND_CHARGES, KEY_BENEFITS, OPTIONAL_RIDERS,
    FREE_LOOK_PERIOD_DAYS, REQUIRED_DOCUMENTS, CLAIMS_SERVICE_CONTACT,
    KEY_EXCLUSIONS, TAX_ZAKAT_TREATMENT, PREMIUM_PAYMENT_FREQUENCY

15. All multi-value list fields use " | " as separator (not ";" or ","):
    KEY_BENEFITS, OPTIONAL_RIDERS, KEY_EXCLUSIONS, REQUIRED_DOCUMENTS,
    SPECIAL_CONDITIONS, PREMIUM_PAYMENT_FREQUENCY, TENURE_OPTIONS, COVERAGE_AMOUNT

If ANY rule is violated, return CORRECTED JSON. Otherwise return JSON unchanged.
Fix ONLY the violations, preserve everything else.
Return ONLY valid JSON, no explanations."""

    return f"{SYSTEM_PROMPT}\n\n{validation_instructions}"


# ============================================================================
# Model inference (unchanged from original)
# ============================================================================

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_raw_generation(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    if getattr(model.config, "is_encoder_decoder", False):
        generated_ids = output_ids[0]
    else:
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    del inputs, output_ids, generated_ids
    free_gpu_memory()

    return text


# ============================================================================
# HYBRID RAG EXTRACTION (new path)
# ============================================================================

def extract_group(
    model,
    tokenizer,
    entry: dict,
    doc_index: DocumentIndex,
    group_name: str,
    group_fields: list[str],
) -> dict[str, dict]:
    """
    Retrieve relevant chunks for this group, run focused extraction,
    return evidence dict mapping field → {value, confidence, evidence, chunk_id}.
    """
    query = FIELD_GROUP_QUERIES[group_name]
    retrieved = doc_index.retrieve(query, k=HYBRID_TOP_K)

    if not retrieved:
        # Fallback: use first 4 chunks
        retrieved = doc_index.chunks[:HYBRID_TOP_K]

    prompt = build_group_extraction_prompt(
        group_name, group_fields, entry, retrieved, tokenizer
    )
    raw = get_raw_generation(model, tokenizer, prompt)

    try:
        parsed = parse_json_blob(raw)
    except Exception:
        # Attempt repair
        repair_prompt = build_repair_prompt(entry, raw)
        repaired_raw = get_raw_generation(model, tokenizer, repair_prompt)
        try:
            parsed = parse_json_blob(repaired_raw)
        except Exception as e:
            print(f"    Warning: group '{group_name}' failed to parse after repair: {e}")
            return {}

    # Attach evidence tracking
    evidence = _extract_with_evidence(parsed, retrieved)
    return evidence


def extract_one_product_hybrid(model, tokenizer, entry: dict, text: str) -> dict:
    """
    Full Hybrid RAG extraction pipeline:
    1. Build per-document token-chunk index (FAISS + BM25)
    2. For each field group: retrieve relevant chunks → focused extraction
    3. Merge all groups using confidence-weighted strategy
    4. Run one final validation pass
    5. Normalize and return
    6. Destroy the document index
    """
    doc_index = DocumentIndex(text)
    doc_index.build()

    # Confidence-tracking accumulator: col → {value, confidence}
    accumulated: dict[str, dict] = {
        col: {"value": DEFAULT_VALUE, "confidence": 0.0} for col in COLUMNS
    }

    for group_name, group_fields in FIELD_GROUPS.items():
        print(f"    Extracting group: {group_name} ({len(group_fields)} fields)…")
        group_evidence = extract_group(
            model, tokenizer, entry, doc_index, group_name, group_fields
        )
        accumulated = _confidence_merge(accumulated, group_evidence, group_fields)
        free_gpu_memory()

    # Flatten to plain dict for normalization and validation
    merged = _flatten_accumulated(accumulated, COLUMNS)
    merged_normalized = normalize_record(merged, entry)

    # Final validation pass (full prompt with all 56 fields)
    validation_prompt = build_validation_prompt(entry, merged_normalized)
    validation_raw = get_raw_generation(model, tokenizer, validation_prompt)
    try:
        validated_parsed = parse_json_blob(validation_raw)
        result = normalize_record(validated_parsed, entry)
    except Exception:
        result = merged_normalized

    # Destroy document index — no cross-product contamination
    doc_index.destroy()
    free_gpu_memory()

    return result


# ============================================================================
# LEGACY CHUNKED EXTRACTION (unchanged, used when ENABLE_HYBRID_RAG=False)
# ============================================================================

def merge_records(accumulated: dict, new_record: dict, columns) -> dict:
    """Legacy first-non-null-wins merge."""
    for col in columns:
        old = accumulated.get(col)
        new = new_record.get(col)
        old_is_empty = old is None or old == "" or old == DEFAULT_VALUE
        new_is_real = new not in (None, "", DEFAULT_VALUE)
        if old_is_empty and new_is_real:
            accumulated[col] = new
    return accumulated


def extract_one_chunk(model, tokenizer, entry, chunk, chunk_idx, chunk_total):
    """Legacy single-chunk extraction with repair fallback."""
    prompt = build_prompt(entry, chunk, tokenizer, chunk_idx, chunk_total)
    raw = get_raw_generation(model, tokenizer, prompt)

    try:
        parsed = parse_json_blob(raw)
        return normalize_record(parsed, entry)
    except Exception:
        repair_prompt = build_repair_prompt(entry, raw)
        repaired_raw = get_raw_generation(model, tokenizer, repair_prompt)
        try:
            repaired_parsed = parse_json_blob(repaired_raw)
            return normalize_record(repaired_parsed, entry)
        except Exception as second_error:
            print(
                f"  Warning: chunk {chunk_idx}/{chunk_total} unparseable after "
                f"repair attempt, skipping this chunk: {second_error}"
            )
            return None


def extract_one_product_legacy(model, tokenizer, entry, text):
    """Legacy chunked extraction path (ENABLE_HYBRID_RAG=False)."""
    if ENABLE_CHUNKING:
        chunks = chunk_text(text, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
    else:
        chunks = [text.strip()]
    chunk_total = len(chunks)

    accumulated = blank_record(entry)
    any_chunk_succeeded = False

    for i, chunk in enumerate(chunks, start=1):
        chunk_record = extract_one_chunk(model, tokenizer, entry, chunk, i, chunk_total)
        if chunk_record is not None:
            any_chunk_succeeded = True
            accumulated = merge_records(accumulated, chunk_record, COLUMNS)
        free_gpu_memory()

    if not any_chunk_succeeded:
        return blank_record(entry)

    validation_prompt = build_validation_prompt(entry, accumulated)
    validation_raw = get_raw_generation(model, tokenizer, validation_prompt)
    try:
        validated_parsed = parse_json_blob(validation_raw)
        return normalize_record(validated_parsed, entry)
    except Exception:
        return accumulated


# ============================================================================
# UNIFIED ENTRY POINT
# ============================================================================

def extract_one(model, tokenizer, entry, text):
    """
    Main extraction entry point. Routes to Hybrid RAG or legacy pipeline
    based on ENABLE_HYBRID_RAG environment variable.
    """
    if ENABLE_HYBRID_RAG:
        return extract_one_product_hybrid(model, tokenizer, entry, text)
    else:
        return extract_one_product_legacy(model, tokenizer, entry, text)


# ============================================================================
# Model loader (unchanged from original)
# ============================================================================

def make_generator():
    if not MODEL_NAME:
        raise SystemExit(
            "Set HF_MODEL_NAME_OR_PATH in .env to a local Hugging Face model name or folder."
        )

    if LOAD_IN_4BIT or LOAD_IN_8BIT:
        try:
            bnb_version = version("bitsandbytes")
        except PackageNotFoundError as exc:
            raise SystemExit(
                "4-bit/8-bit loading needs bitsandbytes.\n"
                "Run this in Colab, then restart runtime:\n"
                "pip install -U 'bitsandbytes>=0.46.1'"
            ) from exc
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", bnb_version)
        major, minor, patch = (int(part) for part in match.groups()) if match else (0, 0, 0)
        if (major, minor, patch) < (0, 46, 1):
            raise SystemExit(
                f"bitsandbytes {bnb_version} is too old for 4-bit loading.\n"
                "Run this in Colab, then restart runtime:\n"
                "pip install -U 'bitsandbytes>=0.46.1'"
            )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
    except RepositoryNotFoundError as exc:
        raise SystemExit(
            f"Model not found on Hugging Face: {MODEL_NAME}\n"
            "This usually means the repo id is wrong.\n"
            "Use the exact Hugging Face model id from the model card, or set "
            "HF_MODEL_NAME_OR_PATH to a local folder path.\n"
            "If the repo is private or gated, authenticate in Colab first."
        ) from exc

    model_kwargs: dict = {
        "local_files_only": LOCAL_FILES_ONLY,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if DEVICE_MAP.lower() != "none":
        model_kwargs["device_map"] = DEVICE_MAP

        if torch.cuda.is_available() and DEVICE_MAP.lower() == "auto":
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                usable_gib = max(total_gib - GPU_RESERVE_GIB, 1.0)
                max_memory[i] = f"{usable_gib:.1f}GiB"
            max_memory["cpu"] = os.environ.get("HF_CPU_MEMORY", "48GiB")
            model_kwargs["max_memory"] = max_memory
            print(f"GPU headroom reserved: {GPU_RESERVE_GIB} GiB (max_memory={max_memory})")

    if TORCH_DTYPE == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif TORCH_DTYPE == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif TORCH_DTYPE == "float32":
        model_kwargs["torch_dtype"] = torch.float32
    elif TORCH_DTYPE == "auto":
        model_kwargs["torch_dtype"] = "auto"

    if LOAD_IN_4BIT or LOAD_IN_8BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {MODEL_NAME}")
    print(f"Model class:   {MODEL_CLASS}")
    print(f"Device map:    {DEVICE_MAP}")
    print(f"Torch dtype:   {TORCH_DTYPE}")
    print(f"4-bit quant:   {LOAD_IN_4BIT}")
    print(f"8-bit quant:   {LOAD_IN_8BIT}")

    if MODEL_CLASS == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if hasattr(model, "generation_config"):
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_done_ids():
    done: set[int] = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["product_no"])
                except Exception:
                    pass
    return done


# ============================================================================
# Main
# ============================================================================

def main():
    pipeline_mode = "Hybrid RAG" if ENABLE_HYBRID_RAG else "Legacy Chunked"
    print(f"\n  Pipeline mode: {pipeline_mode}")

    input_files = select_input_files()
    index = build_index_from_files(input_files)

    print(f"\n  {len(index)} file(s) queued for extraction.")
    print(f"  Output will be appended to: {OUT_JSONL}\n")

    try:
        model, tokenizer = make_generator()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA ran out of memory while loading the model.\n"
            "For Colab T4/L4 GPUs, use these .env settings:\n"
            "HF_LOAD_IN_4BIT=true\n"
            "HF_DEVICE_MAP=auto\n"
            "HF_TORCH_DTYPE=float16\n"
            "HF_GPU_RESERVE_GIB=3.0\n"
            "You can also use a smaller model like Qwen/Qwen2.5-3B-Instruct."
        ) from exc

    done = load_done_ids()
    print(f"{len(index)} products total, {len(done)} already extracted")

    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        for entry in index:
            if entry["product_no"] in done:
                continue

            abs_path: Path = entry["_abs_path"]

            try:
                text = read_file_text(abs_path)
            except Exception as read_err:
                print(
                    f"[{entry['product_no']:03d}/{len(index)}] SKIP  "
                    f"Could not read '{abs_path.name}': {read_err}"
                )
                continue

            if not text.strip():
                print(
                    f"[{entry['product_no']:03d}/{len(index)}] SKIP  "
                    f"'{abs_path.name}' produced no text after reading."
                )
                continue

            for attempt in range(3):
                try:
                    data = extract_one(model, tokenizer, entry, text)
                    rec = {"product_no": entry["product_no"], **data}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(
                        f"[{entry['product_no']:03d}/{len(index)}] OK  "
                        f"{entry['title'][:60]}  ({abs_path.name})"
                    )
                    break
                except torch.cuda.OutOfMemoryError as e:
                    free_gpu_memory()
                    print(
                        f"[{entry['product_no']:03d}/{len(index)}] "
                        f"CUDA OOM on attempt {attempt + 1}, cleared cache and retrying: {e}"
                    )
                    time.sleep(5)
                except Exception as e:
                    free_gpu_memory()
                    print(
                        f"[{entry['product_no']:03d}/{len(index)}] retry {attempt + 1}: {e}"
                    )
                    time.sleep(3)
            else:
                print(f"[{entry['product_no']:03d}/{len(index)}] FAILED after retries")

            free_gpu_memory()

    print("Done. Output:", OUT_JSONL)


if __name__ == "__main__":
    main()