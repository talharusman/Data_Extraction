"""
 Step 2: Extract the fixed 56-column schema as JSON from product files.
ENHANCED with 3 advanced prompting techniques:
  1. Few-shot examples showing correct vs incorrect extraction
  2. Step-by-step extraction workflow
  3. Self-validation with error correction

SYSTEM PROMPT IS READ FROM: EXTRACTION_SYSTEM_PROMPT.txt

Instead of reading a pre-built JSON index, the script now asks you at startup
how you want to supply input:

  1) Single file   – give one file path (any supported type)
  2) Folder        – all supported files in one directory (non-recursive)
  3) Nested folder – all supported files under a directory tree (recursive)

Supported file types: .txt, .pdf, .docx, .doc, .csv, .json, .xlsx, .xls

Required packages for non-txt formats:
  pip install pdfplumber python-docx openpyxl
  # for .doc: sudo apt install antiword  (Linux) or install antiword on PATH

Configure the model through .env:
  HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
  HF_MODEL_CLASS=causal
  HF_LOCAL_FILES_ONLY=true

Resumable: already-extracted products (present in OUT_JSONL) are skipped,
so you can safely re-run after an interruption.

FIXED VERSION: All three prompt functions now properly use SYSTEM_PROMPT
- build_prompt(): Already correct, no changes
- build_repair_prompt(): FIXED - now includes SYSTEM_PROMPT + typo removed
- build_validation_prompt(): FIXED - now includes SYSTEM_PROMPT

FIX (SOURCE_FILE_PRODUCT): This field used to store the full absolute file
path (e.g. ".../Documents/Banca Takaful/Alfalah Insurance Zaamin Takaful
Plan.docx"). It now stores just the source filename (e.g. "Plan.docx")
instead. Changes are in build_index_from_files(), blank_record(), and
normalize_record() - search for "filename" to find them.
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

# FIX: Reduce CUDA memory fragmentation on small GPUs (Colab T4/L4 ~15GB).
# Must be set before the CUDA context is created, so it goes before `import torch`.
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

MODEL_NAME = os.environ.get("HF_MODEL_NAME_OR_PATH", "").strip()
MODEL_CLASS = os.environ.get("HF_MODEL_CLASS", "causal").strip().lower()
LOCAL_FILES_ONLY = env_bool("HF_LOCAL_FILES_ONLY", True)
TRUST_REMOTE_CODE = env_bool("HF_TRUST_REMOTE_CODE", False)
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 1000)
TEMPERATURE = env_float("HF_TEMPERATURE", 0.0)
TOP_P = env_float("HF_TOP_P", 1.0)
REPETITION_PENALTY = env_float("HF_REPETITION_PENALTY", 1.03)
# FIX: TEXT_CHUNK_SIZE used to be 15000 chars and was sent in ONE shot
# together with the ~8000-token system prompt, which spiked prefill memory
# enough to OOM on a single generate() call before any token was even
# produced. It's now the size of each CHUNK (documents longer than this are
# split into multiple smaller calls and the results are merged - see
# chunk_text() / extract_one_product()).
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", 6000)
TEXT_CHUNK_OVERLAP = env_int("TEXT_CHUNK_OVERLAP", 300)
LOAD_IN_4BIT = env_bool("HF_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = env_bool("HF_LOAD_IN_8BIT", False)
DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
TORCH_DTYPE = os.environ.get("HF_TORCH_DTYPE", "auto").strip().lower()

# FIX (root cause of the "OOM on the very first generate() call" behavior):
# With device_map="auto", Accelerate is free to pack model weights onto the
# GPU up to (almost) its full physical capacity, because by default it has
# no idea you also need room left over for the KV-cache/activations that
# generate() allocates on top of the weights. On a ~14-15GB Colab GPU this
# means the model alone can end up occupying 13+ GB, leaving well under 1GB
# free - so the very first generate() call (which needs a few hundred MB for
# activations) fails immediately, before it even gets to process a large
# prompt. HF_GPU_RESERVE_GIB tells the loader to hold back this much GPU
# memory from the weight placement step, guaranteeing headroom for inference.
GPU_RESERVE_GIB = env_float("HF_GPU_RESERVE_GIB", 3.0)

# FIX: chunking is what lets one call's prefill stay small instead of
# stuffing (system prompt + entire document) into a single generate() call.
# It is kept as a togglable flag (not deleted) because turning it off just
# brings back the original large-single-prompt OOM risk on the longer
# documents in this dataset - see the note in extract_one_product() below.
ENABLE_CHUNKING = env_bool("ENABLE_CHUNKING", True)
DEFAULT_VALUE = "N/A"
NUMERIC_COLUMNS = {
    "MIN_AGE",
    "MAX_AGE",
    "IS_BANK_OFFERED",
    "MIN_BALANCE",
    "AVG_BALANCE_REQUIREMENT",
    "MIN_INCOME",
    "MIN_INCOME_USD",
    "MIN_INVESTMENT",
    "MIN_CONTRIBUTION",
    "MIN_TERM_YEARS",
    "MAX_TERM_YEARS",
    "FREE_LOOK_PERIOD_DAYS",
}

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc", ".csv", ".json", ".xlsx", ".xls"}

# ============================================================================
# LOAD SYSTEM PROMPT FROM SEPARATE FILE
# ============================================================================

def load_system_prompt(prompt_file: str = "EXTRACTION_SYSTEM_PROMPT.txt") -> str:
    """
    Load the system prompt from a separate file.

    Looks for the prompt file in this order:
    1. Current directory
    2. Same directory as this script
    3. Parent directory

    If not found, raises an error.
    """
    search_paths = [
        Path(prompt_file),
        Path(__file__).parent / prompt_file,
        Path(__file__).parent.parent / prompt_file,
    ]

    for prompt_path in search_paths:
        if prompt_path.exists() and prompt_path.is_file():
            print(f"✓ Loaded system prompt from: {prompt_path.resolve()}")
            return prompt_path.read_text(encoding="utf-8")

    # Not found - provide helpful error message
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


# Load the system prompt at module level
SYSTEM_PROMPT = load_system_prompt()


# ============================================================================
# File reading functions
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
# Interactive input selection
# ============================================================================

def _prompt_choice(prompt: str, choices: list[str]) -> str:
    """Ask the user to pick from a numbered list."""
    while True:
        print(prompt)
        for i, choice in enumerate(choices, 1):
            print(f"  {i}) {choice}")
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(choices)}.\n")


def _collect_files(path: Path, recursive: bool) -> list[Path]:
    """Return all supported files under *path*."""
    if recursive:
        all_files = path.rglob("*")
    else:
        all_files = path.glob("*")
    return sorted(
        f for f in all_files
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def select_input_files() -> list[Path]:
    """Interactively ask user how to supply input files."""
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
                print(
                    f"  Unsupported file type '{p.suffix}'. "
                    f"Supported: {supported_str}\n"
                )
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
                print(
                    f"  No supported files found directly in '{p}'.\n"
                    f"  Supported formats: {supported_str}\n"
                )
            else:
                print(f"  Not a valid directory: {raw}\n")

    else:  # Nested folder (recursive)
        while True:
            raw = input("\nEnter the root folder path: ").strip()
            p = Path(raw)
            if p.is_dir():
                files = _collect_files(p, recursive=True)
                if files:
                    print(
                        f"  Found {len(files)} supported file(s) under '{p}' "
                        f"(all sub-folders included)."
                    )
                    return files
                print(
                    f"  No supported files found anywhere under '{p}'.\n"
                    f"  Supported formats: {supported_str}\n"
                )
            else:
                print(f"  Not a valid directory: {raw}\n")


def build_index_from_files(files: list[Path]) -> list[dict]:
    """Build index from file paths."""
    index = []
    for i, path in enumerate(files, start=1):
        index.append(
            {
                "product_no": i,
                "title": path.stem.replace("_", " ").replace("-", " "),
                "file": str(path),
                # FIX: added so SOURCE_FILE_PRODUCT can store just the
                # immediate parent folder name (e.g. "Banca Takaful")
                # instead of the full absolute file path.
                "folder": path.parent.name,
                "_abs_path": path.resolve(),
            }
        )
    return index


# ============================================================================
# Document chunking (FIX for large-document OOM)
# ============================================================================

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long document into smaller pieces so each model call only has to
    process chunk_size characters instead of the whole document at once.

    Breaks are made on a paragraph/sentence boundary near chunk_size where
    possible (instead of a hard character cut) so a field's value isn't
    split mid-sentence across two chunks. A small overlap is carried into
    the next chunk so context right at a boundary isn't lost.
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


def merge_records(accumulated: dict, new_record: dict, columns) -> dict:
    """
    Merge a single chunk's extracted record into the running accumulated
    record for the product. A field is filled in from this chunk only if it
    hasn't already been found (still DEFAULT_VALUE/empty) in an earlier
    chunk - first chunk to find a real value for a field wins.
    """
    for col in columns:
        old = accumulated.get(col)
        new = new_record.get(col)
        old_is_empty = old is None or old == "" or old == DEFAULT_VALUE
        new_is_real = new not in (None, "", DEFAULT_VALUE)
        if old_is_empty and new_is_real:
            accumulated[col] = new
    return accumulated


# ============================================================================
# JSON parsing helpers
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
        yield from emit(text[first : last + 1])

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
                        yield from emit(text[brace_start : idx + 1])
                        break


def parse_json_blob(raw):
    for candidate in _iter_json_candidates(raw):
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = _parse_candidate(cleaned)
        if parsed is not None:
            return parsed
    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


# ============================================================================
# Record helpers with aggressive normalization
# ============================================================================

def blank_record(entry):
    record = {col: DEFAULT_VALUE for col in COLUMNS}
    record["PRODUCT_NAME"] = entry["title"]
    # FIX: was entry["file"] (full absolute path). Now stores just the
    # immediate parent folder name, e.g. "Banca Takaful".
    record["SOURCE_FILE_PRODUCT"] = entry["folder"]
    return record


field_max_lengths = {
    "PRODUCT_NAME": 50,
    "PRODUCT_DESCRIPTION": 250,
    "PROVIDER_NAME": 100,
    "PRODUCT_VARIANT_TIER": 50,
    "PRICING_RATE": 50,
    "FEES_AND_CHARGES": 200,
    "KEY_BENEFITS": 200,
    "OPTIONAL_RIDERS": 200,
    "REQUIRED_DOCUMENTS": 200,
    "CLAIMS_SERVICE_CONTACT": 200,
    "KEY_EXCLUSIONS": 200,
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 15,
    "TARGET_GOAL": 30,
    "CUSTOMER_TYPE": 25,
    "EMPLOYMENT_TYPE": 20,
    "ACCOUNT_TYPE": 20,
    "CARD_TYPE": 25,
    "CHANNEL": 30,
    "ELIGIBILITY_TYPE": 50,
    "SERVICE_TYPE": 25,
    "REWARD_TYPE": 25,
    "CURRENCY": 30,
    "CURRENCY_TYPE": 15,
    "LOAN_AMOUNT_RANGE": 50,
    "COVERAGE_AMOUNT": 50,
    "FINANCING_TYPE": 30,
    "DEPOSIT_PROFIT_TYPE": 30,
    "DEPOSIT_PROFIT_FREQUENCY": 20,
    "TENURE": 30,
    "TENURE_OPTIONS": 50,
    "BUSINESS_TENURE": 30,
    "COLLATERAL_TYPE": 50,
    "EQUITY_REQUIREMENT": 20,
    "DBR_LIMIT": 20,
    "TRANSACTION_LIMIT": 50,
    "SPECIAL_CONDITIONS": 200,
}


def truncate_to_boundary(value: str, max_len: int) -> str:
    """Truncate text at a word or punctuation boundary when possible."""
    if len(value) <= max_len:
        return value

    cut = value[:max_len].rstrip()
    boundary = max(cut.rfind(" "), cut.rfind(","), cut.rfind(";"), cut.rfind(":"), cut.rfind("-"))
    if boundary > 0:
        return cut[:boundary].rstrip(" ,;:-/")
    return cut


def normalize_record(record, entry):
    """Normalize extracted record with aggressive conciseness enforcement."""
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}

    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)

        if value in (None, "", []):
            value = DEFAULT_VALUE

        # Numeric columns: numbers ONLY
        if col in NUMERIC_COLUMNS and isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.upper() == DEFAULT_VALUE:
                value = DEFAULT_VALUE
            else:
                match = re.match(r'^(\d+(?:\.\d+)?)', stripped)
                if match:
                    value = match.group(1)
                else:
                    value = DEFAULT_VALUE

        # Truncate verbose text
        if col in field_max_lengths and isinstance(value, str):
            max_len = field_max_lengths[col]
            if len(value) > max_len:
                value = truncate_to_boundary(value, max_len)

        # Special case: CUSTOMER_TYPE - ONE value only
        if col == "CUSTOMER_TYPE" and isinstance(value, str):
            if "," in value and len(value) > 25:
                first_item = value.split(",")[0].strip()
                if len(first_item) < 25:
                    value = first_item
                else:
                    value = DEFAULT_VALUE

        # Special case: GENDER - standardize
        if col == "GENDER" and isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ("male", "m"):
                value = "Male"
            elif value_lower in ("female", "f"):
                value = "Female"
            elif value_lower in ("all", "both"):
                # FIX: spec requires exactly "All" (see EXTRACTION_SYSTEM_PROMPT.txt
                # rule #13) - this was writing "ALL", which fails validation.
                value = "All"
            elif value_lower not in ("male", "female", "all"):
                value = DEFAULT_VALUE

        # Special case: PLAN_TYPE - single word
        if col == "PLAN_TYPE" and isinstance(value, str):
            valid_types = {"Loan", "Deposit", "Savings", "Card", "Investment", "Insurance", "Service", "Loyalty"}
            words = value.strip().split()
            found = False
            for word in words:
                if word in valid_types:
                    value = word
                    found = True
                    break
            if not found:
                value = DEFAULT_VALUE

        normalized[col] = value

    normalized["PRODUCT_NAME"] = record.get("PRODUCT_NAME") or entry["title"]
    normalized["SOURCE_FILE_PRODUCT"] = entry["filename"]

    return normalized


# ============================================================================
# Prompt builders - using the loaded system prompt (ALL FIXED)
# ============================================================================

def build_prompt(entry, chunk, tokenizer, chunk_idx=1, chunk_total=1):
    """Build prompt with step-by-step extraction workflow for ONE chunk of
    the document. `chunk` is already cut to size by the caller (chunk_text())
    - do not slice it again here, that's the whole point of chunking."""

    if chunk_total > 1:
        chunk_note = (
            f"\nNOTE: This is PART {chunk_idx} of {chunk_total} of a single, longer "
            f"product document (it has been split only because of length, not because "
            f"it is a different product). Extract whatever fields you can find in THIS "
            f"part only. If a field is not mentioned in this part, set it to \"N/A\" - "
            f"it may simply be described in another part of the same document.\n"
        )
    else:
        chunk_note = ""

    user_msg = f"""Product title: {entry['title']}
Source file: {entry['filename']}
{chunk_note}--- PRODUCT TEXT START ---
{chunk}
--- PRODUCT TEXT END ---

Return only one JSON object that matches the system prompt exactly."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass

    return f"{SYSTEM_PROMPT}\n\n{user_msg}"


def build_repair_prompt(entry, raw_text):
    """Build repair prompt using the full system context. (FIXED - now includes SYSTEM_PROMPT)"""
    repair_instructions = f"""You are repairing a failed extraction.
Return exactly ONE valid JSON object and nothing else.

Product: {entry['title']}

Broken output to repair:
{raw_text[:5000]}

Apply ALL rules from the system prompt above. Focus on:
- Use "N/A" for missing fields
- Keep values SHORT and within character limits
- CUSTOMER_TYPE: ONE value only: Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A
- GENDER: Must be Male, Female, All, or N/A only
- PLAN_TYPE: ONE word: Loan|Deposit|Savings|Card|Investment|Insurance|Service|Loyalty
- Numeric fields: NUMBERS ONLY (no currency, no text)
- New detail fields must stay concise:
    PRODUCT_DESCRIPTION, PROVIDER_NAME, PRODUCT_VARIANT_TIER, PRICING_RATE,
    FEES_AND_CHARGES, KEY_BENEFITS, OPTIONAL_RIDERS, REQUIRED_DOCUMENTS,
    CLAIMS_SERVICE_CONTACT, KEY_EXCLUSIONS, TAX_ZAKAT_TREATMENT,
- No markdown fences or explanations

Return the repaired JSON object now."""

    # Return with SYSTEM_PROMPT prepended for full context
    return f"{SYSTEM_PROMPT}\n\n{repair_instructions}"


def build_validation_prompt(entry, extracted_json):
    """Build prompt for self-validation and correction using full system context. (FIXED - now includes SYSTEM_PROMPT)"""
    validation_instructions = f"""You extracted this JSON. Validate and fix any issues:

EXTRACTED JSON:
{json.dumps(extracted_json, indent=2)}

VALIDATION RULES - Check each rule and FIX if violated:

1. CUSTOMER_TYPE: Is it ONE value only (no commas)?
   ✓ "Salaried"  ✗ "Salaried, Self-Employed, Corporate"

2. GENDER: Is it exactly one of: Male|Female|All|N/A?
   ✓ "Female"  ✗ "Male or Female"

3. PLAN_TYPE: Is it ONE word from: Loan|Deposit|Savings|Card|Investment|Insurance|Service|Loyalty?
   ✓ "Savings"  ✗ "Savings and Protection"

4. Text field lengths - Are they within limits?
   - PRODUCT_NAME ≤ 50 chars
    - PRODUCT_DESCRIPTION ≤ 250 chars
    - PROVIDER_NAME ≤ 100 chars
    - PRODUCT_VARIANT_TIER ≤ 50 chars
    - PRICING_RATE ≤ 50 chars
    - FEES_AND_CHARGES ≤ 200 chars
    - KEY_BENEFITS ≤ 200 chars
    - OPTIONAL_RIDERS ≤ 200 chars
    - REQUIRED_DOCUMENTS ≤ 200 chars
    - CLAIMS_SERVICE_CONTACT ≤ 200 chars
    - KEY_EXCLUSIONS ≤ 200 chars
    - TAX_ZAKAT_TREATMENT ≤ 100 chars
    - PDF_PAGE_REFERENCE ≤ 50 chars
   - CUSTOMER_TYPE ≤ 25 chars
   - TENURE ≤ 30 chars
   - SPECIAL_CONDITIONS ≤ 200 chars

5. Numeric fields - Do they contain ONLY numbers (no currency/text)?
    ✓ "250000"  ✗ "PKR 250,000" or "25000 per month"
    - FREE_LOOK_PERIOD_DAYS is numeric too

6. SPECIAL_CONDITIONS - Is it concise, max 200 chars?
   ✓ "Free 14-day look, waiting period 90 days"
   ✗ "Free 14-day look period, optional riders, death benefit includes sum covered plus PIA value, maturity benefit..."

Refer to the system prompt above for full field definitions and critical rules.

If ANY rule is violated, return CORRECTED JSON. Otherwise return JSON unchanged.
Fix ONLY the violations, keep everything else as-is.
Return ONLY valid JSON, no explanations."""

    # Return with SYSTEM_PROMPT prepended for full context
    return f"{SYSTEM_PROMPT}\n\n{validation_instructions}"


# ============================================================================
# Model inference
# ============================================================================

def free_gpu_memory():
    """
    FIX: Release cached/fragmented CUDA memory between generate() calls.
    PyTorch's caching allocator keeps freed tensors reserved for reuse, but
    with varying prompt/KV-cache lengths across calls that cache fragments
    instead of being reused, eventually exhausting the GPU. Without this,
    successive products on a small Colab GPU (T4/L4, ~15GB) run out of
    memory after just 1-2 products even though each individual call would
    fit in memory on its own.

    IMPORTANT LIMITATION (why this only ever frees ~1-2GB, not more):
    empty_cache() can only return memory that is RESERVED-but-UNUSED back to
    the CUDA driver - i.e. leftover KV-cache/activation buffers from past
    generate() calls that PyTorch is holding "just in case". It cannot
    touch memory backing a live tensor. The model's weights stay resident
    for the entire run, so no amount of gc.collect()/empty_cache() will ever
    shrink that footprint. The only ways to reduce it are: use a smaller or
    more heavily quantized model, or cap how much of the GPU the loader is
    allowed to fill in the first place (see GPU_RESERVE_GIB above). This
    function fixes cross-call fragmentation; it cannot fix a model that is
    simply too big for the GPU.
    """
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

    # FIX: explicitly drop references to the input/output tensors (which can
    # hold a full KV cache worth of GPU memory) and clear the CUDA cache
    # before returning. Without this, every generate() call across every
    # product leaves fragmented memory behind, which is why later products
    # in a batch OOM even though the first one succeeded.
    del inputs, output_ids, generated_ids
    free_gpu_memory()

    return text


def extract_one_chunk(model, tokenizer, entry, chunk, chunk_idx, chunk_total):
    """
    Run extraction on a SINGLE chunk of the document (initial attempt, with
    one repair retry if the model's output isn't parseable JSON). No
    self-validation here on purpose - validation is run once on the final
    merged record instead, to avoid multiplying generate() calls by the
    number of chunks.
    Returns a normalized record dict, or None if both attempts failed to
    produce parseable JSON for this chunk (the chunk is then simply skipped;
    other chunks may still cover those fields).
    """
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


def extract_one_product(model, tokenizer, entry, text):
    """
    FIX (large-document OOM): instead of sending the ENTIRE document text in
    one giant prompt (which on top of the long system prompt was enough to
    blow past available GPU memory during a single prefill), the document is
    split into chunk_text()-sized pieces and each piece is sent to the model
    SEPARATELY. Results are merged field-by-field (merge_records picks the
    first real value found for each field across chunks), so a field
    mentioned anywhere in the document is still captured even though no
    single call ever sees the whole document at once.

    A single self-validation pass runs once on the final merged record
    (not once per chunk) to keep the number of generate() calls roughly the
    same as before for short documents, while still being far cheaper than
    validating every chunk for long ones.

    If ENABLE_CHUNKING=False, this falls back to sending the whole document
    in one call. That is a real regression risk: this dataset's longer PDFs
    combined with the ~8k-token system prompt are exactly what caused the
    original single-shot OOM, so only disable this if GPU_RESERVE_GIB /
    HF_LOAD_IN_4BIT headroom fixes below have already been verified to give
    enough free VRAM for the longest document in your batch.
    """
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

    # Single validation/correction pass on the merged record.
    validation_prompt = build_validation_prompt(entry, accumulated)
    validation_raw = get_raw_generation(model, tokenizer, validation_prompt)
    try:
        validated_parsed = parse_json_blob(validation_raw)
        return normalize_record(validated_parsed, entry)
    except Exception:
        # If the validation call itself fails to parse, the merged record
        # (already normalized field-by-field) is still a valid result.
        return accumulated


def extract_one(model, tokenizer, entry, text):
    """Main extraction function: chunked extraction + merge + single validation."""
    return extract_one_product(model, tokenizer, entry, text)


# ============================================================================
# Model loader
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
            "Use the exact Hugging Face model id from the model card, or set HF_MODEL_NAME_OR_PATH\n"
            "to a local folder path.\n"
            "If the repo is private or gated, authenticate in Colab first."
        ) from exc

    model_kwargs: dict = {
        "local_files_only": LOCAL_FILES_ONLY,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if DEVICE_MAP.lower() != "none":
        model_kwargs["device_map"] = DEVICE_MAP

        # FIX: cap how much GPU memory Accelerate is allowed to fill with
        # weights, so it always leaves GPU_RESERVE_GIB free for the KV-cache
        # and activation memory that generate() needs on top of the weights.
        # Without this, device_map="auto" happily packs the model up to
        # ~full physical capacity (that's why the log showed 13.71/14.56 GiB
        # already in use immediately after loading, before generation even
        # produced one token).
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
            "HF_GPU_RESERVE_GIB=3.0   # leaves headroom for generate(), see comment above\n"
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
                    # FIX: an OOM mid-generation leaves the allocator in a
                    # fragmented state. Clear it BEFORE retrying, otherwise
                    # each retry starts from an even worse memory state than
                    # the last (which is what caused every retry in the
                    # original run to fail worse than the one before it).
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

            # FIX: also clean up after every product (success or failure),
            # not just on error, so memory doesn't slowly accumulate across
            # a long batch even when nothing technically throws an OOM.
            free_gpu_memory()

    print("Done. Output:", OUT_JSONL)


if __name__ == "__main__":
    main()