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

CHANGES IN THIS VERSION:
- Added PREMIUM_PAYMENT_FREQUENCY as the 56th column (injected if missing from
  pipeline_config to maintain backward compatibility).
- PLAN_TYPE normalization now accepts Protection and Health in addition to the
  original set.
- FINANCING_TYPE normalization added: maps "unit linked", "hybrid" patterns to
  canonical values.
- PRODUCT_NAME normalization: ALL-CAPS product names are converted to Title Case.
- OPTIONAL_RIDERS normalization: ensures comma-separated output.
- Updated field_max_lengths to match corrected-dataset ground-truth lengths
  (PRICING_RATE 400, FEES_AND_CHARGES 300, KEY_BENEFITS 250, OPTIONAL_RIDERS 300,
  TENURE 50, TARGET_GOAL 50, COVERAGE_AMOUNT 150, ELIGIBILITY_TYPE 100,
  EMPLOYMENT_TYPE 500, FINANCING_TYPE 50, PREMIUM_PAYMENT_FREQUENCY 50).
- Repair prompt is now a compact focused version (no full SYSTEM_PROMPT duplication)
  to stay within Qwen2.5-3B's context limit during error recovery.
- Validation prompt updated with corrected field rules (GENDER, DEPOSIT_PROFIT,
  FINANCING_TYPE, TENURE_OPTIONS vs PREMIUM_PAYMENT_FREQUENCY, etc.).
- MAX_NEW_TOKENS default raised to 1500 to accommodate 56-field JSON with
  longer EMPLOYMENT_TYPE and PRICING_RATE values.
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

# ---------------------------------------------------------------------------
# Inject PREMIUM_PAYMENT_FREQUENCY if pipeline_config was not yet updated.
# This keeps the script backward-compatible: older pipeline_config files that
# only define 55 columns will still work; the 56th column is added here.
# ---------------------------------------------------------------------------
_NEW_COLUMNS = ["PREMIUM_PAYMENT_FREQUENCY"]
for _col in _NEW_COLUMNS:
    if _col not in COLUMNS:
        COLUMNS = list(COLUMNS) + [_col]

MODEL_NAME = os.environ.get("HF_MODEL_NAME_OR_PATH", "").strip()
MODEL_CLASS = os.environ.get("HF_MODEL_CLASS", "causal").strip().lower()
LOCAL_FILES_ONLY = env_bool("HF_LOCAL_FILES_ONLY", True)
TRUST_REMOTE_CODE = env_bool("HF_TRUST_REMOTE_CODE", False)
# Raised from 1000 to 1500: 56-field JSON with long EMPLOYMENT_TYPE and
# PRICING_RATE values can approach ~1100 tokens; 1500 gives safe headroom.
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 1500)
TEMPERATURE = env_float("HF_TEMPERATURE", 0.0)
TOP_P = env_float("HF_TOP_P", 1.0)
REPETITION_PENALTY = env_float("HF_REPETITION_PENALTY", 1.03)
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", 6000)
TEXT_CHUNK_OVERLAP = env_int("TEXT_CHUNK_OVERLAP", 300)
LOAD_IN_4BIT = env_bool("HF_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = env_bool("HF_LOAD_IN_8BIT", False)
DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
TORCH_DTYPE = os.environ.get("HF_TORCH_DTYPE", "auto").strip().lower()
GPU_RESERVE_GIB = env_float("HF_GPU_RESERVE_GIB", 3.0)
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
                "filename": path.name,
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
    possible so a field's value isn't split mid-sentence across two chunks.
    A small overlap is carried into the next chunk so context right at a
    boundary isn't lost.
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
    chunk — first chunk to find a real value for a field wins.
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

    salvaged = salvage_json_object(raw)
    if salvaged is not None:
        return salvaged

    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


def salvage_json_object(raw):
    """Recover a best-effort JSON object from line-oriented model output."""
    if not raw:
        return None

    text = _strip_wrappers(raw)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    record = {col: DEFAULT_VALUE for col in COLUMNS}
    found_any = False

    pair_pattern = re.compile(r'^"(?P<key>[^"]+)"\s*:\s*(?P<value>.*?)(?:,)?\s*$')
    for line in lines:
        match = pair_pattern.match(line)
        if not match:
            continue

        key = match.group("key")
        if key not in record:
            continue

        value_text = match.group("value").strip()
        if value_text.endswith(","):
            value_text = value_text[:-1].rstrip()

        if value_text.startswith('"') and value_text.endswith('"') and len(value_text) >= 2:
            try:
                value = json.loads(value_text)
            except Exception:
                value = value_text[1:-1]
        elif value_text.lower() in {"null", "none", "n/a", "na", "nan"}:
            value = DEFAULT_VALUE
        elif value_text.startswith("[") and value_text.endswith("]"):
            try:
                value = json.loads(value_text)
            except Exception:
                value = value_text
        else:
            value = value_text

        record[key] = value
        found_any = True

    return record if found_any else None


# ============================================================================
# Record helpers with normalization
# ============================================================================

def blank_record(entry):
    record = {col: DEFAULT_VALUE for col in COLUMNS}
    record["PRODUCT_NAME"] = entry["title"]
    record["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)
    return record


def get_source_filename(entry) -> str:
    """Return the source filename for prompt/output fields."""
    for key in ("filename", "file"):
        raw = entry.get(key)
        if raw:
            return Path(str(raw)).name

    abs_path = entry.get("_abs_path")
    if abs_path:
        return Path(str(abs_path)).name

    return entry.get("title", DEFAULT_VALUE)


# Updated max lengths to match corrected-dataset ground-truth observations.
# Fields absent from this dict are not truncated (e.g. EMPLOYMENT_TYPE can
# be a long semicolon-separated Target Market list).
field_max_lengths = {
    "PRODUCT_NAME": 50,
    "PRODUCT_DESCRIPTION": 250,
    "PROVIDER_NAME": 100,
    "PRODUCT_VARIANT_TIER": 50,
    "PRICING_RATE": 400,           # age-band pricing tables can be ~370 chars
    "FEES_AND_CHARGES": 300,       # detailed fee schedules up to ~256 chars
    "KEY_BENEFITS": 250,           # benefits list up to ~212 chars
    "OPTIONAL_RIDERS": 300,        # rider lists up to ~253 chars
    "REQUIRED_DOCUMENTS": 200,
    "CLAIMS_SERVICE_CONTACT": 200,
    "KEY_EXCLUSIONS": 200,
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 15,
    "TARGET_GOAL": 50,             # "Children's Education Planning" style values
    "CUSTOMER_TYPE": 25,
    "EMPLOYMENT_TYPE": 500,        # full Target Market list ~334 chars
    "ACCOUNT_TYPE": 20,
    "CARD_TYPE": 25,
    "CHANNEL": 30,
    "ELIGIBILITY_TYPE": 100,       # brief eligibility summaries up to ~52 chars
    "SERVICE_TYPE": 25,
    "REWARD_TYPE": 25,
    "CURRENCY": 30,
    "CURRENCY_TYPE": 15,
    "LOAN_AMOUNT_RANGE": 50,
    "COVERAGE_AMOUNT": 150,        # tier coverage descriptions up to ~94 chars
    "FINANCING_TYPE": 50,          # "Hybrid (Bonus Based and Unit Linked)" = 36 chars
    "DEPOSIT_PROFIT_TYPE": 30,
    "DEPOSIT_PROFIT_FREQUENCY": 20,
    "TENURE": 50,                  # "10-67 years (up to attained age of 85)" = 38 chars
    "TENURE_OPTIONS": 50,
    "BUSINESS_TENURE": 30,
    "COLLATERAL_TYPE": 50,
    "EQUITY_REQUIREMENT": 20,
    "DBR_LIMIT": 20,
    "TRANSACTION_LIMIT": 50,
    "SPECIAL_CONDITIONS": 200,
    "PREMIUM_PAYMENT_FREQUENCY": 50,
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


def _normalize_plan_type(value: str) -> str:
    """
    Normalize PLAN_TYPE to exactly one of the allowed values.
    Now includes Protection and Health in addition to the original set.
    """
    # Expanded set includes Protection and Health added in corrected dataset.
    valid_types = {
        "Loan", "Deposit", "Savings", "Card", "Investment",
        "Insurance", "Service", "Loyalty", "Protection", "Health",
    }
    stripped = value.strip()
    # Exact match first (case-sensitive)
    if stripped in valid_types:
        return stripped
    # Case-insensitive exact match
    for vt in valid_types:
        if stripped.lower() == vt.lower():
            return vt
    # Find the first valid type word inside the value
    for word in re.split(r"[\s,;/]+", stripped):
        word_clean = word.strip(".,;:()")
        if word_clean in valid_types:
            return word_clean
        for vt in valid_types:
            if word_clean.lower() == vt.lower():
                return vt
    return DEFAULT_VALUE


def _normalize_financing_type(value: str) -> str:
    """
    Normalize FINANCING_TYPE to a canonical form.
    Added Unit Linked and Hybrid (Bonus Based and Unit Linked) per corrected dataset.
    """
    stripped = value.strip()
    lower = stripped.lower()

    # Hybrid check first (most specific)
    if "hybrid" in lower or ("bonus" in lower and "unit" in lower):
        return "Hybrid (Bonus Based and Unit Linked)"

    # Unit Linked
    if "unit linked" in lower or "unit-linked" in lower:
        return "Unit Linked"

    # Known single-word canonicals
    canonical_map = {
        "conventional": "Conventional",
        "islamic": "Islamic",
        "takaful": "Takaful",
        "mudarabah": "Mudarabah",
        "mudarabah": "Mudarabah",
    }
    for key, canonical in canonical_map.items():
        if key in lower:
            return canonical

    # Pass through if already in a known exact form
    known_exact = {
        "Conventional", "Islamic", "Takaful", "Mudarabah",
        "Unit Linked", "Hybrid (Bonus Based and Unit Linked)", DEFAULT_VALUE,
    }
    if stripped in known_exact:
        return stripped

    # Unknown value — preserve as-is (don't silently discard it)
    return stripped


def normalize_record(record, entry):
    """
    Normalize an extracted record with corrections for all known model errors.

    Key normalization rules applied here:
    - PRODUCT_NAME: ALL-CAPS converted to Title Case
    - PLAN_TYPE: expanded valid set (Protection, Health now accepted)
    - FINANCING_TYPE: canonical Unit Linked / Hybrid mapping
    - GENDER: strict allowed-value enforcement
    - CUSTOMER_TYPE: single-value enforcement
    - Numeric fields: strip units, commas, currency symbols
    - All text fields: truncated at word boundary to max length
    """
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}

    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)

        if value in (None, "", []):
            value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # Numeric columns: numbers ONLY
        # ----------------------------------------------------------------
        if col in NUMERIC_COLUMNS and isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.upper() == DEFAULT_VALUE:
                value = DEFAULT_VALUE
            else:
                match = re.match(r'^(\d+(?:\.\d+)?)', stripped.replace(",", ""))
                if match:
                    num_str = match.group(1)
                    # Strip trailing ".0"
                    if "." in num_str:
                        try:
                            value = str(int(float(num_str)))
                        except ValueError:
                            value = DEFAULT_VALUE
                    else:
                        value = num_str
                else:
                    value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # PRODUCT_NAME: normalize ALL-CAPS headings to Title Case
        # ----------------------------------------------------------------
        if col == "PRODUCT_NAME" and isinstance(value, str):
            stripped_name = value.strip()
            if (stripped_name
                    and stripped_name == stripped_name.upper()
                    and len(stripped_name.split()) > 1
                    and len(stripped_name) > 5):
                value = stripped_name.title()

        # ----------------------------------------------------------------
        # PLAN_TYPE: expanded valid set
        # ----------------------------------------------------------------
        if col == "PLAN_TYPE" and isinstance(value, str):
            value = _normalize_plan_type(value)

        # ----------------------------------------------------------------
        # FINANCING_TYPE: canonical mapping
        # ----------------------------------------------------------------
        if col == "FINANCING_TYPE" and isinstance(value, str):
            if value not in (DEFAULT_VALUE, ""):
                value = _normalize_financing_type(value)

        # ----------------------------------------------------------------
        # GENDER: strict enforcement of allowed values
        # ----------------------------------------------------------------
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
                # Unrecognized value — do not silently accept
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # CUSTOMER_TYPE: single value only
        # ----------------------------------------------------------------
        if col == "CUSTOMER_TYPE" and isinstance(value, str):
            allowed = {
                "Salaried", "Self-Employed", "SME",
                "Corporate", "Retail", "Government",
            }
            stripped_ct = value.strip()
            if stripped_ct in allowed:
                pass  # already valid
            elif "," in stripped_ct:
                # Multiple values — take first valid token
                first = stripped_ct.split(",")[0].strip()
                value = first if first in allowed else DEFAULT_VALUE
            elif stripped_ct.lower() == "n/a":
                value = DEFAULT_VALUE
            # Note: values like "Salaried Individuals" are not in the allowed set;
            # keep them as-is so the validation pass can flag and fix them.

        # ----------------------------------------------------------------
        # PLAN_TYPE length guard (single word expected)
        # ----------------------------------------------------------------
        # Already handled above by _normalize_plan_type.

        # ----------------------------------------------------------------
        # OPTIONAL_RIDERS: ensure comma-separated (not semicolon-separated)
        # The corrected dataset uses commas for rider lists.
        # ----------------------------------------------------------------
        if col == "OPTIONAL_RIDERS" and isinstance(value, str):
            if value != DEFAULT_VALUE:
                # Replace semicolons with commas if the field is a flat list
                # (i.e. not a descriptive sentence containing semicolons for
                #  different purposes). Heuristic: if no period in the value,
                # it's a list — replace semicolons.
                if "." not in value:
                    value = re.sub(r"\s*;\s*", ", ", value).strip().strip(",").strip()

        # ----------------------------------------------------------------
        # Truncate verbose text fields to max length
        # ----------------------------------------------------------------
        if col in field_max_lengths and isinstance(value, str):
            max_len = field_max_lengths[col]
            if len(value) > max_len:
                value = truncate_to_boundary(value, max_len)

        normalized[col] = value

    # Always preserve PRODUCT_NAME and SOURCE_FILE_PRODUCT
    raw_name = record.get("PRODUCT_NAME") or entry["title"]
    # Apply title-case fix to the preserved name too
    if isinstance(raw_name, str):
        if (raw_name.strip()
                and raw_name.strip() == raw_name.strip().upper()
                and len(raw_name.strip().split()) > 1):
            raw_name = raw_name.strip().title()
    normalized["PRODUCT_NAME"] = raw_name
    normalized["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)

    return normalized


# ============================================================================
# Prompt builders
# ============================================================================

def build_prompt(entry, chunk, tokenizer, chunk_idx=1, chunk_total=1):
    """
    Build the extraction prompt for ONE chunk of the document.
    `chunk` is already cut to size by chunk_text() — do not slice it again.
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
    """
    Compact repair prompt for malformed JSON output.

    Intentionally does NOT include the full SYSTEM_PROMPT to avoid exceeding
    Qwen2.5-3B's context limit when the broken output is also long. The essential
    rules are inlined here instead.
    """
    template_json = json.dumps(blank_record(entry), indent=2, ensure_ascii=False)

    repair_instructions = f"""You are repairing broken JSON from a data extraction task.
Product: {entry['title']}
Source file: {get_source_filename(entry)}

BROKEN OUTPUT TO REPAIR:
{raw_text[:2000]}

EXPECTED JSON TEMPLATE:
{template_json}

REPAIR RULES — apply all of these:
- Return exactly ONE valid JSON object with all fields from the template, nothing else
- Use "N/A" for every missing or unparseable field (never null/None/NaN/"")
- PRODUCT_NAME: Title Case (never ALL CAPS)
- LEAD_MARKER: exactly "IBG" or "BNK"
- PLAN_TYPE: one word from Insurance|Protection|Health|Savings|Deposit|Loan|Card|Investment|Service|Loyalty
- CUSTOMER_TYPE: exactly one of Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A
- GENDER: exactly one of Male|Female|All|N/A
- FINANCING_TYPE: one of Conventional|Islamic|Takaful|Mudarabah|Unit Linked|Hybrid (Bonus Based and Unit Linked)|N/A
- Numeric fields (MIN_AGE, MAX_AGE, MIN_BALANCE, MIN_INCOME, MIN_INCOME_USD,
  MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS, MAX_TERM_YEARS,
  FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED): integers only, no units, no .0
- TENURE_OPTIONS: plan duration choices only, NOT payment frequency
- PREMIUM_PAYMENT_FREQUENCY: how customer pays (Annual/Quarterly/etc.) or "N/A"
- OPTIONAL_RIDERS: comma-separated, not semicolons
- SPECIAL_CONDITIONS: max 200 chars
- SOURCE_FILE_PRODUCT: filename only, no path
- No markdown fences, no explanations outside the JSON

If the broken output already contains most of the JSON, correct only the invalid parts and
keep the existing structure. Do not add prose before or after the object.

Return the repaired JSON object now."""

    messages = [
        {"role": "user", "content": repair_instructions},
    ]

    # Use chat template if available (no system prompt to save tokens)
    if hasattr(entry.get("_tokenizer_ref"), "apply_chat_template"):
        pass  # no tokenizer ref stored in entry; fall through

    return repair_instructions


def build_validation_prompt(entry, extracted_json):
    """
    Validation and correction prompt using the full system context.
    Checks the 56-field output against the corrected extraction rules.
    """
    validation_instructions = f"""You extracted this JSON. Validate and fix any issues:

EXTRACTED JSON:
{json.dumps(extracted_json, indent=2)}

VALIDATION RULES — check each and FIX if violated:

1. PRODUCT_NAME: Is it Title Case? Not ALL CAPS?
   WRONG: "JUBILEE KAMIL TAKAFUL SAVINGS PLAN"
   CORRECT: "Jubilee Kamil Takaful Savings Plan"

2. PLAN_TYPE: Is it ONE word from Insurance|Protection|Health|Savings|Deposit|Loan|Card|Investment|Service|Loyalty?
   Protection plans (accident/theft only) → "Protection"
   Hospitalization plans → "Health"
   Unit-linked savings/endowment → "Savings"

3. CUSTOMER_TYPE: Is it exactly ONE value (no commas)?
   Allowed: Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A

4. GENDER: Is it exactly one of Male|Female|All|N/A?
   "N/A" if gender is not mentioned. "All" ONLY if explicitly stated in document.
   Do NOT use "All" merely because an eligibility section exists.

5. FINANCING_TYPE: For unit-linked plans (PIA, fund allocation) → "Unit Linked"
   Hybrid (bonus + unit-linked) → "Hybrid (Bonus Based and Unit Linked)"
   NOT "N/A" for plans that explicitly mention unit-linked structure.

6. DEPOSIT_PROFIT_TYPE / DEPOSIT_PROFIT_FREQUENCY: Unit-linked plans → both "N/A"
   Health/protection plans (no savings) → both "N/A"
   Do NOT set "At Maturity" for unit-linked plans.

7. TENURE_OPTIONS: Is it ONLY plan duration choices (e.g. "10, 15, 20 years")?
   Payment frequencies ("Annual, Quarterly") belong in PREMIUM_PAYMENT_FREQUENCY.
   If no distinct plan duration menu → "N/A"

8. PREMIUM_PAYMENT_FREQUENCY: Is it the payment frequency (Annual/Semi-Annual/Quarterly/Monthly)?
   Example: "Annual, Semi-Annual, Quarterly" or "N/A"

9. OPTIONAL_RIDERS: Are they comma-separated (not semicolons)?
   WRONG: "Accidental Death; Income Benefit"
   CORRECT: "Accidental Death, Income Benefit"

10. Numeric fields: Do they contain ONLY integers (no PKR, no commas, no .0)?
    Fields: MIN_AGE, MAX_AGE, MIN_BALANCE, AVG_BALANCE_REQUIREMENT, MIN_INCOME,
    MIN_INCOME_USD, MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS,
    MAX_TERM_YEARS, FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED
    WRONG: "18.0", "PKR 250,000"  CORRECT: "18", "250000"

11. FREE_LOOK_PERIOD_DAYS: Is it set ONLY because this product explicitly mentions it?
    If not explicitly stated → "N/A". Do NOT default to 14.

12. SEGMENT_TIER / SERVICE_TYPE / CUSTOMER_SEGMENT / TARGET_SEGMENT:
    "N/A" unless explicitly stated in the document. Do NOT derive from other fields.

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

Refer to the system prompt above for full field definitions and all rules.

If ANY rule is violated, return CORRECTED JSON. Otherwise return JSON unchanged.
Fix ONLY the violations, preserve everything else.
Return ONLY valid JSON, no explanations."""

    return f"{SYSTEM_PROMPT}\n\n{validation_instructions}"


# ============================================================================
# Model inference
# ============================================================================

def free_gpu_memory():
    """
    Release cached/fragmented CUDA memory between generate() calls.
    empty_cache() returns RESERVED-but-UNUSED memory to the CUDA driver.
    It cannot free the model's weights (live tensors). Use GPU_RESERVE_GIB
    to limit how much of the GPU the loader fills with weights.
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

    # Explicitly drop tensor references to free GPU memory before next call.
    del inputs, output_ids, generated_ids
    free_gpu_memory()

    return text


def extract_one_chunk(model, tokenizer, entry, chunk, chunk_idx, chunk_total):
    """
    Run extraction on a SINGLE chunk of the document (initial attempt +
    one compact repair retry if output is not parseable JSON).
    Validation is done once on the final merged record, not per chunk.
    Returns a normalized record dict, or None if both attempts failed.
    """
    prompt = build_prompt(entry, chunk, tokenizer, chunk_idx, chunk_total)
    raw = get_raw_generation(model, tokenizer, prompt)

    try:
        parsed = parse_json_blob(raw)
        return normalize_record(parsed, entry)
    except Exception:
        # Compact repair prompt (no full SYSTEM_PROMPT) to stay within context
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
    Chunked extraction + field-level merge + single validation pass.

    The document is split into TEXT_CHUNK_SIZE-char pieces. Each piece is
    sent to the model separately. Results are merged field-by-field
    (first real value found for a field across chunks wins). A single
    validation pass runs on the final merged record.

    If ENABLE_CHUNKING=False, the whole document is sent in one call
    (risks OOM on long documents + large system prompt).
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
    """Main extraction entry point: chunked extraction + merge + validation."""
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

        # Cap how much GPU memory Accelerate fills with weights, leaving
        # GPU_RESERVE_GIB free for the KV-cache that generate() needs.
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
                    # Clear fragmented allocator state before retrying —
                    # each retry must start from a clean memory state.
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

            # Clean up after every product to prevent slow memory accumulation.
            free_gpu_memory()

    print("Done. Output:", OUT_JSONL)


if __name__ == "__main__":
    main()