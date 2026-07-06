"""
Step 2: Extract the fixed 56-column schema as JSON from product files.

SYSTEM PROMPT IS READ FROM: EXTRACTION_SYSTEM_PROMPT.txt

Input modes (prompted at startup):
  1) Single file   – one file path (any supported type)
  2) Folder        – all supported files in one directory (non-recursive)
  3) Nested folder – all supported files under a directory tree (recursive)

Supported file types: .txt, .pdf, .docx, .doc, .csv, .json, .xlsx, .xls

Required packages for non-txt formats:
  pip install pdfplumber python-docx openpyxl
  # for .doc: sudo apt install antiword  (Linux)

Configure via .env:
  HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
  HF_MODEL_CLASS=causal
  HF_LOCAL_FILES_ONLY=true

Resumable: already-extracted products (present in OUT_JSONL) are skipped.

KEY FIXES IN THIS VERSION:
- FINANCING_TYPE: Takaful/WTO products always → "Takaful" (not "Unit Linked")
- DEPOSIT_PROFIT_TYPE/FREQUENCY: "Variable"/"At Maturity" for Takaful savings plans
- OPTIONAL_RIDERS: now semicolon-separated (was comma-separated)
- Added _normalize_channel(): "Bank Alfalah branches" → "Bank Branch"
- Added _normalize_target_goal(): maps phrases to single controlled-vocab keyword
- Added _normalize_tenure(): cleans "Minimum Term: X; Maximum Term: Y" format
- normalize_record(): cross-field corrections for FINANCING_TYPE, IS_BANK_OFFERED,
  GENDER, SEGMENT_TIER, MAX_TERM_YEARS (computed from attained age - MIN_AGE)
- Validation and repair prompts updated with all corrected rules
- MAX_NEW_TOKENS raised to 2400 for 56-field JSON output
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

# Reduce CUDA memory fragmentation on small GPUs (Colab T4/L4).
# Must be set before the CUDA context is created, so before `import torch`.
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

MODEL_NAME = os.environ.get("HF_MODEL_NAME_OR_PATH", "").strip()
MODEL_CLASS = os.environ.get("HF_MODEL_CLASS", "causal").strip().lower()
LOCAL_FILES_ONLY = env_bool("HF_LOCAL_FILES_ONLY", True)
TRUST_REMOTE_CODE = env_bool("HF_TRUST_REMOTE_CODE", False)
# 2400 tokens for 56-field JSON with long PRICING_RATE / EMPLOYMENT_TYPE values.
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 2400)
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
    """Load the system prompt from a separate file."""
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

    else:  # Nested folder (recursive)
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
# Document chunking
# ============================================================================

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long document into smaller pieces so each model call only
    processes chunk_size characters. Breaks on paragraph/sentence
    boundaries where possible. A small overlap is carried into the next
    chunk so context at boundaries is not lost.
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
    Merge a chunk's extracted record into the running accumulated record.
    First real value found for a field across chunks wins.
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


def _close_unterminated_json(text: str):
    """
    Best-effort recovery for JSON cut off mid-generation (model hit
    MAX_NEW_TOKENS before finishing). Walks the text tracking bracket/
    string nesting, rewinds to the last structurally complete position,
    then appends the required closing brackets.
    Returns the repaired JSON text, or None if input doesn't start with '{'.
    """
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    def bracket_stack(s: str):
        stack = []
        in_str = False
        esc = False
        for ch in s:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
        return stack, in_str

    stack, in_string = bracket_stack(text)

    if in_string:
        in_str = False
        esc = False
        last_safe_end = 0
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                    last_safe_end = i + 1
            else:
                if ch == '"':
                    in_str = True
                elif ch in "}]":
                    last_safe_end = i + 1
                elif ch == ",":
                    last_safe_end = i
        text = text[:last_safe_end]
        stack, _ = bracket_stack(text)

    text = text.rstrip().rstrip(",").rstrip()
    closers = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        text += closers[opener]

    return text


def parse_json_blob(raw):
    for candidate in _iter_json_candidates(raw):
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = _parse_candidate(cleaned)
        if parsed is not None:
            return parsed

    # Last resort: deterministic close for MAX_NEW_TOKENS truncations.
    closed = _close_unterminated_json(raw)
    if closed:
        parsed = _parse_candidate(closed)
        if parsed is not None:
            return parsed

    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


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


# Field max lengths aligned to corrected-dataset ground-truth observations.
field_max_lengths = {
    "PRODUCT_NAME": 50,
    "PRODUCT_DESCRIPTION": 250,
    "PROVIDER_NAME": 100,
    "PRODUCT_VARIANT_TIER": 50,
    "PRICING_RATE": 400,
    "FEES_AND_CHARGES": 300,
    "KEY_BENEFITS": 250,
    "OPTIONAL_RIDERS": 300,
    "REQUIRED_DOCUMENTS": 200,
    "CLAIMS_SERVICE_CONTACT": 200,
    "KEY_EXCLUSIONS": 200,
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 15,
    "TARGET_GOAL": 50,
    "CUSTOMER_TYPE": 25,
    "CUSTOMER_SEGMENT": 50,
    "TARGET_SEGMENT": 50,
    "SEGMENT_TIER": 30,
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
    "COVERAGE_AMOUNT": 150,
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


# ============================================================================
# Per-field normalization functions
# ============================================================================

def _normalize_plan_type(value: str) -> str:
    """Normalize PLAN_TYPE to exactly one of the allowed values."""
    valid_types = {
        "Loan", "Deposit", "Savings", "Card", "Investment",
        "Insurance", "Service", "Loyalty", "Protection", "Health",
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
    """
    Normalize FINANCING_TYPE to a canonical form.

    CRITICAL FIX: "Takaful" is the highest priority — if the value
    explicitly mentions takaful, it wins over Unit Linked or Hybrid.
    The cross-field post-processing in normalize_record() further
    enforces Takaful for all WTO/takaful-branded products regardless
    of what the model outputs.
    """
    stripped = value.strip()
    lower = stripped.lower()

    # Takaful first — highest priority (corrected-dataset rule)
    if "takaful" in lower:
        return "Takaful"

    # Hybrid check (most specific non-Takaful form)
    if "hybrid" in lower or ("bonus" in lower and "unit" in lower):
        return "Hybrid (Bonus Based and Unit Linked)"

    # Unit Linked
    if "unit linked" in lower or "unit-linked" in lower:
        return "Unit Linked"

    # Single-word canonicals
    canonical_map = {
        "conventional": "Conventional",
        "islamic": "Islamic",
        "mudarabah": "Mudarabah",
    }
    for key, canonical in canonical_map.items():
        if key in lower:
            return canonical

    # Pass through if already an exact known form
    known_exact = {
        "Conventional", "Islamic", "Takaful", "Mudarabah",
        "Unit Linked", "Hybrid (Bonus Based and Unit Linked)", DEFAULT_VALUE,
    }
    if stripped in known_exact:
        return stripped

    return stripped


def _normalize_channel(value: str) -> str:
    """
    Normalize CHANNEL to canonical values.
    'Bank Alfalah branches / Bank Alfalah Limited branches / bank outlets'
    → 'Bank Branch'
    """
    if value in (DEFAULT_VALUE, ""):
        return DEFAULT_VALUE
    lower = value.strip().lower()
    # Any mention of bank + branch/outlet/limited → "Bank Branch"
    if ("bank" in lower and
            any(kw in lower for kw in ("branch", "branche", "outlet", "limited", "offices"))):
        return "Bank Branch"
    if "mobile app" in lower or "mobile application" in lower:
        return "Mobile App"
    if lower in ("online", "web", "internet"):
        return "Online"
    return value.strip()


# Mapping of phrase fragments → canonical TARGET_GOAL keyword.
_TARGET_GOAL_PHRASE_MAP = [
    # Health / hospitalization
    ("hospitali", "Health"),
    ("health cover", "Health"),
    ("medical cover", "Health"),
    ("shifa", "Health"),
    # Education
    ("education", "Education"),
    ("danish", "Education"),
    # Marriage
    ("marriage", "Marriage"),
    ("uroos", "Marriage"),
    ("wedding", "Marriage"),
    # Protection / accident
    ("accident", "Protection"),
    ("accidental death", "Protection"),
    ("theft", "Protection"),
    ("disability", "Protection"),
    ("zaamin", "Protection"),
    ("protect", "Protection"),
    # Retirement / pension
    ("retirement", "Retirement"),
    ("pension", "Retirement"),
    # Investment
    ("investment", "Investment"),
    # Housing
    ("housing", "Housing"),
    ("home financ", "Housing"),
    ("property", "Housing"),
    # Business
    ("business", "Business"),
    # Income
    ("income", "Income"),
    # Loyalty
    ("loyalty", "Loyalty"),
    # Savings (broad catch-all — keep last)
    ("saving", "Savings"),
    ("endowment", "Savings"),
    ("multipurpose", "Savings"),
    ("tadbeer", "Savings"),
    ("zeenat", "Savings"),
    ("saholat", "Savings"),
    ("zindagi", "Savings"),
    ("kamil", "Savings"),
    ("tayyab", "Savings"),
    ("banca", "Savings"),
]

_TARGET_GOAL_CANONICAL = {
    "Protection", "Health", "Education", "Marriage", "Savings",
    "Retirement", "Investment", "Housing", "Business", "Loyalty", "Income",
}


def _normalize_target_goal(value: str) -> str:
    """Map TARGET_GOAL to the fixed single-keyword controlled vocabulary."""
    stripped = value.strip()
    if stripped in _TARGET_GOAL_CANONICAL:
        return stripped
    # Case-insensitive exact match
    for kw in _TARGET_GOAL_CANONICAL:
        if stripped.lower() == kw.lower():
            return kw
    lower = stripped.lower()
    for phrase, canonical in _TARGET_GOAL_PHRASE_MAP:
        if phrase in lower:
            return canonical
    # Unknown — keep as-is; validation pass may fix it
    return stripped


def _normalize_tenure(value: str) -> str:
    """
    Clean up TENURE format:
    - 'Minimum Term: X years; Maximum Term: Y years' → 'X-Y years'
    - '1 Year (renewable)' / '1 Year (Renewable)' → '1 year, yearly renewable'
    """
    if value in (DEFAULT_VALUE, ""):
        return DEFAULT_VALUE
    stripped = value.strip()

    # 'Minimum Term: X years; Maximum Term: Y years' pattern
    m = re.match(
        r"minimum\s+term:?\s*(\d+)\s*years?[;,\s]+maximum\s+term:?\s*(\d+)\s*years?",
        stripped, re.IGNORECASE
    )
    if m:
        return f"{m.group(1)}-{m.group(2)} years"

    # '1 Year (renewable)' → '1 year, yearly renewable'
    if re.match(r"1\s*[Yy]ear\s*\(?[Rr]enewable\)?", stripped):
        return "1 year, yearly renewable"

    return stripped


# ============================================================================
# Main normalization function
# ============================================================================

def normalize_record(record, entry):
    """
    Normalize an extracted record applying all field-level and cross-field
    correction rules.

    Per-field normalizations:
    - PRODUCT_NAME: ALL-CAPS → Title Case
    - PLAN_TYPE: validated against allowed set
    - FINANCING_TYPE: canonical mapping (Takaful priority)
    - GENDER: strict allowed-value enforcement
    - CUSTOMER_TYPE: single-value enforcement
    - CHANNEL: canonical "Bank Branch" mapping
    - TARGET_GOAL: controlled single-keyword vocabulary
    - TENURE: compact format cleanup
    - OPTIONAL_RIDERS: comma-separated → semicolon-separated
    - Numeric fields: strip units/commas/currency symbols

    Cross-field post-processing:
    - FINANCING_TYPE: forced to "Takaful" when WTO/takaful indicator present
    - IS_BANK_OFFERED: set to 1 when CHANNEL = "Bank Branch"
    - SEGMENT_TIER: inferred from product name / gender / plan type when N/A
    - MAX_TERM_YEARS: computed as attained_age - MIN_AGE when TENURE references attained age
    """
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}

    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)

        if value in (None, "", []):
            value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # Numeric columns: integers only
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # PRODUCT_NAME: ALL-CAPS → Title Case
        # ----------------------------------------------------------------
        if col == "PRODUCT_NAME" and isinstance(value, str):
            stripped_name = value.strip()
            if (stripped_name
                    and stripped_name == stripped_name.upper()
                    and len(stripped_name.split()) > 1
                    and len(stripped_name) > 5):
                value = stripped_name.title()

        # ----------------------------------------------------------------
        # PLAN_TYPE: validated set
        # ----------------------------------------------------------------
        if col == "PLAN_TYPE" and isinstance(value, str):
            value = _normalize_plan_type(value)

        # ----------------------------------------------------------------
        # FINANCING_TYPE: canonical mapping (Takaful priority built-in)
        # ----------------------------------------------------------------
        if col == "FINANCING_TYPE" and isinstance(value, str):
            if value not in (DEFAULT_VALUE, ""):
                value = _normalize_financing_type(value)

        # ----------------------------------------------------------------
        # CHANNEL: canonical "Bank Branch" etc.
        # ----------------------------------------------------------------
        if col == "CHANNEL" and isinstance(value, str):
            value = _normalize_channel(value)

        # ----------------------------------------------------------------
        # TARGET_GOAL: controlled single-keyword vocabulary
        # ----------------------------------------------------------------
        if col == "TARGET_GOAL" and isinstance(value, str):
            if value not in (DEFAULT_VALUE, ""):
                value = _normalize_target_goal(value)

        # ----------------------------------------------------------------
        # TENURE: compact format cleanup
        # ----------------------------------------------------------------
        if col == "TENURE" and isinstance(value, str):
            value = _normalize_tenure(value)

        # ----------------------------------------------------------------
        # GENDER: strict allowed-value enforcement
        # ----------------------------------------------------------------
        if col == "GENDER" and isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ("male", "m"):
                value = "Male"
            elif value_lower in ("female", "f", "ladies", "women"):
                value = "Female"
            elif value_lower in ("all", "both", "all genders", "all customers",
                                  "all bank customers", "male and female"):
                value = "All"
            elif value_lower in ("n/a", "na", ""):
                value = DEFAULT_VALUE
            else:
                # Unrecognized — reset; cross-field step may infer "All"
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
                first = stripped_ct.split(",")[0].strip()
                value = first if first in allowed else DEFAULT_VALUE
            elif stripped_ct.lower() == "n/a":
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # OPTIONAL_RIDERS: ensure semicolon-separated (corrected-dataset standard)
        # Previously this converted semicolons → commas (wrong). Now we do
        # the reverse: convert commas → semicolons for flat list values.
        # ----------------------------------------------------------------
        if col == "OPTIONAL_RIDERS" and isinstance(value, str):
            if value not in (DEFAULT_VALUE, ""):
                # Only convert when the value looks like a flat list
                # (no periods = not a descriptive sentence)
                if "." not in value:
                    value = re.sub(r"\s*,\s*", "; ", value).strip().strip(";").strip()

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
    if isinstance(raw_name, str):
        if (raw_name.strip()
                and raw_name.strip() == raw_name.strip().upper()
                and len(raw_name.strip().split()) > 1):
            raw_name = raw_name.strip().title()
    normalized["PRODUCT_NAME"] = raw_name
    normalized["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)

    # ====================================================================
    # Cross-field post-processing corrections
    # ====================================================================

    financing = normalized.get("FINANCING_TYPE", DEFAULT_VALUE)
    provider_lower = normalized.get("PROVIDER_NAME", "").lower()
    product_name_lower = normalized.get("PRODUCT_NAME", "").lower()
    lead_marker = normalized.get("LEAD_MARKER", "")

    # ------------------------------------------------------------------
    # 1. FINANCING_TYPE: force "Takaful" for WTO / takaful-branded products.
    #    The prompt rule states: any product with WTO / Takaful operator
    #    underwriting must have FINANCING_TYPE = "Takaful", even when PIA/
    #    unit-linked language is present (those describe the investment
    #    mechanism, not the financing structure).
    # ------------------------------------------------------------------
    wto_indicators = ("wto", "window takaful", "takaful operator")
    takaful_in_provider = any(ind in provider_lower for ind in wto_indicators)
    takaful_in_name = "takaful" in product_name_lower
    is_ibg = (lead_marker == "IBG")

    if financing in ("Unit Linked", "Hybrid (Bonus Based and Unit Linked)", DEFAULT_VALUE):
        if takaful_in_provider or (is_ibg and (takaful_in_name or takaful_in_provider)):
            normalized["FINANCING_TYPE"] = "Takaful"
    elif financing == DEFAULT_VALUE and is_ibg and (takaful_in_name or takaful_in_provider):
        normalized["FINANCING_TYPE"] = "Takaful"

    # ------------------------------------------------------------------
    # 2. IS_BANK_OFFERED: set to 1 when product is distributed via bank.
    # ------------------------------------------------------------------
    if normalized.get("IS_BANK_OFFERED", DEFAULT_VALUE) == DEFAULT_VALUE:
        channel = normalized.get("CHANNEL", DEFAULT_VALUE)
        if channel in ("Bank Branch", "Mobile App", "Online"):
            normalized["IS_BANK_OFFERED"] = "1"
        elif any(ind in provider_lower for ind in ("distributed via", "administered via", "in partnership with bank")):
            normalized["IS_BANK_OFFERED"] = "1"

    # ------------------------------------------------------------------
    # 3. GENDER: infer "All" for products with no gender restriction when
    #    the product is for general bank customers.
    # ------------------------------------------------------------------
    if normalized.get("GENDER", DEFAULT_VALUE) == DEFAULT_VALUE:
        customer_seg = normalized.get("CUSTOMER_SEGMENT", "").lower()
        eligibility = normalized.get("ELIGIBILITY_TYPE", "").lower()
        product_lower = normalized.get("PRODUCT_NAME", "").lower()
        # Female-specific products keep N/A (model should have set "Female")
        is_female_product = any(
            kw in product_lower for kw in ("zeenat", "ladies", "female", "women")
        )
        is_female_seg = "female" in customer_seg
        if not is_female_product and not is_female_seg:
            if ("all bank" in customer_seg or
                    "all customers" in customer_seg or
                    "bank alfalah customers" in eligibility or
                    "bank alfalah" in eligibility):
                normalized["GENDER"] = "All"

    # ------------------------------------------------------------------
    # 4. SEGMENT_TIER: infer when still N/A after extraction.
    # ------------------------------------------------------------------
    if normalized.get("SEGMENT_TIER", DEFAULT_VALUE) == DEFAULT_VALUE and is_ibg:
        p_lower = normalized.get("PRODUCT_NAME", "").lower()
        gender_val = normalized.get("GENDER", DEFAULT_VALUE)
        target_goal = normalized.get("TARGET_GOAL", DEFAULT_VALUE)

        if any(kw in p_lower for kw in ("premier", "premium", "elite", "vip")):
            normalized["SEGMENT_TIER"] = "Premium"
        elif gender_val == "Female" or "zeenat" in p_lower:
            normalized["SEGMENT_TIER"] = "Niche"
        elif target_goal == "Health" or "shifa" in p_lower:
            normalized["SEGMENT_TIER"] = "Mass Market"
        else:
            normalized["SEGMENT_TIER"] = "Retail"

    # ------------------------------------------------------------------
    # 5. MAX_TERM_YEARS: compute from attained age when TENURE references
    #    "attained age X" and MIN_AGE is known.
    #    Corrected rule: MAX_TERM_YEARS = attained_age - MIN_AGE
    #    (e.g. coverage to attained age 85, min entry age 18 → 67)
    # ------------------------------------------------------------------
    max_term = normalized.get("MAX_TERM_YEARS", DEFAULT_VALUE)
    min_age_val = normalized.get("MIN_AGE", DEFAULT_VALUE)
    tenure_val = normalized.get("TENURE", "")

    if min_age_val != DEFAULT_VALUE and tenure_val:
        attained_match = re.search(
            r"attained\s+age\s+(?:of\s+)?(\d+)", tenure_val, re.IGNORECASE
        )
        if attained_match:
            try:
                attained = int(attained_match.group(1))
                min_age = int(min_age_val)
                computed = attained - min_age
                if computed > 0:
                    # Only override if current value looks like it IS the
                    # attained age (i.e. same as the attained number) or is N/A
                    if max_term == DEFAULT_VALUE or max_term == str(attained):
                        normalized["MAX_TERM_YEARS"] = str(computed)
            except (ValueError, TypeError):
                pass

    return normalized


# ============================================================================
# Prompt builders
# ============================================================================

def build_prompt(entry, chunk, tokenizer, chunk_idx=1, chunk_total=1):
    """Build the extraction prompt for ONE chunk of the document."""
    if chunk_total > 1:
        chunk_note = (
            f"\nNOTE: This is PART {chunk_idx} of {chunk_total} of a single longer "
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
    Does NOT include the full SYSTEM_PROMPT to stay within context limits.
    All essential rules are inlined.
    """
    repair_instructions = f"""You are repairing broken JSON from a data extraction task.
Product: {entry['title']}
Source file: {get_source_filename(entry)}

BROKEN OUTPUT TO REPAIR:
{raw_text[:3000]}

REPAIR RULES — apply ALL:
- Return exactly ONE valid JSON object with 56 fields, nothing else
- Use "N/A" for every missing or unparseable field (never null/None/NaN/"")
- PRODUCT_NAME: Title Case (never ALL CAPS)
- LEAD_MARKER: exactly "IBG" or "BNK"
- PLAN_TYPE: one word from Insurance|Protection|Health|Savings|Deposit|Loan|Card|Investment|Service|Loyalty
  Any insurer/takaful-underwritten product → "Insurance"
- TARGET_GOAL: single keyword from Protection|Health|Education|Marriage|Savings|Retirement|Investment|Housing|Business|Loyalty|Income
- CUSTOMER_TYPE: exactly one of Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A
- GENDER: exactly one of Male|Female|All|N/A
  Products for all general bank customers → "All"
- FINANCING_TYPE: one of Conventional|Islamic|Takaful|Mudarabah|Unit Linked|Hybrid (Bonus Based and Unit Linked)|N/A
  Any WTO/Window Takaful Operations product → ALWAYS "Takaful" (even if PIA/unit-linked language present)
- DEPOSIT_PROFIT_TYPE: Variable|Tier-based|Fixed|Bonus-based|Bonus-based & Unit-linked|N/A
  Takaful savings plan with PIA/fund returns → "Variable"
  Hybrid bonus+unit-linked plan → "Bonus-based & Unit-linked"
  Pure health/protection plan → "N/A"
- DEPOSIT_PROFIT_FREQUENCY: savings/endowment plans → "At Maturity"; pure protection → "N/A"
- CHANNEL: "Bank Alfalah branches" / "bank outlets" → "Bank Branch"
- IS_BANK_OFFERED: 1 if distributed via bank; "N/A" if fully independent
- TENURE: compact format only: "X-Y years" or "1 year, yearly renewable" or "X years to attained age Y"
  Never "Minimum Term: X; Maximum Term: Y"
- TENURE_OPTIONS: savings plans → "Annual, Semi-Annual, Quarterly contributions"; annual plans → "1 year"
- MAX_TERM_YEARS: plan term years (NOT attained age); if coverage to attained age X: MAX_TERM_YEARS = X - MIN_AGE
- Numeric fields (MIN_AGE, MAX_AGE, MIN_BALANCE, MIN_INCOME, MIN_INCOME_USD,
  MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS, MAX_TERM_YEARS,
  FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED): integers only, no units, no .0
- OPTIONAL_RIDERS: semicolon-separated (NOT commas)
  Example: "Accidental Death Benefit; Income Benefit-Disability; Income Benefit-Death"
- SPECIAL_CONDITIONS: include free look period, unit allocation schedule if stated
- PRICING_RATE: contribution/premium amounts only; NO unit allocation % tables
- PROVIDER_NAME: WTO products → "[Insurer] (WTO), distributed via [Bank]"
- SOURCE_FILE_PRODUCT: filename only, no path
- No markdown fences, no explanations outside the JSON

Return the repaired JSON object now."""

    return repair_instructions


def build_validation_prompt(entry, extracted_json):
    """
    Validation and correction prompt.
    Checks the 56-field output against all corrected extraction rules.
    """
    validation_instructions = f"""You extracted this JSON. Validate and fix any issues:

EXTRACTED JSON:
{json.dumps(extracted_json, indent=2)}

VALIDATION RULES — check each and FIX if violated:

1. PRODUCT_NAME: Title Case? Not ALL CAPS?
   WRONG: "JUBILEE KAMIL TAKAFUL SAVINGS PLAN"
   CORRECT: "Jubilee Kamil Takaful Savings Plan"

2. PLAN_TYPE: ONE word from Insurance|Protection|Health|Savings|Deposit|Loan|Card|Investment|Service|Loyalty?
   Any insurer/takaful-underwritten product (LEAD_MARKER=IBG) → ALWAYS "Insurance".

3. TARGET_GOAL: Single keyword from Protection|Health|Education|Marriage|Savings|Retirement|Investment|Housing|Business|Loyalty|Income?
   WRONG: "Savings and Protection", "Complete Protection Against Accidental Death", "Hospitalization Protection"
   CORRECT: "Savings", "Protection", "Health"

4. CUSTOMER_TYPE: exactly ONE value — Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A?
   No bank names, no commas, no free text.

5. GENDER: exactly one of Male|Female|All|N/A?
   Products available to ALL general bank customers → "All".
   Female-specific products → "Female".
   "N/A" ONLY if genuinely undeterminable.

6. CUSTOMER_SEGMENT: Is it set for products targeting general bank customers?
   General bank customer products → "All Bank Alfalah Customers" (or relevant bank name).
   Female-specific → "Female Customers and Spouse".

7. TARGET_SEGMENT: Is it derived from product purpose?
   Education → "Education Planning"; Marriage → "Marriage Planning"; Health → "Healthcare Coverage";
   Female savings → "Female Savings Planning"; Multipurpose savings → "Multipurpose Planning";
   Premium/wellness savings → "Wellness Focused"; Standard savings → "Savings Planning";
   Protection/accident → "Protection Planning".

8. SEGMENT_TIER: Is it set?
   "Premier"/"Premium" in product name → "Premium"; Female-specific → "Niche";
   Health/hospitalization plan → "Mass Market"; Standard retail → "Retail".

9. CHANNEL: "Bank Alfalah branches / Limited branches / outlets" → "Bank Branch"?

10. SERVICE_TYPE: Is it set based on product type?
    Education → "Education Planning"; Health → "Healthcare"; Marriage → "Marriage Planning";
    Savings/endowment → "Savings Planning"; Protection → "Protection Planning";
    Premium wellness savings → "Wellness Savings".

11. FINANCING_TYPE: For WTO / Takaful operator products → "Takaful"?
    WRONG: "Unit Linked" or "Hybrid (Bonus Based and Unit Linked)" for a WTO/takaful product
    CORRECT: "Takaful" for ALL products underwritten by WTO or any takaful operator,
    EVEN IF PIA / unit-linked / fund-allocation language appears in the document.

12. DEPOSIT_PROFIT_TYPE: Is it set for savings/investment plans?
    Takaful savings plan with PIA/fund returns → "Variable"
    Hybrid bonus-based + unit-linked plan → "Bonus-based & Unit-linked"
    Pure protection/health/annual-term plan → "N/A"
    WRONG: "N/A" for a unit-linked/PIA savings plan

13. DEPOSIT_PROFIT_FREQUENCY: Is it set for savings/investment plans?
    Savings/endowment plans → "At Maturity"
    Pure protection/health/annual-term plans → "N/A"
    WRONG: "N/A" for a savings plan with returns at maturity

14. TENURE: Is it in compact format?
    WRONG: "Minimum Term: 10 years; Maximum Term: 25 years"
    CORRECT: "10-25 years"
    WRONG: "1 Year (renewable)"
    CORRECT: "1 year, yearly renewable"

15. TENURE_OPTIONS: Is it contribution payment options for savings plans?
    Savings plans → "Annual, Semi-Annual, Quarterly contributions"
    Annual renewable → "1 year"
    NOT payment frequencies as plan durations.

16. MAX_TERM_YEARS: Is it the plan TERM (not attained age)?
    If TENURE says "to attained age X": MAX_TERM_YEARS = X - MIN_AGE
    WRONG: 85 when coverage ends at attained age 85 with MIN_AGE=18
    CORRECT: 67 (= 85 - 18)

17. IS_BANK_OFFERED: Set to 1 when CHANNEL = "Bank Branch" or distributed via bank?
    Products sold through bank branches → 1

18. OPTIONAL_RIDERS: Are they semicolon-separated (NOT commas)?
    WRONG: "Accidental Death Benefit, Income Benefit-Disability"
    CORRECT: "Accidental Death Benefit; Income Benefit-Disability"

19. PRICING_RATE: Does it contain ONLY contribution/premium amounts?
    Unit allocation % schedules belong in SPECIAL_CONDITIONS, not PRICING_RATE.
    WRONG: "Year 1: 60%; Year 2: 80%; Year 3: 95%"
    CORRECT: "Min contribution PKR 25,000/yr; Annual, Semi-Annual, Quarterly"

20. PROVIDER_NAME: For WTO products → "[Insurer] (WTO), distributed via [Bank]"?
    WRONG: "IGI Life Insurance WTO"
    CORRECT: "IGI Life Insurance (WTO), distributed via Bank Alfalah Ltd"

21. SPECIAL_CONDITIONS: Does it include free look period and unit allocation schedule if stated?
    Include: "Free 14-day look period; unit allocation: Yr1 60%, Yr2 80%, Yr3 95%, Yr4+ 100%"

22. Numeric fields: integers only (no PKR, no commas, no .0)?
    Fields: MIN_AGE, MAX_AGE, MIN_BALANCE, AVG_BALANCE_REQUIREMENT, MIN_INCOME,
    MIN_INCOME_USD, MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS,
    MAX_TERM_YEARS, FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED
    WRONG: "18.0", "PKR 250,000"  CORRECT: "18", "250000"

23. FREE_LOOK_PERIOD_DAYS: Set ONLY if THIS product explicitly states it. Do NOT default to 14.

24. All 56 fields present? No null/None/NaN/empty string → "N/A"
    Fields: PRODUCT_NAME, LEAD_MARKER, SOURCE_FILE_PRODUCT, PLAN_TYPE, TARGET_GOAL,
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

If ANY rule is violated, return CORRECTED JSON. Otherwise return JSON unchanged.
Fix ONLY the violations, preserve everything else.
Return ONLY valid JSON, no explanations."""

    return f"{SYSTEM_PROMPT}\n\n{validation_instructions}"


# ============================================================================
# Model inference
# ============================================================================

def free_gpu_memory():
    """Release cached/fragmented CUDA memory between generate() calls."""
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

    hit_token_limit = generated_ids.shape[-1] >= MAX_NEW_TOKENS
    if hit_token_limit:
        print(
            f"    note: generation hit MAX_NEW_TOKENS={MAX_NEW_TOKENS} "
            f"(output likely truncated) — attempting auto-close recovery"
        )

    del inputs, output_ids, generated_ids
    free_gpu_memory()

    return text


def extract_one_chunk(model, tokenizer, entry, chunk, chunk_idx, chunk_total):
    """
    Run extraction on a SINGLE chunk (initial attempt + compact repair retry).
    Validation runs on the final merged record, not per chunk.
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
    (risks OOM on long documents).
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
        # If validation call fails to parse, the merged record (already
        # normalized field-by-field) is still a valid result.
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
            "Check the repo id or set HF_MODEL_NAME_OR_PATH to a local folder path.\n"
            "If the repo is private or gated, authenticate in Colab first."
        ) from exc

    model_kwargs: dict = {
        "local_files_only": LOCAL_FILES_ONLY,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if DEVICE_MAP.lower() != "none":
        model_kwargs["device_map"] = DEVICE_MAP

        # Reserve GPU_RESERVE_GIB for the KV-cache that generate() needs.
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