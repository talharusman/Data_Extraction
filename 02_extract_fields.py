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

CHANGES IN THIS VERSION (audit fixes — generic, not product-specific):
- Ground-truth analysis showed the corrected dataset consistently uses " | "
  as the separator for every multi-value/list field (never semicolons, and
  never commas as a *list* separator since commas already appear inside
  amounts like "500,000"). Added a generic delimiter-normalization step
  that:
    * always converts ";" -> " | " for every text field (safe: semicolons
      never appear inside amounts/prose in this domain), and
    * converts ", " -> " | " ONLY for fields that are strictly list-type by
      design and don't carry monetary/prose commas (CUSTOMER_TYPE,
      PRODUCT_VARIANT_TIER, SEGMENT_TIER, TENURE_OPTIONS,
      PREMIUM_PAYMENT_FREQUENCY, OPTIONAL_RIDERS, EMPLOYMENT_TYPE).
  This is a generic, document-driven normalization rule, not a per-product
  mapping, so it generalizes across all 200+ documents.
- NUMERIC_COLUMNS coercion previously collapsed tiered values (e.g. a
  Bronze/Silver/Gold contribution table in MIN_CONTRIBUTION) down to a
  single number, destroying real information. Added a tiered-value
  detector (_is_tiered_value) that, for the subset of numeric fields that
  can legitimately be tiered (balances/income/investment/contribution
  fields), preserves the full tiered text instead of forcing a single
  integer. True single-value numeric fields (ages, term years, free-look
  days, IS_BANK_OFFERED) are unaffected and still coerced to plain integers.
- CUSTOMER_TYPE normalization previously forced a single enum value and
  discarded anything after the first comma. Ground truth allows multiple
  pipe-separated enum values (e.g. "Salaried|Self-Employed"). Rewrote the
  normalizer to validate each "|"-segment against the enum and keep all
  valid ones, joined by " | ".
- OPTIONAL_RIDERS/other list-style normalizers updated to emit " | " instead
  of "," to match the ground-truth delimiter standard.
- field_max_lengths increased for COVERAGE_AMOUNT (150->300), KEY_BENEFITS
  (250->280), REQUIRED_DOCUMENTS (200->220), SPECIAL_CONDITIONS (200->250)
  based on corrected-dataset ground-truth lengths, so full tiered/category
  lists aren't truncated mid-list.
- _normalize_target_goal extended with solar/green energy/financing
  categories to correctly handle bank loan products (e.g. green energy term
  finance facilities). The previous mapping only covered insurance/savings
  plan goals and produced "Protection" for unrelated loan products.
- _normalize_equity_requirement added to strip verbose prefixes (e.g.
  "Minimum 20% (Salaried & Business)" → "20%") and normalize to the
  canonical short percentage format expected in EQUITY_REQUIREMENT.
- _crossfield_validate added to enforce LEAD_MARKER-based consistency:
  BNK (bank-direct) products cannot have Unit Linked or Hybrid financing
  types (which are IBG-only structures). This catches cascading errors when
  a model misclassifies a loan product as IBG and then sets FINANCING_TYPE
  to a fund-based value.
- build_validation_prompt extended with rules 21-25 for LEAD_MARKER
  consistency: BNK products must have Conventional/Islamic FINANCING_TYPE,
  bank-only PROVIDER_NAME, N/A for COVERAGE_AMOUNT unless explicit amounts
  are stated, and N/A for insurance-specific fields (FREE_LOOK_PERIOD_DAYS,
  OPTIONAL_RIDERS, PREMIUM_PAYMENT_FREQUENCY, MIN_CONTRIBUTION).
- All previous functionality (chunking, merge, repair/validation passes,
  JSON recovery, model loading) preserved unchanged.
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
# Raised from 1500 to 2200: the 1500 budget was still getting hit on
# products with long CUSTOMER_TYPE lists, EMPLOYMENT_TYPE target-market
# text, and multi-tier PRICING_RATE tables, which truncated the JSON
# mid-field (see _close_unterminated_json for the recovery path when this
# still happens). Override via HF_MAX_NEW_TOKENS in .env if needed.
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 2200)
# FIX: Enforce a minimum safe budget for 56-field JSON generation.
# If .env has HF_MAX_NEW_TOKENS set too low (e.g. 1000), the model will
# truncate the JSON mid-field — the auto-close recovery cannot reliably
# repair both the initial and repair-pass outputs when both hit the same
# token ceiling. Override with a warning instead of silently producing
# blank/partial records in the JSONL output.
_MIN_SAFE_TOKENS = 2200
if MAX_NEW_TOKENS < _MIN_SAFE_TOKENS:
    print(
        f"\nWARNING: HF_MAX_NEW_TOKENS={MAX_NEW_TOKENS} is too small to generate "
        f"the complete 56-field JSON schema (needs ~{_MIN_SAFE_TOKENS} tokens).\n"
        f"Auto-raising MAX_NEW_TOKENS to {_MIN_SAFE_TOKENS}. "
        f"Update HF_MAX_NEW_TOKENS in your .env to suppress this warning.\n"
    )
    MAX_NEW_TOKENS = _MIN_SAFE_TOKENS
TEMPERATURE = env_float("HF_TEMPERATURE", 0.0)
TOP_P = env_float("HF_TOP_P", 1.0)
REPETITION_PENALTY = env_float("HF_REPETITION_PENALTY", 1.03)
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", 6000)
TEXT_CHUNK_OVERLAP = env_int("TEXT_CHUNK_OVERLAP", 500)
LOAD_IN_4BIT = env_bool("HF_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = env_bool("HF_LOAD_IN_8BIT", False)
DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
TORCH_DTYPE = os.environ.get("HF_TORCH_DTYPE", "auto").strip().lower()
GPU_RESERVE_GIB = env_float("HF_GPU_RESERVE_GIB", 5.0)
ENABLE_CHUNKING = env_bool("ENABLE_CHUNKING", True)
DEFAULT_VALUE = "N/A"

# Numeric fields that are ALWAYS a single plain integer (never legitimately
# tiered), so they are always force-coerced to a single number.
STRICT_NUMERIC_COLUMNS = {
    "MIN_AGE",
    "MAX_AGE",
    "IS_BANK_OFFERED",
    "MIN_TERM_YEARS",
    "MAX_TERM_YEARS",
    "FREE_LOOK_PERIOD_DAYS",
}

# Numeric fields that CAN legitimately be a tiered table (e.g. a
# Bronze/Silver/Gold contribution or balance schedule). For these, a tiered
# value is preserved as text instead of being collapsed to one number.
TIER_AWARE_NUMERIC_COLUMNS = {
    "MIN_BALANCE",
    "AVG_BALANCE_REQUIREMENT",
    "MIN_INCOME",
    "MIN_INCOME_USD",
    "MIN_INVESTMENT",
    "MIN_CONTRIBUTION",
}

NUMERIC_COLUMNS = STRICT_NUMERIC_COLUMNS | TIER_AWARE_NUMERIC_COLUMNS

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
    record for the product.

    Strategy (improved):
    - Empty → real value: accept the new value
    - Both real: for text fields, prefer the LONGER value (more likely to be
      complete, e.g. a full tier table vs a partial one from a chunk boundary).
      For strict numeric fields, keep the first value found.
    """
    for col in columns:
        old = accumulated.get(col)
        new = new_record.get(col)
        old_is_empty = old is None or old == "" or old == DEFAULT_VALUE
        new_is_real = new not in (None, "", DEFAULT_VALUE)
        if old_is_empty and new_is_real:
            accumulated[col] = new
        elif not old_is_empty and new_is_real and col not in STRICT_NUMERIC_COLUMNS:
            # Both have real values — prefer the longer one for non-strict-
            # numeric fields, as it's more likely to be the complete value
            # (e.g. a full tier table vs a partial one from a chunk boundary).
            if len(str(new)) > len(str(old)) + 20:
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
    Best-effort recovery for JSON that was cut off mid-generation (i.e. the
    model hit MAX_NEW_TOKENS before finishing the object) rather than being
    genuinely malformed. All the existing candidate strategies in
    _iter_json_candidates() require a BALANCED object ({...} fully closed);
    a mid-field truncation never satisfies that, so they all fail together.

    Strategy:
      1. Walk the text tracking bracket/string nesting, remembering the last
         position where we had a *structurally complete* token (end of a
         closed string, a closed {}/[], or just before a trailing comma).
      2. If we end while still inside an open string, we don't know how the
         string was meant to end, so we rewind to that last safe position
         instead of guessing at the missing content.
      3. Recompute the open-bracket stack up to that safe cut point and
         append the matching closers.

    This sacrifices only the one field that was mid-generation when the
    model was cut off (it will fall back to "N/A" via normalize_record);
    every field completed before the cutoff is preserved. Returns the
    repaired JSON text, or None if the input doesn't even start with '{'.
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
        # Rewind to the last point where we had a fully-closed string, a
        # fully-closed nested object/array, or a trailing comma — i.e. the
        # last spot we can safely cut without inventing content.
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

    # Last resort: the output may be a genuinely truncated (not malformed)
    # JSON object — try to close it deterministically before giving up.
    # This is cheap (no model call) and recovers most MAX_NEW_TOKENS cutoffs.
    # FIX: strip ```json fences BEFORE passing to the bracket-tracker so the
    # rewind logic operates on clean JSON text, not on the raw model output
    # that still has the opening fence and possible preamble text.
    closed = _close_unterminated_json(_strip_wrappers(raw))
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
    "KEY_BENEFITS": 280,           # widened: full core-benefit lists observed >250 chars
    "OPTIONAL_RIDERS": 300,        # rider lists up to ~253 chars
    "REQUIRED_DOCUMENTS": 220,     # widened slightly for full doc lists
    "CLAIMS_SERVICE_CONTACT": 200,
    "KEY_EXCLUSIONS": 200,
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 30,                # widened: descriptive category, not single word
    "TARGET_GOAL": 50,             # "Children's Education Planning" style values
    "CUSTOMER_TYPE": 60,            # widened: may hold multiple "|"-joined enum values
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
    "COVERAGE_AMOUNT": 300,        # widened: full multi-category coverage lists observed
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
    "SPECIAL_CONDITIONS": 250,     # widened: multi-condition " | " lists observed >200 chars
    "PREMIUM_PAYMENT_FREQUENCY": 50,
}

# Fields where a comma is *always* a list separator (never a monetary or
# prose comma), so it's safe to normalize ", " -> " | " for these. Fields
# NOT in this set (e.g. COVERAGE_AMOUNT, KEY_BENEFITS, PROVIDER_NAME) can
# legitimately contain commas inside amounts or prose, so they are only
# normalized for ";" -> " | ", never for ",".
COMMA_IS_LIST_SEPARATOR_FIELDS = {
    "CUSTOMER_TYPE",
    "PRODUCT_VARIANT_TIER",
    "SEGMENT_TIER",
    "TENURE_OPTIONS",
    "PREMIUM_PAYMENT_FREQUENCY",
    "OPTIONAL_RIDERS",
    "EMPLOYMENT_TYPE",
}


def _normalize_list_delimiters(col: str, value: str) -> str:
    """
    Generic, document-driven delimiter normalization (not product-specific).

    The corrected ground-truth dataset consistently separates multi-value
    fields with " | " and never uses semicolons as a list separator.
    Semicolons never legitimately appear inside amounts or prose in this
    domain, so ";" -> " | " is always safe. Commas DO legitimately appear
    inside amounts ("500,000") and prose (PROVIDER_NAME partnership text),
    so ", " -> " | " is only applied to fields that are strictly list-type
    by design (COMMA_IS_LIST_SEPARATOR_FIELDS).
    """
    if not isinstance(value, str) or value in (DEFAULT_VALUE, ""):
        return value

    # Semicolons: always a list separator in this domain.
    value = re.sub(r"\s*;\s*", " | ", value)

    if col in COMMA_IS_LIST_SEPARATOR_FIELDS:
        # Only collapse comma-space sequences that look like list breaks,
        # not thousands-separators inside a number (e.g. "10,000").
        value = re.sub(r"(?<!\d),\s+(?!\d{3}\b)", " | ", value)

    # Collapse accidental doubled separators / stray spacing.
    value = re.sub(r"\s*\|\s*\|\s*", " | ", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip().strip("|").strip()


def _is_tiered_value(stripped: str) -> bool:
    """
    Detect a tiered/table-style value (e.g. "Bronze:5,000, Silver:10,000")
    that should be PRESERVED as text rather than collapsed to one integer.
    Handles multiple formats: Label:number, Label=number, Label-number,
    pipe-separated tiers, etc.
    """
    # Pattern 1: "Label: number" or "Label= number" pairs
    pairs = re.findall(r"[A-Za-z][\w\s]{0,24}[:=]\s*[\d,]+", stripped)
    if len(pairs) >= 2:
        return True
    # Pattern 2: pipe-separated segments, each containing a number
    if "|" in stripped:
        segments = [s.strip() for s in stripped.split("|")]
        num_segments = sum(1 for seg in segments if seg and re.search(r"\d", seg))
        if num_segments >= 2:
            return True
    # Pattern 3: "Label - number" pairs (dash separator)
    dash_pairs = re.findall(r"[A-Za-z][\w\s]{0,24}\s*[-\u2013\u2014]\s*[\d,]+", stripped)
    if len(dash_pairs) >= 2:
        return True
    return False


def truncate_to_boundary(value: str, max_len: int) -> str:
    """Truncate text at a clean boundary when possible, preferring pipe
    separators for list fields so partial items aren't left dangling."""
    if len(value) <= max_len:
        return value

    cut = value[:max_len].rstrip()
    # Prefer a pipe separator boundary so list items stay complete
    pipe_idx = cut.rfind(" | ")
    if pipe_idx > max_len * 0.5:
        return cut[:pipe_idx].rstrip(" ,;:-/|")
    boundary = max(cut.rfind(" "), cut.rfind(","), cut.rfind(";"), cut.rfind(":"), cut.rfind("-"), cut.rfind("|"))
    if boundary > 0:
        return cut[:boundary].rstrip(" ,;:-/|")
    return cut


def _normalize_plan_type(value: str) -> str:
    """
    PLAN_TYPE is now a short descriptive category (see prompt) rather than a
    rigid single word. We only lightly clean it here:
      - Preserve it as-is if it already contains "Insurance" (the anchor
        word required for any insurer-underwritten product).
      - Otherwise, map to the closest bank-only enum word if the value
        clearly corresponds to one.
      - Never silently discard an unrecognized but plausible value — that
        would hide real extraction content behind "N/A" and doesn't
        generalize well across 200+ documents with varied phrasing.
    """
    stripped = value.strip()
    if not stripped or stripped.upper() == DEFAULT_VALUE:
        return DEFAULT_VALUE

    if "insurance" in stripped.lower():
        return stripped[:30]

    bank_only = {"Deposit", "Loan", "Card", "Service", "Loyalty", "Investment", "Savings"}
    for word in bank_only:
        if stripped.lower() == word.lower():
            return word
    for word in bank_only:
        if word.lower() in stripped.lower().split():
            return word

    # Keep the model's descriptive value rather than forcing N/A — this
    # preserves genuinely new categories seen in unseen documents.
    return stripped[:30]


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


def _normalize_target_goal(value: str) -> str:
    """Normalize TARGET_GOAL to corrected style.

    Extended with solar/green energy/financing categories so that bank loan
    products (term finance, auto finance, SME financing) are correctly
    classified instead of being mapped to insurance-product goals like
    'Protection'. More specific multi-word keys are checked before shorter
    single-word keys to prevent partial matches (e.g. 'green energy' before
    'energy' alone).
    """
    if not value or value == DEFAULT_VALUE:
        return DEFAULT_VALUE
    val_lower = value.lower()
    mappings = {
        # Financing / loan categories — checked first so they take priority
        # over shorter generic keys like "home" or "income".
        "solar energy": "Solar Energy Financing",
        "green energy": "Green Energy Financing",
        "working capital": "SME Financing",
        "income protection": "Income Protection",
        # Single-word financing keys
        "solar": "Solar Energy Financing",
        "energy": "Green Energy Financing",
        "green": "Green Energy Financing",
        "vehicle": "Vehicle Financing",
        "car": "Vehicle Financing",
        "motor": "Vehicle Financing",
        "housing": "Housing",
        "sme": "SME Financing",
        "business": "Business Financing",
        "personal": "Personal Financing",
        "micro": "Microfinance",
        # Insurance / savings plan categories
        "multipurpose": "Multipurpose Savings",
        "hospitalization": "Health",
        "protection": "Protection",
        "accidental": "Protection",
        "savings": "Savings",
        "education": "Education",
        "health": "Health",
        "marriage": "Marriage",
        "retirement": "Retirement",
        "children": "Education",
        "hajj": "Hajj Savings",
        "umrah": "Hajj Savings",
        "home": "Housing",
        "investment": "Investment",
        "wealth": "Wealth Management",
        "income": "Income Protection",
    }
    for key, norm in mappings.items():
        if key in val_lower:
            return norm
    return value.strip()[:50]


def _normalize_channel(value: str) -> str:
    """Standardize CHANNEL."""
    if not value or value == DEFAULT_VALUE:
        return DEFAULT_VALUE
    val_lower = value.lower()
    if "branch" in val_lower or "branches" in val_lower:
        return "Bank Branch"
    return value.strip()[:30]


def _normalize_eligibility(value: str) -> str:
    """Clean ELIGIBILITY_TYPE — generic across all banks."""
    if not value or value == DEFAULT_VALUE:
        return DEFAULT_VALUE
    # Generic: normalize "Bank X Limited" → "Bank X" for any bank name
    value = re.sub(r"\b(Bank\s+\w+(?:\s+\w+)?)\s+Limited\b", r"\1", value, flags=re.I)
    return truncate_to_boundary(value.strip(), 100)


def _normalize_customer_type(value: str) -> str:
    """
    CUSTOMER_TYPE may legitimately hold MULTIPLE enum values (ground truth
    shows e.g. "Salaried|Self-Employed"), so — unlike the previous version —
    we no longer collapse to a single value. Each "|"-or-","-separated
    segment is validated against the allowed enum; valid segments are kept
    (deduplicated, order preserved) and joined with " | ". Segments that
    don't match the enum are dropped rather than kept as free text, since
    CUSTOMER_TYPE must stay a controlled vocabulary field.
    """
    allowed = {
        "Salaried", "Self-Employed", "SME",
        "Corporate", "Retail", "Government",
    }
    stripped = value.strip()
    if not stripped or stripped.upper() == DEFAULT_VALUE:
        return DEFAULT_VALUE

    segments = re.split(r"[|,;]+", stripped)
    kept = []
    for seg in segments:
        seg_clean = seg.strip()
        if not seg_clean:
            continue
        for vt in allowed:
            if seg_clean.lower() == vt.lower() and vt not in kept:
                kept.append(vt)
                break
    if kept:
        return " | ".join(kept)
    return DEFAULT_VALUE


def _normalize_equity_requirement(value: str) -> str:
    """
    Normalize EQUITY_REQUIREMENT to a short percentage value.

    Strips verbose prefixes such as "Minimum", "Min.", "At least" and
    trailing parenthetical qualifiers so only the core percentage (e.g.
    "20%") is stored. This matches the prompt's EQUITY_REQUIREMENT example
    format and the data dictionary TYPE_HINT ("30%").

    Examples:
      "Minimum 20% (Salaried & Business)" → "20%"
      "Min. 30%"                           → "30%"
      "At least 25%"                       → "25%"
      "20%"                                → "20%"
    """
    if not value or value == DEFAULT_VALUE:
        return DEFAULT_VALUE
    cleaned = value.strip()
    # Remove leading verbose qualifiers
    cleaned = re.sub(
        r"^(minimum|min\.?|at\s+least|equity[:\s]+)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove trailing parenthetical qualifiers e.g. "(Salaried & Business)"
    cleaned = re.sub(r"\s*\(.*\)\s*$", "", cleaned).strip()
    # Keep only the first percentage token if multiple remain
    pct_match = re.search(r"(\d+(?:\.\d+)?%)", cleaned)
    if pct_match:
        return pct_match.group(1)
    # If no % symbol but a plain number, append %
    num_match = re.match(r"^(\d+(?:\.\d+)?)\s*$", cleaned)
    if num_match:
        return num_match.group(1) + "%"
    return truncate_to_boundary(cleaned, 20)


def _crossfield_validate(normalized: dict) -> dict:
    """
    Apply cross-field consistency rules that cannot be enforced on a
    field-by-field basis during per-column normalization.

    Called at the end of normalize_record after all per-field normalizations
    have run. Rules:

    1. BNK (bank-direct loan/deposit/card) products cannot have a
       fund-based FINANCING_TYPE (Unit Linked or Hybrid), because those
       structures are only valid for insurer-underwritten savings/investment
       plans (IBG products). When a BNK product has been incorrectly
       classified and FINANCING_TYPE has been set to a fund-based value, it
       is reset to "Conventional" — the correct value for a bank-direct
       loan with markup-based pricing and no fund/unit allocation language.

    This function is intentionally narrow. It only auto-corrects cases where
    the combination is logically impossible (a bank loan cannot be "Unit
    Linked"). Fields that are merely unlikely for a BNK product (e.g.
    COVERAGE_AMOUNT, FREE_LOOK_PERIOD_DAYS) are left for the validation
    prompt pass to handle, since they can occasionally appear for bundled
    insurance components within a loan product.
    """
    lead = normalized.get("LEAD_MARKER", DEFAULT_VALUE)
    if not isinstance(lead, str):
        return normalized

    if lead.strip().upper() == "BNK":
        # BNK: Unit Linked and Hybrid are IBG-only financing structures.
        # A bank-direct loan product with KIBOR-based or fixed markup is
        # by definition Conventional unless Islamic/Shariah wording appears.
        fin_type = normalized.get("FINANCING_TYPE", DEFAULT_VALUE)
        if isinstance(fin_type, str) and fin_type in (
            "Unit Linked", "Hybrid (Bonus Based and Unit Linked)"
        ):
            normalized["FINANCING_TYPE"] = "Conventional"

    return normalized


def normalize_record(record, entry):
    """
    Normalize an extracted record with corrections for all known model errors.

    Key normalization rules applied here:
    - PRODUCT_NAME: ALL-CAPS converted to Title Case
    - PLAN_TYPE: descriptive-category cleanup (see _normalize_plan_type)
    - FINANCING_TYPE: canonical Unit Linked / Hybrid mapping
    - GENDER: strict allowed-value enforcement
    - CUSTOMER_TYPE: multi-value enum enforcement, "|"-joined
    - TARGET_GOAL, CHANNEL, ELIGIBILITY_TYPE, tenure fields enhanced
    - EQUITY_REQUIREMENT: verbose prefix stripped to short percentage
    - Numeric fields: strip units, commas, currency symbols — UNLESS the
      value is a genuine tiered table for a tier-aware numeric field, in
      which case the tiered text is preserved (see _is_tiered_value)
    - List-type text fields: delimiter normalized to " | "
    - All text fields: truncated at word boundary to max length
    - Cross-field: BNK product FINANCING_TYPE consistency enforced
    """
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}

    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)

        if value in (None, "", []):
            value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # Numeric columns
        # ----------------------------------------------------------------
        if col in NUMERIC_COLUMNS and isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.upper() == DEFAULT_VALUE:
                value = DEFAULT_VALUE
            elif col in TIER_AWARE_NUMERIC_COLUMNS and _is_tiered_value(stripped):
                # Preserve the tiered table as text instead of collapsing
                # it to a single number (root cause of a real data-loss bug
                # observed in the audit: tiered contribution/balance tables
                # were being reduced to just the first number).
                value = _normalize_list_delimiters(col, stripped)
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
        # PLAN_TYPE: descriptive-category cleanup
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
        # TARGET_GOAL normalization
        # ----------------------------------------------------------------
        if col == "TARGET_GOAL" and isinstance(value, str):
            value = _normalize_target_goal(value)

        # ----------------------------------------------------------------
        # CHANNEL normalization
        # ----------------------------------------------------------------
        if col == "CHANNEL" and isinstance(value, str):
            value = _normalize_channel(value)

        # ----------------------------------------------------------------
        # ELIGIBILITY_TYPE normalization
        # ----------------------------------------------------------------
        if col == "ELIGIBILITY_TYPE" and isinstance(value, str):
            value = _normalize_eligibility(value)

        # ----------------------------------------------------------------
        # EQUITY_REQUIREMENT: strip verbose prefix to short percentage
        # ----------------------------------------------------------------
        if col == "EQUITY_REQUIREMENT" and isinstance(value, str):
            value = _normalize_equity_requirement(value)

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
        # CUSTOMER_TYPE: multi-value enum enforcement
        # ----------------------------------------------------------------
        if col == "CUSTOMER_TYPE" and isinstance(value, str):
            value = _normalize_customer_type(value)

        # ----------------------------------------------------------------
        # Generic list-delimiter normalization (see G10 in the prompt):
        # ";" -> " | " always; ", " -> " | " only for strictly list-type
        # fields where a comma can never be a monetary/prose separator.
        # ----------------------------------------------------------------
        if isinstance(value, str) and value not in (DEFAULT_VALUE, ""):
            if col in (
                COMMA_IS_LIST_SEPARATOR_FIELDS
                | {
                    "KEY_BENEFITS", "KEY_EXCLUSIONS", "REQUIRED_DOCUMENTS",
                    "SEGMENT_TIER", "SPECIAL_CONDITIONS", "FEES_AND_CHARGES",
                    "COVERAGE_AMOUNT",
                }
            ):
                value = _normalize_list_delimiters(col, value)
            elif ";" in value:
                # Even fields not in the list above should never keep a
                # semicolon list separator (G10) — safe to convert globally.
                value = _normalize_list_delimiters(col, value)

        # ----------------------------------------------------------------
        # TENURE normalization - keep as-is but truncate
        # ----------------------------------------------------------------
        if col in ("TENURE", "TENURE_OPTIONS") and isinstance(value, str):
            value = truncate_to_boundary(value.strip(), field_max_lengths.get(col, 50))

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

    # ----------------------------------------------------------------
    # Cross-field consistency: must run after all per-field normalizations
    # ----------------------------------------------------------------
    normalized = _crossfield_validate(normalized)

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
    repair_instructions = f"""You are repairing broken JSON from a data extraction task.
Product: {entry['title']}
Source file: {get_source_filename(entry)}

BROKEN OUTPUT TO REPAIR:
{raw_text[:3000]}

REPAIR RULES — apply all of these:
- Return exactly ONE valid JSON object with 56 fields, nothing else
- Use "N/A" for every missing or unparseable field (never null/None/NaN/"")
- PRODUCT_NAME: Title Case (never ALL CAPS)
- LEAD_MARKER: exactly "IBG" or "BNK"
  → "BNK" for term finance, auto finance, home finance, green energy loan,
    SME loan, personal loan, deposit, card — even if bundled insurance exists
  → "IBG" ONLY when a named insurer/takaful company underwrites the product
- PLAN_TYPE: must contain "Insurance" for IBG products; for BNK products
  use one of Deposit|Loan|Card|Service|Loyalty|Investment|Savings
- CUSTOMER_TYPE: one or more of Salaried|Self-Employed|SME|Corporate|Retail|
  Government joined by " | " if multiple, or "N/A"
- GENDER: exactly one of Male|Female|All|N/A
- FINANCING_TYPE: Conventional|Islamic|Takaful|Mudarabah|Unit Linked|
  Hybrid (Bonus Based and Unit Linked)|N/A.
  BNK loan products with KIBOR markup → "Conventional" (never Unit Linked/Hybrid)
- TARGET_GOAL: standardized short term like Protection, Savings, Education,
  Health, Marriage, Solar Energy Financing, Green Energy Financing,
  Vehicle Financing, Housing, SME Financing, Business Financing
- CHANNEL: "Bank Branch" if applicable
- Numeric fields (MIN_AGE, MAX_AGE, MIN_TERM_YEARS, MAX_TERM_YEARS,
  FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED): integers only, no units, no .0.
  MIN_BALANCE/MIN_INCOME/MIN_INCOME_USD/MIN_INVESTMENT/MIN_CONTRIBUTION:
  a single integer, OR the full tiered table text if genuinely tiered.
- EQUITY_REQUIREMENT: short percentage only, e.g. "20%" not "Minimum 20%"
- TENURE_OPTIONS: plan duration choices only, NOT payment frequency
- PREMIUM_PAYMENT_FREQUENCY: how customer pays premiums (Annual/Quarterly/etc.)
  or "N/A" for loan/deposit products
- PROVIDER_NAME: for BNK products, bank name only; never an insurance company
- COVERAGE_AMOUNT: for BNK products, "N/A" unless specific PKR/USD cover
  amounts are stated; insurance RATES (% p.a.) are NOT coverage amounts
- Use " | " as the separator for every multi-value field (never semicolons)
- SPECIAL_CONDITIONS: max 250 chars; only conditions from THIS document
- SOURCE_FILE_PRODUCT: filename only, no path
- No markdown fences, no explanations outside the JSON

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

2. PLAN_TYPE: For any insurer/takaful-underwritten product (LEAD_MARKER="IBG"),
   does it contain the word "Insurance" (optionally with a short qualifier like
   "Savings & Protection Insurance" or "Insurance (Hospitalization)")?
   For a bank-direct product (LEAD_MARKER="BNK"), is it one of
   Deposit|Loan|Card|Service|Loyalty|Investment|Savings (never "Insurance")?

3. TARGET_GOAL: Standardized short value like "Protection", "Savings",
   "Education", "Health", "Marriage", "Solar Energy Financing",
   "Green Energy Financing", "Vehicle Financing", "Housing", "SME Financing",
   "Business Financing", "Personal Financing".

4. CUSTOMER_TYPE: Are all values from Salaried|Self-Employed|SME|Corporate|
   Retail|Government, joined by " | " if more than one? No free text/bank names.
   Extract ONLY types explicitly named in the document. "Individuals" maps to
   "Salaried | Self-Employed". Do NOT add Corporate/Retail unless those exact
   category words appear in the document.

5. CUSTOMER_SEGMENT/TARGET_SEGMENT/SEGMENT_TIER: Clean values or N/A

6. CHANNEL: "Bank Branch" if branches mentioned

7. ELIGIBILITY_TYPE: Concise summary including age, income, and CNIC rules
   for all customer segments mentioned.

8. GENDER: Is it exactly one of Male|Female|All|N/A?
   "All" if the product is offered broadly with no gender restriction and
   eligibility info is present. "N/A" only if no customer/eligibility
   information is given at all.

9. FINANCING_TYPE: For unit-linked plans (PIA, fund allocation) → "Unit Linked"
   Hybrid (bonus + unit-linked) → "Hybrid (Bonus Based and Unit Linked)"
   NOT "N/A" for plans that explicitly mention unit-linked structure.

10. DEPOSIT_PROFIT_TYPE / DEPOSIT_PROFIT_FREQUENCY: Unit-linked plans → both "N/A"
    Health/protection plans (no savings) → both "N/A"
    Loan products → both "N/A"
    Do NOT set "At Maturity" for unit-linked plans.

11. TENURE_OPTIONS: Is it ONLY plan duration choices (e.g. "5 | 10 | 15 | 20 years")?
    Payment frequencies belong in PREMIUM_PAYMENT_FREQUENCY.
    If no distinct plan duration menu → "N/A"

12. PREMIUM_PAYMENT_FREQUENCY: Is it the payment frequency (Annual/Semi-Annual/Quarterly/Monthly)?
    For loan or deposit products → "N/A" (repayment schedule ≠ premium payment).
    Example: "Annual | Semi-Annual | Quarterly" or "N/A"

13. Are all multi-value fields separated by " | " (never semicolons, never
    commas as the list separator)?
    WRONG: "Accidental Death; Income Benefit"  or  "Accidental Death, Income Benefit"
    CORRECT: "Accidental Death | Income Benefit"

14. Numeric fields: Do single-value numeric fields contain ONLY integers
    (no PKR, no commas, no .0)? Tiered fields (MIN_BALANCE, MIN_INCOME,
    MIN_INVESTMENT, MIN_CONTRIBUTION, etc.) may legitimately keep a full
    tiered table instead of one number if the source document tiers them.
    Fields: MIN_AGE, MAX_AGE, MIN_TERM_YEARS, MAX_TERM_YEARS,
    FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED must always be plain integers.
    WRONG: "18.0", "PKR 250,000"  CORRECT: "18", "250000"

15. FREE_LOOK_PERIOD_DAYS: Is it set ONLY because this product explicitly mentions it?
    If not explicitly stated → "N/A". Do NOT default to 14.

16. MIN_CONTRIBUTION vs PRICING_RATE: tiered premium/contribution amounts
    belong in MIN_CONTRIBUTION, not PRICING_RATE. PRICING_RATE is reserved
    for interest/profit/markup rates. MIN_CONTRIBUTION="N/A" for loan products.

17. OPTIONAL_RIDERS vs KEY_BENEFITS: OPTIONAL_RIDERS should only contain
    items the document explicitly labels as optional add-ons, not the
    product's core/default benefits (which belong in KEY_BENEFITS).
    Loan products → OPTIONAL_RIDERS="N/A".

18. SEGMENT_TIER / SERVICE_TYPE / CUSTOMER_SEGMENT / TARGET_SEGMENT:
    "N/A" unless explicitly stated in the document. Do NOT derive from other fields.

19. SOURCE_FILE_PRODUCT: Filename only (no folder path).

20. All 56 fields present? No null/None/NaN/empty string → "N/A"
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

21. LEAD_MARKER + PLAN_TYPE consistency:
    If LEAD_MARKER="BNK" → PLAN_TYPE must NOT contain "Insurance". Fix to the
    correct bank-only value: Loan|Deposit|Card|Service|Loyalty|Investment|Savings.
    If LEAD_MARKER="IBG" → PLAN_TYPE MUST contain "Insurance".

22. LEAD_MARKER + FINANCING_TYPE consistency:
    If LEAD_MARKER="BNK" and the document has no fund/unit-allocation/PIA language:
    FINANCING_TYPE must be Conventional|Islamic|N/A — never "Unit Linked" or
    "Hybrid (Bonus Based and Unit Linked)". A bank loan with KIBOR-based markup
    and no Shariah/Islamic wording → "Conventional".

23. LEAD_MARKER + PROVIDER_NAME:
    If LEAD_MARKER="BNK": PROVIDER_NAME must be the bank name only (e.g.
    "Bank Alfalah Limited"). It must NOT contain an insurance or takaful
    company name. Fix to the bank name found in the document.

24. LEAD_MARKER + COVERAGE_AMOUNT:
    If LEAD_MARKER="BNK": COVERAGE_AMOUNT="N/A" unless the document explicitly
    states rupee or dollar sum-assured amounts for a bundled insurance component.
    Insurance RATES such as "0.49% p.a." or "0.5% p.a." are NOT coverage
    amounts — they belong in PRICING_RATE or FEES_AND_CHARGES. Fix accordingly.

25. LEAD_MARKER + insurance-specific fields:
    If LEAD_MARKER="BNK" and the product is a loan/deposit/card with no
    insurance plan structure: the following should all be "N/A" unless the
    document explicitly provides these values for a bundled insurance component:
    FREE_LOOK_PERIOD_DAYS, OPTIONAL_RIDERS, PREMIUM_PAYMENT_FREQUENCY,
    MIN_CONTRIBUTION.

26. EQUITY_REQUIREMENT: Must be a short percentage (e.g. "20%"), not a verbose
    phrase. Strip "Minimum", "Min.", "At least" prefixes and parenthetical
    qualifiers. WRONG: "Minimum 20% (Salaried & Business)"  CORRECT: "20%"

27. MIN_INCOME for multi-segment products: If the document gives different
    income thresholds for different segments (e.g. Salaried 50,000 and
    Self-Employed 100,000), use the tiered format:
    "Salaried:50000 | Self-Employed:100000"
    Do NOT use only the lower segment's income and discard the others.

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


def get_raw_generation(model, tokenizer, prompt, max_new_tokens=None):
    effective_max_new_tokens = max_new_tokens or MAX_NEW_TOKENS

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": effective_max_new_tokens,
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

    hit_token_limit = generated_ids.shape[-1] >= effective_max_new_tokens
    if hit_token_limit:
        # The model ran out of budget before emitting EOS on its own, which
        # means the JSON is almost certainly truncated mid-field rather than
        # genuinely malformed. Surface this distinctly from a real parse
        # error so it's obvious in the logs which one you're dealing with.
        print(
            f"    note: generation hit max_new_tokens={effective_max_new_tokens} "
            f"(output likely truncated, not malformed) — attempting auto-close recovery"
        )

    # Explicitly drop tensor references to free GPU memory before next call.
    del inputs, output_ids, generated_ids
    free_gpu_memory()

    return text


def extract_one_chunk(model, tokenizer, entry, chunk, chunk_idx, chunk_total, max_new_tokens=None):
    """
    Run extraction on a SINGLE chunk of the document (initial attempt +
    one compact repair retry if output is not parseable JSON).
    Validation is done once on the final merged record, not per chunk.
    Returns a normalized record dict, or None if both attempts failed.
    """
    prompt = build_prompt(entry, chunk, tokenizer, chunk_idx, chunk_total)
    raw = get_raw_generation(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

    try:
        parsed = parse_json_blob(raw)
        return normalize_record(parsed, entry)
    except Exception:
        # Compact repair prompt (no full SYSTEM_PROMPT) to stay within context
        repair_prompt = build_repair_prompt(entry, raw)
        repaired_raw = get_raw_generation(model, tokenizer, repair_prompt, max_new_tokens=max_new_tokens)
        try:
            repaired_parsed = parse_json_blob(repaired_raw)
            return normalize_record(repaired_parsed, entry)
        except Exception as second_error:
            print(
                f"  Warning: chunk {chunk_idx}/{chunk_total} unparseable after "
                f"repair attempt, skipping this chunk: {second_error}"
            )
            return None


def extract_one_product(model, tokenizer, entry, text, max_new_tokens=None):
    """
    Chunked extraction + field-level merge + single validation pass.

    The document is split into TEXT_CHUNK_SIZE-char pieces. Each piece is
    sent to the model separately. Results are merged field-by-field
    (first real value found for a field across chunks wins). A single
    validation pass runs on the final merged record.

    If ENABLE_CHUNKING=False, the whole document is sent in one call
    (risks OOM on long documents + large system prompt).

    max_new_tokens: optional override (used by the OOM retry loop in main()
    to shrink the generation budget — and therefore the KV-cache/activation
    memory — on subsequent attempts instead of repeating an identical call).
    """
    if ENABLE_CHUNKING:
        chunks = chunk_text(text, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
    else:
        chunks = [text.strip()]
    chunk_total = len(chunks)

    accumulated = blank_record(entry)
    any_chunk_succeeded = False

    for i, chunk in enumerate(chunks, start=1):
        chunk_record = extract_one_chunk(
            model, tokenizer, entry, chunk, i, chunk_total, max_new_tokens=max_new_tokens
        )
        if chunk_record is not None:
            any_chunk_succeeded = True
            accumulated = merge_records(accumulated, chunk_record, COLUMNS)
        free_gpu_memory()

    if not any_chunk_succeeded:
        return blank_record(entry)

    # Single validation/correction pass on the merged record.
    validation_prompt = build_validation_prompt(entry, accumulated)
    validation_raw = get_raw_generation(model, tokenizer, validation_prompt, max_new_tokens=max_new_tokens)
    try:
        validated_parsed = parse_json_blob(validation_raw)
        return normalize_record(validated_parsed, entry)
    except Exception:
        # If the validation call itself fails to parse, the merged record
        # (already normalized field-by-field) is still a valid result.
        return accumulated


def extract_one(model, tokenizer, entry, text, max_new_tokens=None):
    """Main extraction entry point: chunked extraction + merge + validation."""
    return extract_one_product(model, tokenizer, entry, text, max_new_tokens=max_new_tokens)


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

    # FIX: always stream weights directly into place instead of materializing
    # a full-precision copy first — this alone can be several extra GiB on a
    # 7B model.
    model_kwargs["low_cpu_mem_usage"] = True

    if LOAD_IN_4BIT or LOAD_IN_8BIT:
        # FIX: do NOT also set model_kwargs["torch_dtype"] here. Passing a
        # top-level torch_dtype alongside quantization_config is what caused
        # this process to hold ~14GiB right after loading (essentially the
        # full fp16 model), instead of the ~4-5GiB a real 4-bit 7B model
        # should take. bnb_4bit_compute_dtype below is the correct place to
        # control dtype when quantizing.
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif TORCH_DTYPE == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif TORCH_DTYPE == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif TORCH_DTYPE == "float32":
        model_kwargs["torch_dtype"] = torch.float32
    elif TORCH_DTYPE == "auto":
        model_kwargs["torch_dtype"] = "auto"

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

    if torch.cuda.is_available():
        allocated_gib = torch.cuda.memory_allocated() / (1024**3)
        reserved_gib = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"Post-load GPU memory: {allocated_gib:.2f} GiB allocated, "
            f"{reserved_gib:.2f} GiB reserved "
            f"(expect ~4-6 GiB for a 4-bit 7B model — if this is much higher, "
            f"quantization isn't actually shrinking memory)."
        )

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
                # FIX: shrink the generation budget on each retry. Retrying
                # with the exact same max_new_tokens after an OOM just fails
                # the same way again — empty_cache() only frees already-
                # reserved-but-unused memory, it can't create headroom that
                # wasn't there. Cutting the token budget shrinks the
                # KV-cache/activation memory generate() needs, giving later
                # attempts an actual chance to succeed instead of a
                # guaranteed repeat failure.
                attempt_max_new_tokens = max(
                    400, int(MAX_NEW_TOKENS * (0.6 ** attempt))
                )
                try:
                    data = extract_one(
                        model, tokenizer, entry, text,
                        max_new_tokens=attempt_max_new_tokens,
                    )
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
                        f"CUDA OOM on attempt {attempt + 1} "
                        f"(max_new_tokens={attempt_max_new_tokens}), "
                        f"cleared cache and retrying: {e}"
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