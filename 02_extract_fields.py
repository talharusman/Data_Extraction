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
- All previous functionality (chunking, merge, repair/validation passes,
  JSON recovery, model loading) preserved unchanged.

CHANGES IN THIS VERSION (audit fixes — generic, not product-specific):
- CUSTOMER_SEGMENT / TARGET_SEGMENT / SEGMENT_TIER wiped by validation pass:
  The validation prompt previously contained rule 18 ("N/A unless explicitly
  stated in the document"), but the validation pass only sees the extracted
  JSON — not the source document.  Qwen2.5-3B, being conservative, would
  reset any non-N/A value for these fields to "N/A" because it could not
  verify "explicitly stated."  Rule 18 has been replaced by rule 5 which
  instructs the model to PRESERVE existing non-N/A values and only clears
  fields that are already empty/null.  The extraction-time constraint (G7 in
  the system prompt) is still in force where the model has access to the text.
- MIN_CONTRIBUTION tiered value incorrectly collapsed: _is_tiered_value now
  also accepts a dash ("-") as the label-to-amount separator, so models that
  emit "Bronze - 5000" still trigger the tiered path.  Before calling the
  general delimiter pass (which skips commas for TIER_AWARE_NUMERIC_COLUMNS),
  a pre-pass converts inter-tier commas (", Letter") to " | " — thousands-
  separators (",NNN") are never followed by a letter, so this is unambiguous.
- MIN_CONTRIBUTION extracted into MIN_INVESTMENT: Added a cross-field post-
  processing step in normalize_record: for IBG (insurer-underwritten) products,
  if MIN_CONTRIBUTION is N/A and MIN_INVESTMENT is not, the value is moved to
  MIN_CONTRIBUTION and MIN_INVESTMENT is cleared.  Generalises to any insurer/
  takaful product across all documents.
- TENURE_OPTIONS / PREMIUM_PAYMENT_FREQUENCY confusion: Added a cross-field
  post-processing step that detects payment-frequency terms (Annual, Quarterly,
  Monthly, Semi-Annual, Bi-Annual) placed in TENURE_OPTIONS and moves them to
  PREMIUM_PAYMENT_FREQUENCY when that field is empty, leaving only genuine plan
  duration items (those containing a digit) in TENURE_OPTIONS.
- Validation prompt updated: added explicit ELIGIBILITY_TYPE vs TARGET_SEGMENT
  contrast and MIN_CONTRIBUTION vs MIN_INVESTMENT rule (rule 16) so the
  validation pass actively corrects both confusions when the initial pass missed
  them.
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
TEXT_CHUNK_OVERLAP = env_int("TEXT_CHUNK_OVERLAP", 300)
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

    Accepts colon, equals, or dash as the label-to-amount separator so the
    detector fires regardless of which punctuation the model chose.
    Generalizes to any provider's tier naming (Bronze/Silver/Gold/Platinum
    or any other word-based tier label), not just this dataset.
    """
    # Pattern: "Word(s) <sep> number" where sep is ':', '=', or '-'
    pairs = re.findall(r"[A-Za-z][\w\s]{0,24}[:=\-]\s*[\d,]+", stripped)
    return len(pairs) >= 2


def truncate_to_boundary(value: str, max_len: int) -> str:
    """Truncate text at a word or punctuation boundary when possible."""
    if len(value) <= max_len:
        return value

    cut = value[:max_len].rstrip()
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
    """Normalize TARGET_GOAL to corrected style."""
    if not value or value == DEFAULT_VALUE:
        return DEFAULT_VALUE
    val_lower = value.lower()
    mappings = {
        "protection": "Protection",
        "accidental": "Protection",
        "savings": "Savings",
        "education": "Education",
        "health": "Health",
        "hospitalization": "Health",
        "marriage": "Marriage",
        "multipurpose": "Multipurpose Savings",
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
    """Clean ELIGIBILITY_TYPE."""
    if not value or value == DEFAULT_VALUE:
        return DEFAULT_VALUE
    # Keep concise
    value = re.sub(r"Bank Alfalah Limited?", "Bank Alfalah", value, flags=re.I)
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
    - Numeric fields: strip units, commas, currency symbols — UNLESS the
      value is a genuine tiered table for a tier-aware numeric field, in
      which case the tiered text is preserved (see _is_tiered_value)
    - List-type text fields: delimiter normalized to " | "
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
                #
                # FIX: Before the general delimiter pass, normalise inter-tier
                # commas.  The general pass skips commas for TIER_AWARE fields
                # (they're not in COMMA_IS_LIST_SEPARATOR_FIELDS) because commas
                # also appear INSIDE amounts ("5,000"). However a comma followed
                # by a letter is unambiguous as a list separator — thousands-
                # separators are always followed by digits, never letters.
                # e.g. "Bronze:5,000, Silver:10,000" → "Bronze:5,000 | Silver:10,000"
                stripped = re.sub(r",\s+(?=[A-Za-z])", " | ", stripped)
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

    # ----------------------------------------------------------------
    # Cross-field post-processing (generic, not product-specific)
    # ----------------------------------------------------------------

    # FIX A: MIN_CONTRIBUTION / MIN_INVESTMENT swap for insurance products.
    # Qwen2.5-3B conflates "contribution/premium" with "investment" for
    # insurance plans.  For any IBG (insurer-underwritten) product, premiums
    # and contribution amounts belong in MIN_CONTRIBUTION, not MIN_INVESTMENT.
    # If MIN_CONTRIBUTION is empty but MIN_INVESTMENT is not, move the value.
    # MIN_INVESTMENT is kept as N/A because it is not a concept that applies
    # to insurance/takaful plans (it is for portfolio or fund products).
    lead_marker = normalized.get("LEAD_MARKER", DEFAULT_VALUE)
    if lead_marker == "IBG":
        if normalized.get("MIN_CONTRIBUTION", DEFAULT_VALUE) == DEFAULT_VALUE:
            rescued = normalized.get("MIN_INVESTMENT", DEFAULT_VALUE)
            if rescued != DEFAULT_VALUE:
                normalized["MIN_CONTRIBUTION"] = rescued
                normalized["MIN_INVESTMENT"] = DEFAULT_VALUE

    # FIX B: TENURE_OPTIONS / PREMIUM_PAYMENT_FREQUENCY confusion.
    # The model often places payment frequencies (Annual, Quarterly, etc.)
    # in TENURE_OPTIONS instead of PREMIUM_PAYMENT_FREQUENCY.  Detect this
    # by checking whether TENURE_OPTIONS items look like payment periods
    # (no year/number in them) rather than plan durations (contain a digit).
    _FREQ_TERMS = {"annual", "semi-annual", "semi annual", "quarterly", "monthly", "bi-annual"}
    tenure_opts_val = normalized.get("TENURE_OPTIONS", DEFAULT_VALUE)
    if isinstance(tenure_opts_val, str) and tenure_opts_val != DEFAULT_VALUE:
        parts = [p.strip() for p in re.split(r"\s*\|\s*", tenure_opts_val) if p.strip()]
        freq_parts = [p for p in parts if p.lower() in _FREQ_TERMS]
        dur_parts  = [p for p in parts if p not in freq_parts]  # has digits or unknown
        if freq_parts:
            # Rescue payment frequencies into PREMIUM_PAYMENT_FREQUENCY if empty.
            ppf = normalized.get("PREMIUM_PAYMENT_FREQUENCY", DEFAULT_VALUE)
            if ppf == DEFAULT_VALUE:
                normalized["PREMIUM_PAYMENT_FREQUENCY"] = " | ".join(freq_parts)
            # Leave only genuine duration options in TENURE_OPTIONS.
            normalized["TENURE_OPTIONS"] = " | ".join(dur_parts) if dur_parts else DEFAULT_VALUE

    # Always preserve PRODUCT_NAME and SOURCE_FILE_PRODUCT.
    # BUG FIX: `record.get("PRODUCT_NAME") or entry["title"]` short-circuits
    # to "N/A" when the LLM returned "N/A" (a truthy non-empty string), so
    # the fallback to entry["title"] was NEVER reached. Explicitly check for
    # the DEFAULT_VALUE sentinel so the document title is used as a fallback
    # whenever the model couldn't extract the product name.
    raw_name = record.get("PRODUCT_NAME")
    if not raw_name or raw_name == DEFAULT_VALUE:
        raw_name = entry["title"]
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
- PLAN_TYPE: short descriptive category; must contain "Insurance" for any
  insurer/takaful-underwritten product (e.g. "Insurance",
  "Savings & Protection Insurance", "Insurance (Hospitalization)");
  otherwise one of Deposit|Loan|Card|Service|Loyalty|Investment|Savings
- CUSTOMER_TYPE: one or more of Salaried|Self-Employed|SME|Corporate|Retail|
  Government joined by " | " if multiple, or "N/A"
- GENDER: exactly one of Male|Female|All|N/A
- FINANCING_TYPE: one of Conventional|Islamic|Takaful|Mudarabah|Unit Linked|
  Hybrid (Bonus Based and Unit Linked)|N/A
- TARGET_GOAL: standardized short term like Protection, Savings, Education,
  Health, Marriage
- CHANNEL: "Bank Branch" if applicable
- Numeric fields (MIN_AGE, MAX_AGE, MIN_TERM_YEARS, MAX_TERM_YEARS,
  FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED): integers only, no units, no .0.
  MIN_BALANCE/MIN_INCOME/MIN_INCOME_USD/MIN_INVESTMENT/MIN_CONTRIBUTION:
  a single integer, OR the full tiered table text if genuinely tiered.
- TENURE_OPTIONS: plan duration choices only, NOT payment frequency
- PREMIUM_PAYMENT_FREQUENCY: how customer pays (Annual/Quarterly/etc.) or "N/A"
- Use " | " as the separator for every multi-value field (never semicolons)
- SPECIAL_CONDITIONS: max 250 chars
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

2. PLAN_TYPE: For any insurer/takaful-underwritten product, does it contain
   the word "Insurance" (optionally with a short qualifier like
   "Savings & Protection Insurance" or "Insurance (Hospitalization)")?
   For a bank-only product, is it one of Deposit|Loan|Card|Service|
   Loyalty|Investment|Savings?

3. TARGET_GOAL: Standardized short value like "Protection", "Savings", "Education", "Health", "Marriage"

4. CUSTOMER_TYPE: Are all values from Salaried|Self-Employed|SME|Corporate|
   Retail|Government, joined by " | " if more than one? No free text/bank names.

5. CUSTOMER_SEGMENT, TARGET_SEGMENT, SEGMENT_TIER: Do NOT change these to "N/A"
   if they already contain a value — preserve whatever was extracted. Only set
   "N/A" if the field is currently empty or null.
   Note the distinction:
   - CUSTOMER_SEGMENT: who the product is sold to, e.g. "All Bank Alfalah Customers"
   - TARGET_SEGMENT: marketing positioning / planning purpose, e.g. "Protection Planning"
   - SEGMENT_TIER: named tiers, e.g. "Bronze | Silver | Gold | Platinum"
   - ELIGIBILITY_TYPE: operational eligibility criteria (age, CNIC, income rules)
   Do NOT move ELIGIBILITY_TYPE content into TARGET_SEGMENT or vice versa.

6. CHANNEL: "Bank Branch" if branches mentioned.

7. ELIGIBILITY_TYPE: Concise operational eligibility summary (age range, CNIC
   uniqueness, income floor, one-policy rules). This is NOT a marketing purpose
   statement — do not replace it with positioning language.

8. GENDER: Is it exactly one of Male|Female|All|N/A?
   "All" if the product is offered broadly with no gender restriction and
   eligibility info is present. "N/A" only if no customer/eligibility
   information is given at all.

9. FINANCING_TYPE: For unit-linked plans (PIA, fund allocation) → "Unit Linked"
   Hybrid (bonus + unit-linked) → "Hybrid (Bonus Based and Unit Linked)"
   NOT "N/A" for plans that explicitly mention unit-linked structure.

10. DEPOSIT_PROFIT_TYPE / DEPOSIT_PROFIT_FREQUENCY: Unit-linked plans → both "N/A"
    Health/protection plans (no savings) → both "N/A"
    Do NOT set "At Maturity" for unit-linked plans.

11. TENURE_OPTIONS: Is it ONLY plan duration choices (e.g. "5 | 10 | 15 | 20 years")?
    Words like Annual, Quarterly, Monthly, Semi-Annual are payment frequencies,
    NOT plan durations — move them to PREMIUM_PAYMENT_FREQUENCY instead.
    If no distinct plan duration menu → "N/A".

12. PREMIUM_PAYMENT_FREQUENCY: How the customer pays (Annual/Semi-Annual/Quarterly/
    Monthly), e.g. "Annual | Semi-Annual | Quarterly". "N/A" if not stated.
    If TENURE_OPTIONS holds payment frequency terms, move them here.

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

16. MIN_CONTRIBUTION vs PRICING_RATE vs MIN_INVESTMENT:
    - Tiered premium/contribution amounts belong in MIN_CONTRIBUTION, not
      PRICING_RATE (which is for interest/profit/markup rates) and not
      MIN_INVESTMENT (which is for portfolio/fund/investment products, not
      insurance premiums).
    - For insurance/takaful plans (LEAD_MARKER="IBG"): if MIN_CONTRIBUTION is
      "N/A" but MIN_INVESTMENT has a contribution-style amount, move it to
      MIN_CONTRIBUTION and set MIN_INVESTMENT to "N/A".

17. OPTIONAL_RIDERS vs KEY_BENEFITS: OPTIONAL_RIDERS should only contain
    items the document explicitly labels as optional add-ons, not the
    product's core/default benefits (which belong in KEY_BENEFITS).

18. SOURCE_FILE_PRODUCT: Filename only (no folder path).

19. All 56 fields present? No null/None/NaN/empty string → "N/A"
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