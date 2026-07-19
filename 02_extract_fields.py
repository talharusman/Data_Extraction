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

CHANGES IN THIS VERSION (comprehensive audit fixes — generic, not product-specific):
- CRITICAL LEAD_MARKER FIX: "term finance facility" is ALWAYS a BNK (bank loan)
  product, NEVER IBG (insurance). Added explicit "green energy" and "term finance"
  signal detection that overrides any IBG classification for loan products.

- CRITICAL MIN_AGE FIX: Extract MOST RESTRICTIVE minimum age across all segments.

- CRITICAL MIN_TERM_YEARS FIX: Only extract if EXPLICITLY stated. Never default to 1.

- CRITICAL MISSING FIELD FIXES: Added extraction logic for LOAN_AMOUNT_RANGE,
  COLLATERAL_TYPE, EQUITY_REQUIREMENT.

- CRITICAL PREMIUM_PAYMENT_FREQUENCY FIX: "yearly renewable plan" → "Annual".

- CRITICAL REQUIRED_DOCUMENTS FIX (Insurance): Extract ONLY from explicit
  "Documentation Required" section. For IBG products, if no explicit section exists
  (only claim procedures or general eligibility), return "N/A" immediately.

- CRITICAL FIX — PRODUCT_VARIANT_TIER ANTI-HALLUCINATION: Extract ONLY tier names
  that appear in the document. Do NOT invent tier names. If document says "Option 1"
  and "Option 2", do NOT extract as "Bronze | Silver" (these names don't exist).

- CRITICAL FIX — PRICING_RATE vs MIN_CONTRIBUTION: Insurance premium percentages
  (e.g., "2.75% of Sum Assured") go in MIN_CONTRIBUTION, NOT PRICING_RATE.
  PRICING_RATE is LOAN interest rates only. For insurance products, PRICING_RATE="N/A".

- CRITICAL FIX — Field Applicability by Product Type: Added comprehensive validation
  that BNK products NEVER have insurance-specific fields (COVERAGE_AMOUNT,
  FREE_LOOK_PERIOD_DAYS, OPTIONAL_RIDERS, PREMIUM_PAYMENT_FREQUENCY, MIN_CONTRIBUTION,
  KEY_EXCLUSIONS, CLAIMS_SERVICE_CONTACT) and IBG products NEVER have loan-specific
  fields (LOAN_AMOUNT_RANGE, COLLATERAL_TYPE, PRICING_RATE for interest rates).

- CRITICAL FIX — Insurance Documentation Contamination (AH8): Added
  _clean_insurance_required_documents() to detect and remove loan-specific keywords
  and claim-processing language from insurance REQUIRED_DOCUMENTS field.

- CRITICAL FIX — MIN_INCOME Hallucination Prevention: Strengthened validation to
  prevent inferring income thresholds that aren't explicitly stated in the document.

- CRITICAL FIX — Hallucinated Age Restrictions: Added validation to detect and
  reject age restrictions that aren't explicitly mentioned in the document.

- CRITICAL FIX — CHANNEL vs PROVIDER_NAME: Fixed confusion between channel of access
  (Bank Branch, Mobile App, Telephone) and provider name (Bank/Insurer name).

- CRITICAL FIX — EMPLOYMENT_TYPE/CUSTOMER_TYPE Restrictions: Only extract if
  explicitly restricted. If document says "all customers", do NOT hallucinate
  employment-based segmentation.

- CRITICAL FIX — LOAN_AMOUNT_RANGE Inference: Fixed hallucination where technical
  specifications (4KW-1000KW solar capacity) were incorrectly inferred as loan amounts.

- All previous functionality (chunking, merge, repair/validation passes, JSON recovery,
  model loading) preserved unchanged with full backward compatibility.
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
# FIX (OOM): Lowered default from 2200 to 1500.  The auto-close recovery
# (_close_unterminated_json) reliably handles the rare case where 1500 is
# not enough, so we no longer need to reserve worst-case headroom that
# pushes the KV-cache into OOM territory on Colab T4/L4 GPUs.
# Override via HF_MAX_NEW_TOKENS in .env if needed.
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 1500)
# FIX: Enforce a minimum safe budget for 56-field JSON generation.
_MIN_SAFE_TOKENS = 1500
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
# FIX (OOM): Lowered default from 6000 to 4000 chars.  With the 22 KB
# system prompt each chunk call was ~8,500 input tokens.  At 4000 chars
# the input drops to ~7,000 tokens, saving ~25% KV-cache memory per call.
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", 4000)
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
    """Remove markdown fences, preamble text, and thinking tags from model
    output so the downstream JSON parsers see clean content.

    FIX (JSON recovery): The previous version only stripped leading fences
    and trailing fences.  Repair-pass output from Qwen often starts with
    conversational preamble like ``Here is the repaired JSON:\n```json``.
    We now also strip everything *before* the first ``{`` when no fence is
    found, giving the brace-walker and auto-close logic a clean start.
    """
    text = text.strip()
    # Remove <think> blocks
    text = re.sub(r"(?is)</?think>", "", text)
    # Remove markdown fences
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # FIX: If the text still doesn't start with '{' or '[', strip
    # conversational preamble ("Here is the repaired JSON ...") by
    # jumping to the first JSON-start character.
    if text and text[0] not in ('{', '['):
        first_brace = text.find('{')
        first_bracket = text.find('[')
        candidates = [i for i in (first_brace, first_bracket) if i != -1]
        if candidates:
            text = text[min(candidates):]
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
    "REQUIRED_DOCUMENTS": 280,     # FIXED: widened from 220 to accommodate full document lists
    "CLAIMS_SERVICE_CONTACT": 220, # FIXED: widened from 200 for insurance contact details
    "KEY_EXCLUSIONS": 250,         # FIXED: widened from 200 for comprehensive exclusion lists
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 30,                # widened: descriptive category, not single word
    "TARGET_GOAL": 50,             # "Children's Education Planning" style values
    "CUSTOMER_TYPE": 60,            # widened: may hold multiple "|"-joined enum values
    "EMPLOYMENT_TYPE": 500,        # full Target Market list ~334 chars
    "ACCOUNT_TYPE": 20,
    "CARD_TYPE": 25,
    "CHANNEL": 30,
    "ELIGIBILITY_TYPE": 120,       # FIXED: widened from 100 for detailed eligibility rules
    "SERVICE_TYPE": 25,
    "REWARD_TYPE": 25,
    "CURRENCY": 30,
    "CURRENCY_TYPE": 15,
    "LOAN_AMOUNT_RANGE": 60,       # FIXED: widened from 50 for capacity range descriptions
    "COVERAGE_AMOUNT": 350,        # FIXED: widened from 300 for full 9-tier coverage tables
    "FINANCING_TYPE": 50,          # "Hybrid (Bonus Based and Unit Linked)" = 36 chars
    "DEPOSIT_PROFIT_TYPE": 30,
    "DEPOSIT_PROFIT_FREQUENCY": 20,
    "TENURE": 60,                  # FIXED: widened from 50 for complex tenor descriptions
    "TENURE_OPTIONS": 60,          # FIXED: widened from 50 for multiple tenor options
    "BUSINESS_TENURE": 100,        # widened: may contain tiered tenure like "2 years SEP | 3 years SEB"
    "COLLATERAL_TYPE": 80,         # FIXED: widened from 50 for multiple property types
    "EQUITY_REQUIREMENT": 25,      # FIXED: widened from 20 to handle percentage + qualifiers
    "DBR_LIMIT": 20,
    "TRANSACTION_LIMIT": 60,       # FIXED: widened from 50
    "SPECIAL_CONDITIONS": 280,     # FIXED: widened from 250 for detailed multi-condition rules
    "PREMIUM_PAYMENT_FREQUENCY": 50,
    "PRODUCT_DESCRIPTION": 300,    # FIXED: widened from 250 for full feature descriptions
    "KEY_BENEFITS": 320,           # FIXED: widened from 280 for complete benefit lists

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
    
    CRITICAL: Do NOT set to Unit Linked or Hybrid for loan products.
    Bank loans with markup/KIBOR rates are ALWAYS Conventional or Islamic,
    never Unit Linked or Hybrid (which are insurance investment structures).
    """
    stripped = value.strip()
    if not stripped or stripped == "N/A":
        return "N/A"
    
    lower = stripped.lower()

    # Check for clear loan signals — if found, override to Conventional/Islamic
    # These keywords indicate a bank loan, not an insurance investment structure
    loan_signals = {
        "markup", "kibor", "murabaha", "musharaka", "ijarah",
        "term loan", "credit facility", "financing facility", 
        "overdraft", "conventional bank", "islamic bank"
    }
    
    if any(sig in lower for sig in loan_signals):
        # This is a loan product, not insurance
        if any(word in lower for word in ["islamic", "murabaha", "musharaka", "ijarah", "shariah"]):
            return "Islamic"
        else:
            return "Conventional"

    # Hybrid check (insurance-only structures)
    if "hybrid" in lower or ("bonus" in lower and "unit" in lower):
        return "Hybrid (Bonus Based and Unit Linked)"

    # Unit Linked (insurance-only)
    if "unit linked" in lower or "unit-linked" in lower or "pia" in lower:
        # But if loan signals are present, this is a mistake — use Conventional
        if any(sig in lower for sig in loan_signals):
            return "Conventional"
        return "Unit Linked"

    # Single-word canonicals
    canonical_map = {
        "conventional": "Conventional",
        "islamic": "Islamic",
        "takaful": "Takaful",
        "mudarabah": "Mudarabah",
    }
    for key, canonical in canonical_map.items():
        if key in lower:
            return canonical

    # Pass through if already in known form
    known_exact = {
        "Conventional", "Islamic", "Takaful", "Mudarabah",
        "Unit Linked", "Hybrid (Bonus Based and Unit Linked)", "N/A",
    }
    if stripped in known_exact:
        return stripped

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



def _normalize_employment_type(value: str) -> str:
    """
    EMPLOYMENT_TYPE may hold MULTIPLE enum values (Salaried|Self-Employed|
    Contract|Permanent|Proprietor|Partner|Director), so we preserve all valid
    matches separated by " | ". This fixes the previous behavior of collapsing
    to a single value, which lost employment eligibility information.
    """
    allowed = {
        "Salaried", "Self-Employed", "Contract", "Permanent",
        "Proprietor", "Partner", "Director", "Business Owner",
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


def _extract_tenure_years(tenure_text: str) -> tuple[int | None, int | None]:
    """
    Extract MIN_TERM_YEARS and MAX_TERM_YEARS from tenure description text.
    
    CRITICAL: "Up to X years" (no minimum stated) should return (None, X),
    NOT (1, X). Never default MIN to 1 without explicit statement.
    
    Examples:
      "Up to 10 years" → (None, 10)  # No minimum stated
      "Minimum 2 to 10 years" → (2, 10)
      "5-10 years" → (5, 10)
      "1 Year" → (1, 1)  # Only if explicitly "1 year"
      "10-67 years" → (10, 67)
    
    Returns: (min_years, max_years) or (None, None) if extraction fails.
    """
    if not tenure_text or tenure_text == "N/A":
        return None, None
    
    text = tenure_text.lower().strip()
    
    # Pattern 1: "Up to X years" or "Up to X year" — NO minimum
    match = re.search(r"up\s+to\s+(\d+)\s+years?", text)
    if match:
        max_year = int(match.group(1))
        return None, max_year  # No minimum, only maximum
    
    # Pattern 2: "X-Y years" or "X to Y years"
    match = re.search(r"(\d+)\s*[-to\s]+\s*(\d+)\s+years?", text)
    if match:
        min_year = int(match.group(1))
        max_year = int(match.group(2))
        return min_year, max_year
    
    # Pattern 3: Single value "X year" or "X years"
    match = re.search(r"^(\d+)\s+years?$", text)
    if match:
        year = int(match.group(1))
        return year, year
    
    return None, None


def _normalize_premium_payment_frequency(value: str) -> str:
    """
    Normalize PREMIUM_PAYMENT_FREQUENCY for insurance products.
    
    Valid values: Annual, Semi-Annual, Quarterly, Monthly, Weekly, etc.
    Multiple frequencies should be pipe-separated: "Annual | Semi-Annual | Monthly"
    
    This field should ONLY be populated for insurance products (LEAD_MARKER="IBG").
    For loans/deposits → "N/A".
    """
    if not value or value == "N/A":
        return "N/A"
    
    stripped = value.strip()
    if not stripped or stripped.upper() == "N/A":
        return "N/A"
    
    # Normalize common frequency names
    freq_map = {
        "annual": "Annual",
        "yearly": "Annual",
        "semi-annual": "Semi-Annual",
        "semi annual": "Semi-Annual",
        "semiannual": "Semi-Annual",
        "bi-annual": "Semi-Annual",
        "half-yearly": "Semi-Annual",
        "quarterly": "Quarterly",
        "monthly": "Monthly",
        "weekly": "Weekly",
        "daily": "Daily",
        "monthly": "Monthly",
        "fortnightly": "Fortnightly",
        "maturity": "At Maturity",
    }
    
    lower = stripped.lower()
    
    # If it's a single frequency, map it
    for key, canonical in freq_map.items():
        if lower == key or (len(lower) > 5 and key in lower):
            return canonical
    
    # If it contains multiple frequencies separated by comma or pipe, normalize each
    if "|" in stripped or "," in stripped:
        parts = re.split(r"[|,]", stripped)
        normalized_parts = []
        for part in parts:
            part_clean = part.strip().lower()
            found = False
            for key, canonical in freq_map.items():
                if part_clean == key or (key in part_clean and len(key) > 3):
                    if canonical not in normalized_parts:
                        normalized_parts.append(canonical)
                    found = True
                    break
            if not found and part.strip():
                # Keep unrecognized part as-is (title case)
                normalized_parts.append(part.strip())
        if normalized_parts:
            return " | ".join(normalized_parts)
    
    # Return as-is if already looks canonical
    return stripped


def _validate_and_fix_product_name(extracted_name: str, doc_text: str) -> str:
    """
    Validate PRODUCT_NAME against document content. If the extracted name doesn't
    appear in the document, it's likely hallucinated — return "N/A" instead.
    
    This prevents completely fictional product names from slipping through.
    """
    if not extracted_name or extracted_name == "N/A":
        return "N/A"
    
    name = extracted_name.strip()
    if not name or name.upper() == "N/A":
        return "N/A"
    
    text_lower = doc_text.lower()
    name_lower = name.lower()
    
    # Exact match
    if name_lower in text_lower:
        return name
    
    # Check if at least the first 2+ significant words appear together
    words = name_lower.split()
    if len(words) >= 2:
        # Look for the first two words together
        first_two = f"{words[0]} {words[1]}"
        if first_two in text_lower:
            return name
        
        # Check if major words (len > 3) appear in the document even separately
        major_words = [w for w in words if len(w) > 3]
        if len(major_words) >= 2:
            if all(mw in text_lower for mw in major_words[:2]):
                return name
    
    # If none of the checks pass, the name is likely hallucinated
    return "N/A"


def _clean_insurance_required_documents(required_docs: str) -> str:
    """
    CRITICAL FIX: For insurance products (IBG), REQUIRED_DOCUMENTS must not
    contain loan-specific keywords or claim-processing language. If found,
    return "N/A" (do not hallucinate).
    
    Loan keywords to NEVER appear in insurance REQUIRED_DOCUMENTS:
    "salary slip", "employment certificate", "bank statement", "tax return",
    "proprietorship", "processing fee", "property documents", "collateral"
    
    Claim processing phrases to detect (these are CLAIMS procedures, not
    upfront documentation requirements):
    "claim processing", "step 1", "step 2", "inform alfalah", "call and inform"
    
    These indicate incorrect extraction from a loan section, claim section,
    or hallucination, rather than an insurance-specific documentation section.
    """
    if not required_docs or required_docs == "N/A":
        return "N/A"
    
    loan_contamination_keywords = {
        "salary slip", "salary slips",
        "employment certificate", "employment cert",
        "bank statement", "bank statements",
        "tax return", "tax returns", "income tax return",
        "proprietorship", "proprietor",
        "processing fee",
        "property document", "property documents",
        "collateral",
        "title deed", "ownership certificate",
        "noc",  # No Objection Certificate (used in loan collateral)
    }
    
    claim_processing_phrases = {
        "claim processing", "processing a claim",
        "step 1", "step 2", "step 3",
        "call and inform", "inform alfalah", "inform the insurer",
        "inform police", "get a fir", "provide the required",
        "claim has never been", "within 24 hours", "within 48 hours",
        "fir", "police"
    }
    
    docs_lower = required_docs.lower()
    
    # Check if ANY loan contamination keyword appears
    for keyword in loan_contamination_keywords:
        if keyword in docs_lower:
            # This REQUIRED_DOCUMENTS field is contaminated with loan-specific docs
            # For insurance, if explicit documentation section wasn't found, return N/A
            return "N/A"
    
    # Check if this is actually CLAIM PROCESSING text, not documentation requirement
    for phrase in claim_processing_phrases:
        if phrase in docs_lower:
            # This is claim processing procedure, not upfront documentation requirement
            return "N/A"
    
    return required_docs



def _validate_product_variant_tier(tier_value: str, doc_text: str) -> str:
    """
    CRITICAL FIX: Validate that PRODUCT_VARIANT_TIER values actually appear
    in the document. Prevents hallucination of tier names like "Bronze | Silver"
    when document actually says "Option 1 | Option 2".
    
    If tier names cannot be verified in document, return "N/A".
    """
    if not tier_value or tier_value == "N/A":
        return tier_value
    
    doc_lower = doc_text.lower()
    tier_lower = tier_value.lower()
    
    # If any tier name appears in the document, it's likely valid
    tiers = [t.strip() for t in tier_value.split('|')]
    verified_tiers = []
    
    for tier in tiers:
        tier_clean = tier.strip().lower()
        # Check if this tier name appears in document
        if tier_clean in doc_lower:
            verified_tiers.append(tier)
    
    # Return only verified tiers
    if verified_tiers:
        return ' | '.join(verified_tiers)
    
    # If NO tiers verified, it's hallucinated → return N/A
    return "N/A"


def _validate_employment_restrictions(employment_value: str, doc_text: str) -> str:
    """
    CRITICAL FIX: Prevent hallucination of employment restrictions.
    If document says "all Bank Alfalah customers" with no employment restriction,
    do NOT hallucinate "Salaried | Self-Employed".
    
    If employment types are claimed but document says "all customers", return "N/A".
    """
    if not employment_value or employment_value == "N/A":
        return employment_value
    
    doc_lower = doc_text.lower()
    
    # Check for "all customers" language
    all_customer_phrases = [
        "all bank alfalah customers",
        "all bank alfalah limited customers",
        "all customers",
        "available to all",
        "open to all",
        "eligible to all"
    ]
    
    if any(phrase in doc_lower for phrase in all_customer_phrases):
        # Document says "all customers" with no employment restriction
        # Any employment-based segmentation is hallucinated
        return "N/A"
    
    # Otherwise keep the employment value if it's reasonable
    return employment_value


def _validate_min_term_years(min_term: str, max_term: str, doc_text: str) -> str:
    """
    CRITICAL FIX (AH5a): MIN_TERM_YEARS must ONLY be extracted if document
    EXPLICITLY states a minimum tenure/term.
    
    Do NOT extract from "Up to X years" (that's only MAX_TERM_YEARS).
    Do NOT default to 1.
    
    Valid formats: "Minimum 1 year", "Min 2 years", "Term from 1 to 10 years", etc.
    Invalid: "Up to 10 years" (only MAX), "Generally 5-10 years" (vague range)
    """
    if not min_term or min_term == "N/A":
        return "N/A"
    
    doc_lower = doc_text.lower()
    
    # Patterns that indicate explicit MINIMUM terms (not maximum)
    min_term_patterns = [
        r"minimum\s+(?:tenure|term|years?)\s*(?:of\s+)?(\d+)",
        r"min(?:imum)?\s+(?:tenure|term|years?)\s*(?:of\s+)?(\d+)",
        r"(?:tenure|term)\s+(?:from|starting)\s+(\d+)\s+(?:years?|yrs?)",
        r"(?:tenure|term)\s+(?:minimum|min)\s+(\d+)",
        r"(?:min|minimum)\s+(?:\d+)\s+(?:to|through|-)\s+(?:\d+)\s+years?",
    ]
    
    # Check if document has any explicit minimum term statement
    has_minimum_statement = any(re.search(pattern, doc_lower) for pattern in min_term_patterns)
    
    # CRITICAL: "Up to X years" or "Maximum X years" does NOT indicate minimum
    # If document only has "Up to" language, there's no minimum stated
    has_only_maximum = ("up to" in doc_lower or "upto" in doc_lower or "maximum" in doc_lower) and \
                       not has_minimum_statement
    
    # CRITICAL FIX: If document contains "Up to" and NO explicit minimum phrasing,
    # return N/A immediately regardless of extracted min_term value.
    # This prevents hallucinated defaults like "1" when only max is stated.
    if "up to" in doc_lower and not has_minimum_statement:
        return "N/A"
    
    if has_only_maximum or not has_minimum_statement:
        # Document only states maximum term, not minimum
        return "N/A"
    
    # Minimum term is explicitly stated
    return min_term


def _validate_age_requirements(min_age: str, max_age: str, doc_text: str) -> tuple:
    """
    CRITICAL FIX: Prevent hallucination of age restrictions.
    Only accept extracted ages if document EXPLICITLY states numeric age values.
    
    Returns ("N/A", "N/A") if:
    - Document says "available to all" / "all customers" (no restriction)
    - Document mentions age context but doesn't state explicit MINIMUM/MAXIMUM numbers
    - Document only states segment-specific ages (e.g., "Salaried: 25" but not for all segments)
    """
    if not min_age or not max_age:
        return ("N/A", "N/A")
    
    doc_lower = doc_text.lower()
    
    # CRITICAL: If document explicitly says "all customers" or "available to all",
    # there is NO age restriction. Any extracted ages are hallucinated.
    all_customer_phrases = [
        "available to all",
        "all bank alfalah",
        "all customers",
        "open to all",
        "eligible to all",
        "available to all bank alfalah"
    ]
    
    if any(phrase in doc_lower for phrase in all_customer_phrases):
        return ("N/A", "N/A")
    
    # Check for explicit age numbers in common formats
    # Valid formats: "18 years", "25 years old", "age 60", "minimum 25", "max 65", etc.
    import re
    
    min_age_patterns = [
        r"minimum\s+(?:age\s+)?(\d+)",
        r"min(?:imum)?\s+(?:age\s+)?(\d+)",
        r"age\s+(?:minimum\s+)?(\d+)",
        r"(?:age|from)\s+(\d+)\s+(?:years?|yrs?)"
    ]
    
    max_age_patterns = [
        r"maximum\s+(?:age\s+)?(\d+)",
        r"max(?:imum)?\s+(?:age\s+)?(\d+)",
        r"age\s+(?:up to|upto|maximum)\s+(\d+)",
        r"(?:up to|upto)\s+(\d+)\s+(?:years?|yrs?)"
    ]
    
    min_found = any(re.search(pattern, doc_lower) for pattern in min_age_patterns)
    max_found = any(re.search(pattern, doc_lower) for pattern in max_age_patterns)
    
    # CRITICAL: Only accept both min AND max if both are explicitly stated
    # If only one is stated or if ages are only segment-specific, return N/A
    if not (min_found and max_found):
        # Ages not explicitly stated as global requirements
        return ("N/A", "N/A")
    
    # Both ages explicitly found in patterns, accept the values
    return (min_age, max_age)



def _crossfield_validate(normalized: dict) -> dict:
    """
    Apply cross-field consistency rules that cannot be enforced on a
    field-by-field basis during per-column normalization.

    Called before other cross-field corrections to enforce basic consistency
    between product type (LEAD_MARKER) and field-level values.
    """
    lead = normalized.get("LEAD_MARKER", "").strip().upper()
    
    if not isinstance(lead, str) or lead not in ("BNK", "IBG"):
        return normalized

    # ================================================================
    # BNK (Bank-Direct Loan/Deposit/Account/Card) Product Rules
    # ================================================================
    if lead == "BNK":
        # Bank products cannot have fund-based financing (Unit Linked or Hybrid)
        # These are insurance/investment structures, not bank lending structures
        fin_type = normalized.get("FINANCING_TYPE", "")
        if isinstance(fin_type, str) and fin_type in (
            "Unit Linked", "Hybrid (Bonus Based and Unit Linked)"
        ):
            normalized["FINANCING_TYPE"] = "Conventional"
        
        # Insurance-specific fields MUST be N/A for bank products per G12
        insurance_fields = {
            "COVERAGE_AMOUNT",          # Insurance coverage amounts
            "FREE_LOOK_PERIOD_DAYS",    # Insurance free-look period
            "OPTIONAL_RIDERS",          # Insurance optional riders
            "PREMIUM_PAYMENT_FREQUENCY", # Insurance premium payment mode
            "MIN_CONTRIBUTION",         # Insurance premium minimum
            "KEY_EXCLUSIONS",           # Insurance exclusions
            "CLAIMS_SERVICE_CONTACT",   # Insurance claims contact
        }
        for field in insurance_fields:
            if field in normalized:
                current = normalized[field]
                if isinstance(current, str) and current not in ("N/A", ""):
                    # This is a bank product but has an insurance-specific field populated
                    # Set it to N/A per spec
                    normalized[field] = "N/A"
        
        # CRITICAL FIX: For loan products, PRICING_RATE should contain rate info
        # (e.g. "1 Year KIBOR + 3%"), NOT premium amounts. If MIN_CONTRIBUTION
        # has numeric-only values that look like loan amounts, it's probably
        # misplaced — clear it for bank products
        min_contrib = normalized.get("MIN_CONTRIBUTION", "")
        if isinstance(min_contrib, str) and min_contrib not in ("N/A", ""):
            # If it looks like a premium (e.g. "5000 | 10000 | 15000"), clear it for loans
            if re.search(r"^\d+(\s*\|\s*\d+)*$", min_contrib.strip()):
                normalized["MIN_CONTRIBUTION"] = "N/A"

    # ================================================================
    # IBG (Insurance/Takaful-Underwritten Product) Rules
    # ================================================================
    elif lead == "IBG":
        # Insurance products should not have loan-specific fields populated
        # with actual values (these are for BNK products only)
        loan_fields = {
            "LOAN_AMOUNT_RANGE",    # Only loans have amount ranges
            "COLLATERAL_TYPE",      # Only loans have collateral
            "DBR_LIMIT",           # Only loans have debt ratios
        }
        for field in loan_fields:
            if field in normalized:
                current = normalized[field]
                # Only clear if it looks like a loan field got populated by mistake
                if isinstance(current, str) and current not in ("N/A", "") and \
                   any(kw in current.lower() for kw in ["million", "thousand", "k", "m", "pkr", "usd", "million", "lending"]):
                    normalized[field] = "N/A"
        
        # CRITICAL FIX: For insurance products, PRICING_RATE should be N/A
        # (insurance premiums go in MIN_CONTRIBUTION, not PRICING_RATE).
        # If PRICING_RATE has premium-like values, move them to MIN_CONTRIBUTION.
        pricing = normalized.get("PRICING_RATE", "")
        if isinstance(pricing, str) and pricing not in ("N/A", ""):
            is_premium_rate = False
            
            # Pattern 1: Numeric tiers like "5000 | 10000"
            if re.search(r"^\d+(\s*\|\s*\d+)*$", pricing.strip()):
                is_premium_rate = True
            
            # Pattern 2: Tiered names with amounts like "Bronze: 5000 | Silver: 10000"
            elif re.search(r"(bronze|silver|gold|platinum).*\d+", pricing.lower()):
                is_premium_rate = True
            
            # Pattern 3: Percentage-based premiums like "2.75% of Sum Assured" or "2.75% | 1.50%"
            elif re.search(r"(\d+\.?\d*%.*?(?:sum assured|vehicle|value|insurance|assured))", pricing.lower()):
                is_premium_rate = True
            
            # Pattern 4: Multiple percentage rates separated by | like "2.75% | 1.50%"
            elif re.search(r"^\d+\.?\d*%(\s*\|\s*\d+\.?\d*%)*", pricing.strip()):
                is_premium_rate = True
            
            # Pattern 5: Rates mentioning "of Sum" or "of Value" (insurance premium structure)
            elif " of " in pricing.lower() and ("sum" in pricing.lower() or "value" in pricing.lower() or "vehicle" in pricing.lower()):
                is_premium_rate = True
            
            if is_premium_rate:
                # These look like insurance premium rates, not loan interest rates
                min_contrib = normalized.get("MIN_CONTRIBUTION", "N/A")
                if min_contrib == "N/A" or not min_contrib:
                    normalized["MIN_CONTRIBUTION"] = pricing
                # CRITICAL: For IBG products, PRICING_RATE MUST be N/A
                normalized["PRICING_RATE"] = "N/A"
        
        # CRITICAL FIX: Insurance REQUIRED_DOCUMENTS contamination check (rule AH8)
        # If REQUIRED_DOCUMENTS contains loan-specific keywords, it's hallucinated.
        # Return "N/A" since no explicit "Documentation Required" section was found
        # in the insurance document.
        req_docs = normalized.get("REQUIRED_DOCUMENTS", "")
        if isinstance(req_docs, str) and req_docs not in ("N/A", ""):
            req_docs = _clean_insurance_required_documents(req_docs)
            normalized["REQUIRED_DOCUMENTS"] = req_docs

    return normalized


def _fix_coverage_loan_amount_confusion(normalized: dict) -> dict:
    """
    CRITICAL FIX: Detects and corrects the common mistake of extracting
    insurance coverage amounts (sum insured) into LOAN_AMOUNT_RANGE instead
    of COVERAGE_AMOUNT.
    
    For IBG (insurance) products:
    - LOAN_AMOUNT_RANGE MUST be "N/A" always (it's loan-specific)
    - Coverage amounts (sum insured) MUST go in COVERAGE_AMOUNT
    
    This fixes the case where "Sum Insured up to PKR 5 million" gets
    extracted to LOAN_AMOUNT_RANGE instead of COVERAGE_AMOUNT.
    """
    lead = normalized.get("LEAD_MARKER", "").strip().upper()
    
    # Only validate for insurance products
    if lead != "IBG":
        return normalized
    
    loan_range = normalized.get("LOAN_AMOUNT_RANGE", "").strip()
    coverage = normalized.get("COVERAGE_AMOUNT", "").strip()
    
    # If LOAN_AMOUNT_RANGE has values for an insurance product, check if they're actually coverage amounts
    if loan_range and loan_range not in ("N/A", ""):
        # Check if the value looks like sum insured / coverage (currency amounts with PKR/million/etc)
        # Insurance coverage patterns typically have "million", "thousand", "K", "M", "PKR", etc.
        is_coverage_amount = any(
            keyword in loan_range.lower() 
            for keyword in ["sum insured", "coverage", "covered", "option", "million", "thousand"]
        )
        
        # If LOAN_AMOUNT_RANGE looks like coverage and COVERAGE_AMOUNT is empty, move it
        if is_coverage_amount and (not coverage or coverage == "N/A"):
            normalized["COVERAGE_AMOUNT"] = loan_range
            normalized["LOAN_AMOUNT_RANGE"] = "N/A"
        elif is_coverage_amount:
            # Both fields have values - LOAN_AMOUNT_RANGE should still be N/A for insurance
            normalized["LOAN_AMOUNT_RANGE"] = "N/A"
    
    # Final enforcement: IBG products MUST have LOAN_AMOUNT_RANGE = N/A
    if lead == "IBG" and normalized.get("LOAN_AMOUNT_RANGE") not in ("N/A", ""):
        # Additional check: does it really look like a loan amount or coverage?
        value = normalized.get("LOAN_AMOUNT_RANGE", "").lower()
        if "sum" in value or "coverage" in value or "option" in value:
            normalized["COVERAGE_AMOUNT"] = normalized.get("COVERAGE_AMOUNT", "N/A")
            if normalized["COVERAGE_AMOUNT"] == "N/A":
                normalized["COVERAGE_AMOUNT"] = normalized["LOAN_AMOUNT_RANGE"]
            normalized["LOAN_AMOUNT_RANGE"] = "N/A"
    
    return normalized




def _validate_product_name(product_name: str, doc_text: str) -> bool:
    """
    Validate that PRODUCT_NAME appears somewhere in the document text.
    Returns True if the name or a close variant is found, False if hallucinated.
    This prevents completely fictional product names from slipping through.
    """
    if not product_name or product_name == "N/A":
        return True  # N/A is valid
    
    # Normalize for comparison
    name_lower = product_name.lower().strip()
    text_lower = doc_text.lower()
    
    # Check if the product name appears anywhere in the document
    if name_lower in text_lower:
        return True
    
    # Check if at least the first 2 major words appear together in the document
    # This allows for slight variations (title case, etc.)
    words = name_lower.split()
    if len(words) >= 2:
        # Check if at least the first 2 words appear near each other
        first_two = f"{words[0]} {words[1]}"
        if first_two in text_lower:
            return True
    
    # If we get here, the name doesn't appear to be in the document
    return False


def _infer_and_correct_lead_marker(normalized: dict, doc_text: str = "") -> dict:
    """
    Smart inference and correction of LEAD_MARKER based on explicit content signals.
    Also validates PRODUCT_NAME against the document to catch hallucinations.
    
    Corrects common misclassifications:
    - A loan product (contains "term finance", "loan", "KIBOR", markup rates) 
      should be BNK, not IBG
    - An insurance product (contains "insurance", "policy", "premium", "coverage plan")
      should be IBG, not BNK
    
    Priority: Trust the extracted LEAD_MARKER FIRST (the model may have it right).
    Only correct if product description + field content contradict it.
    """
    if not isinstance(normalized, dict):
        return normalized
    
    desc = (normalized.get("PRODUCT_DESCRIPTION", "") or "").lower()
    plan = (normalized.get("PLAN_TYPE", "") or "").lower()
    prov = (normalized.get("PROVIDER_NAME", "") or "").lower()
    pricing = (normalized.get("PRICING_RATE", "") or "").lower()
    loan_amt = (normalized.get("LOAN_AMOUNT_RANGE", "") or "").lower()
    collateral = (normalized.get("COLLATERAL_TYPE", "") or "").lower()
    equity = (normalized.get("EQUITY_REQUIREMENT", "") or "").lower()
    
    # Signals that indicate a LOAN product (should be BNK)
    # CRITICAL: "term finance" is the strongest BNK signal — a term finance facility
    # is ALWAYS a bank loan product, never insurance, even if insurance is bundled
    loan_signals = {
        "term finance", "loan", "credit", "financing", "overdraft", 
        "markup", "kibor", "murabaha", "musharaka", "ijarah",
        "working capital", "auto", "housing", "vehicle", "sme",
        "business loan", "term facility", "credit facility", "green energy",
        "solar energy", "electricity generation", "renewable energy"
    }
    
    # Signals that indicate an INSURANCE product (should be IBG)
    insurance_signals = {
        "insurance", "protection", "takaful", "endowment",
        "unit-linked", "unit linked", "investment-linked", "cover",
        "policy", "premium", "rider", "hospitalization", "death benefit",
        "claims", "underwritten by"
    }
    
    combined_text = f"{desc} {plan} {prov} {pricing} {loan_amt} {collateral} {equity}".lower()
    
    loan_score = sum(1 for sig in loan_signals if sig in combined_text)
    insurance_score = sum(1 for sig in insurance_signals if sig in combined_text)
    
    current_marker = normalized.get("LEAD_MARKER", "").strip().upper()
    
    # CRITICAL: Check for "term finance" — STRONGEST loan signal, overrides everything
    # BUG FIX: Check BOTH extracted fields AND raw document text because fields may be truncated
    has_term_finance_in_fields = "term finance" in combined_text
    has_term_finance_in_doc = "term finance" in doc_text.lower() if doc_text else False
    has_term_finance = has_term_finance_in_fields or has_term_finance_in_doc
    has_financing = "financing" in combined_text or "financing" in desc
    
    # CRITICAL: For loans, if provider is Bank Alfalah and no insurance company is mentioned,
    # it's almost certainly a bank loan (BNK), not insurance (IBG)
    is_bank_only = ("bank alfalah" in prov or "bank " in prov) and "insurance" not in prov
    has_loan_keywords = any(sig in combined_text for sig in ["term finance", "loan", "financing", "green energy", "solar"])
    
    # BUG FIX: Also detect loan products from document keywords not in extracted fields
    # KIBOR, markup are strong loan indicators that might not appear in normalized fields
    doc_lower = doc_text.lower() if doc_text else ""
    has_kibor_or_markup = any(x in doc_lower for x in ["kibor", "markup", "profit rate", "interest rate"])
    
    # CRITICAL: "term finance facility" ALWAYS means BNK (bank loan), NEVER IBG,
    # even if insurance is bundled with it. The core product is a bank loan,
    # not an insurance product.
    if (has_term_finance or has_kibor_or_markup or (has_financing and is_bank_only)) and current_marker == "IBG":
        # Term finance facility or bank financing misclassified as insurance — MUST correct
        normalized["LEAD_MARKER"] = "BNK"
        # Cascade corrections: ALL insurance-only fields MUST be N/A for BNK products
        insurance_only_fields = [
            "COVERAGE_AMOUNT", "FREE_LOOK_PERIOD_DAYS", 
            "OPTIONAL_RIDERS", "PREMIUM_PAYMENT_FREQUENCY", 
            "MIN_CONTRIBUTION", "KEY_EXCLUSIONS", "CLAIMS_SERVICE_CONTACT"
        ]
        for field in insurance_only_fields:
            if field in normalized and normalized.get(field) not in ("N/A", "", None):
                normalized[field] = "N/A"
        
        # For BNK loan products, PLAN_TYPE should be "Loan"
        if normalized.get("PLAN_TYPE", "").lower() not in ("loan", "deposit", "account", "card"):
            normalized["PLAN_TYPE"] = "Loan"
        
        # Additional sanity check: if PRICING_RATE looks like insurance premiums, clear it for loans
        pricing = normalized.get("PRICING_RATE", "")
        if pricing and pricing != "N/A":
            if any(x in pricing.lower() for x in ["% of sum", "% of vehicle", "% net", "% insurance"]):
                # This looks like insurance premium rate in PRICING_RATE for a loan — clear it
                normalized["PRICING_RATE"] = "N/A"
    
    # Fallback: Strong loan signals with bank provider override weaker classification
    elif ((has_loan_keywords or has_kibor_or_markup or loan_score >= 1) and is_bank_only and current_marker == "IBG"):
        normalized["LEAD_MARKER"] = "BNK"
        # Cascade corrections for insurance-only fields per G12 rule
        for field in ["COVERAGE_AMOUNT", "FREE_LOOK_PERIOD_DAYS", 
                      "OPTIONAL_RIDERS", "PREMIUM_PAYMENT_FREQUENCY", 
                      "MIN_CONTRIBUTION", "KEY_EXCLUSIONS", "CLAIMS_SERVICE_CONTACT"]:
            if field in normalized:
                normalized[field] = "N/A"
        # For BNK loan products, PLAN_TYPE should be "Loan"
        if normalized.get("PLAN_TYPE", "").lower() != "loan":
            normalized["PLAN_TYPE"] = "Loan"
    
    elif loan_score >= 2 and current_marker == "IBG":
        # This is clearly a loan product but marked as insurance — correct it
        normalized["LEAD_MARKER"] = "BNK"
        # Cascade corrections for insurance-only fields per G12 rule
        for field in ["COVERAGE_AMOUNT", "FREE_LOOK_PERIOD_DAYS", 
                      "OPTIONAL_RIDERS", "PREMIUM_PAYMENT_FREQUENCY", 
                      "MIN_CONTRIBUTION", "KEY_EXCLUSIONS", "CLAIMS_SERVICE_CONTACT"]:
            if field in normalized:
                normalized[field] = "N/A"
    
    elif insurance_score >= 2 and current_marker == "BNK":
        # This is clearly an insurance product but marked as bank-only — correct it
        normalized["LEAD_MARKER"] = "IBG"
        # Cascade corrections for loan-only fields
        for field in ["LOAN_AMOUNT_RANGE", "COLLATERAL_TYPE", "EQUITY_REQUIREMENT", "DBR_LIMIT"]:
            if field in normalized:
                normalized[field] = "N/A"
    
    return normalized


def _normalize_employment_type(value: str) -> str:
    """
    Normalize EMPLOYMENT_TYPE to a short form.
    
    Common mistakes:
    - Extracting full target-market paragraphs (100+ chars)
    - Mixing employment types with other eligibility criteria
    
    Short forms allowed:
    - "Salaried" (or "Permanent", "Contractual" as modifiers)
    - "Self-Employed" (or "SEP", "SEB", "Proprietor")
    - "SME" (or "Business Owner")
    - Multiple types: "Salaried | Self-Employed"
    
    Strategy: If value is > 100 chars, it's probably a full paragraph — extract
    only the employment-type keywords from it.
    """
    if not value or value == "N/A":
        return "N/A"
    
    stripped = value.strip()
    if not stripped or len(stripped) == 0:
        return "N/A"
    
    # If the value is already short (< 60 chars), normalize the short forms
    if len(stripped) < 60:
        normalized_map = {
            "permanent": "Salaried",
            "salaried": "Salaried",
            "contractual": "Contractual",
            "self-employed": "Self-Employed",
            "self employed": "Self-Employed",
            "sep": "Self-Employed",
            "seb": "Self-Employed",
            "proprietor": "Self-Employed",
            "business": "SME",
            "sme": "SME",
            "corporate": "Corporate",
            "retail": "Retail",
            "government": "Government",
        }
        lower = stripped.lower()
        for key, norm in normalized_map.items():
            if key in lower:
                return norm
        return stripped
    
    # Value is >= 60 chars — likely a full paragraph
    # Extract employment-type keywords
    lower = stripped.lower()
    
    keywords = {
        "salaried": "Salaried",
        "permanent": "Salaried",
        "self-employed": "Self-Employed",
        "self employed": "Self-Employed",
        "sep": "Self-Employed",
        "business": "SME",
        "sme": "SME",
        "proprietor": "Self-Employed",
        "partnership": "Self-Employed",
        "corporate": "Corporate",
        "contractual": "Contractual",
    }
    
    found = []
    for key, norm in keywords.items():
        if key in lower and norm not in found:
            found.append(norm)
    
    if found:
        return " | ".join(found)
    
    # Fallback: truncate to 50 chars if still unrecognized
    return truncate_to_boundary(stripped, 50)


def normalize_record(record, entry, doc_text=""):
    """
    Normalize an extracted record with corrections for all known model errors.

    Key normalization rules applied here:
    - PRODUCT_NAME: ALL-CAPS converted to Title Case, validated against document
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
    
    Parameters:
    - record: extracted fields dict
    - entry: metadata (file info, etc)
    - doc_text: full document text for validation (optional)
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
        # EMPLOYMENT_TYPE normalization
        # ----------------------------------------------------------------
        if col == "EMPLOYMENT_TYPE" and isinstance(value, str):
            value = _normalize_employment_type(value)

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
        # Extract MIN_TERM_YEARS and MAX_TERM_YEARS from TENURE text if not provided
        # ----------------------------------------------------------------
        if col == "MIN_TERM_YEARS" and (value == DEFAULT_VALUE or not value):
            # Try to extract from TENURE if it exists
            tenure_value = record.get("TENURE") or normalized.get("TENURE")
            if tenure_value and tenure_value != DEFAULT_VALUE:
                min_y, max_y = _extract_tenure_years(tenure_value)
                if min_y is not None:
                    value = str(min_y)
        
        if col == "MAX_TERM_YEARS" and (value == DEFAULT_VALUE or not value):
            # Try to extract from TENURE if it exists
            tenure_value = record.get("TENURE") or normalized.get("TENURE")
            if tenure_value and tenure_value != DEFAULT_VALUE:
                min_y, max_y = _extract_tenure_years(tenure_value)
                if max_y is not None:
                    value = str(max_y)
        
        # ----------------------------------------------------------------
        # PREMIUM_PAYMENT_FREQUENCY normalization (insurance products only)
        # ----------------------------------------------------------------
        if col == "PREMIUM_PAYMENT_FREQUENCY" and isinstance(value, str):
            value = _normalize_premium_payment_frequency(value)

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
    
    # CRITICAL FIX: Validate product name against document to catch hallucinations
    if doc_text and isinstance(raw_name, str) and raw_name != "N/A":
        raw_name = _validate_and_fix_product_name(raw_name, doc_text)
    
    normalized["PRODUCT_NAME"] = raw_name
    normalized["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)

    # ----------------------------------------------------------------
    # Cross-field consistency: must run after all per-field normalizations
    # ----------------------------------------------------------------
    normalized = _crossfield_validate(normalized)
    
    # CRITICAL: Validate product variant tiers against document (prevent hallucination)
    if "PRODUCT_VARIANT_TIER" in normalized and doc_text:
        normalized["PRODUCT_VARIANT_TIER"] = _validate_product_variant_tier(
            normalized.get("PRODUCT_VARIANT_TIER", "N/A"), doc_text
        )
    
    # CRITICAL: Validate employment restrictions against document (prevent hallucination)
    if "EMPLOYMENT_TYPE" in normalized and doc_text:
        normalized["EMPLOYMENT_TYPE"] = _validate_employment_restrictions(
            normalized.get("EMPLOYMENT_TYPE", "N/A"), doc_text
        )
    
    # CRITICAL: Validate age requirements against document (prevent hallucination)
    if doc_text and ("MIN_AGE" in normalized or "MAX_AGE" in normalized):
        min_age, max_age = _validate_age_requirements(
            normalized.get("MIN_AGE", "N/A"),
            normalized.get("MAX_AGE", "N/A"),
            doc_text
        )
        normalized["MIN_AGE"] = min_age
        normalized["MAX_AGE"] = max_age
    
    # CRITICAL FIX (AH5a): Validate MIN_TERM_YEARS is only populated if explicitly stated
    # Prevent hallucination of default "1" when document only states "Up to X years"
    if doc_text and "MIN_TERM_YEARS" in normalized:
        min_term_value = normalized.get("MIN_TERM_YEARS", "N/A")
        max_term_value = normalized.get("MAX_TERM_YEARS", "N/A")
        validated_min_term = _validate_min_term_years(
            min_term_value,
            max_term_value,
            doc_text
        )
        normalized["MIN_TERM_YEARS"] = validated_min_term
    
    # CRITICAL FIX: Infer and correct LEAD_MARKER based on product signals
    # This catches loans incorrectly classified as insurance products
    # Pass the full document text for validation and product name checking
    normalized = _infer_and_correct_lead_marker(normalized, doc_text=doc_text)
    
    # CRITICAL FIX: Correct coverage amounts that were misplaced into LOAN_AMOUNT_RANGE
    normalized = _fix_coverage_loan_amount_confusion(normalized)
    
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


def build_repair_prompt(entry, raw_text, tokenizer=None):
    """
    Compact repair prompt for malformed JSON output.

    Intentionally does NOT include the full SYSTEM_PROMPT to avoid exceeding
    Qwen2.5-3B's context limit when the broken output is also long. The essential
    rules are inlined here instead.

    FIX (JSON recovery): Now accepts the tokenizer and applies the chat
    template so Qwen emits raw JSON instead of conversational preamble.
    The previous version sent a raw string which caused Qwen to prefix its
    output with "Here is the repaired JSON …" text that broke parsing.
    """
    # FIX: Truncate broken output more aggressively to save prompt tokens.
    # 2000 chars is enough context for repair; 3000 was wasting budget.
    repair_instructions = f"""You are repairing broken JSON from a data extraction task.
Product: {entry['title']}
Source file: {get_source_filename(entry)}

BROKEN OUTPUT TO REPAIR:
{raw_text[:2000]}

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

    # FIX: Apply chat template so Qwen generates raw JSON instead of
    # conversational preamble that breaks the parser.
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": repair_instructions},
        ]
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
            pass  # fall through to raw string

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

28. MIN_TERM_YEARS CRITICAL: Only extract if document EXPLICITLY states a
    MINIMUM tenure/term. Do NOT extract from "Up to X years" or "Maximum X years"
    statements — those only give MAX_TERM_YEARS. If minimum is not stated in
    the document, return "N/A" (never default to 1 or assume). Example:
    WRONG: Document says "Up to 10 years" → MIN_TERM_YEARS="1" (guessed)
    CORRECT: Document says "Up to 10 years" → MIN_TERM_YEARS="N/A" (not stated)
    
29. LOAN_AMOUNT_RANGE, COLLATERAL_TYPE, EQUITY_REQUIREMENT for BNK products:
    Must extract EXPLICITLY stated values only. Examples:
    LOAN_AMOUNT_RANGE: "Up to PKR 5 Million" or "Between 1M-10M"
    COLLATERAL_TYPE: "Residential Property" or "Commercial Property"
    EQUITY_REQUIREMENT: "20%" (strip verbose prefixes like "Minimum 20%")
    If not stated, return "N/A".

30. PREMIUM_PAYMENT_FREQUENCY for insurance: "yearly renewable plan" maps to
    "Annual" payment frequency. Must EXPLICITLY extract from document phrases
    like "annual", "yearly", "monthly", etc. Do NOT default to "Annual".

31. REQUIRED_DOCUMENTS for insurance (IBG): Extract ONLY from explicit
    "Documentation Required" or "Required Documents" section. If document shows
    only "Claim Processing" steps or no upfront documentation section, return
    "N/A" immediately (do NOT hallucinate or copy from loan sections).

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


def _gpu_free_gib() -> float:
    """Return free GPU memory in GiB, or inf if CUDA is not available.

    Used to decide at runtime whether the validation LLM pass can safely
    run without risking OOM (its prompt is the largest in the pipeline).
    """
    if not torch.cuda.is_available():
        return float("inf")
    free_bytes = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    return free_bytes / (1024 ** 3)


def get_raw_generation(model, tokenizer, prompt, max_new_tokens=None):
    effective_max_new_tokens = max_new_tokens or MAX_NEW_TOKENS

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]

    # FIX (OOM): Log prompt size so users can diagnose which call is the
    # memory hog without needing a debugger.  Also cap total context to the
    # model's max position embeddings minus a small safety margin.
    model_max_len = getattr(model.config, "max_position_embeddings", 32768)
    total_len = input_len + effective_max_new_tokens
    if total_len > model_max_len:
        trimmed = max(400, model_max_len - input_len)
        print(
            f"    note: prompt ({input_len} tok) + max_new_tokens ({effective_max_new_tokens}) "
            f"= {total_len} exceeds model context ({model_max_len}). "
            f"Capping max_new_tokens to {trimmed}."
        )
        effective_max_new_tokens = trimmed

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
        print(
            f"    note: generation hit max_new_tokens={effective_max_new_tokens} "
            f"(output likely truncated, not malformed) — attempting auto-close recovery"
        )

    # Explicitly drop tensor references to free GPU memory before next call.
    del inputs, output_ids, generated_ids
    free_gpu_memory()

    return text


def extract_one_chunk(model, tokenizer, entry, chunk, chunk_idx, chunk_total, max_new_tokens=None, full_text=""):
    """
    Run extraction on a SINGLE chunk of the document (initial attempt +
    one compact repair retry if output is not parseable JSON).
    Validation is done once on the final merged record, not per chunk.
    Returns a normalized record dict, or None if both attempts failed.
    
    Parameters:
    - full_text: the complete document text (for validation purposes)
    """
    prompt = build_prompt(entry, chunk, tokenizer, chunk_idx, chunk_total)

    # FIX (OOM): Wrap individual generate() calls so a single OOM during
    # one chunk doesn't abort the entire product.  The caller's retry loop
    # then only re-runs the failing chunk, not all previous successful ones.
    try:
        raw = get_raw_generation(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        free_gpu_memory()
        print(
            f"  Warning: chunk {chunk_idx}/{chunk_total} OOM during extraction, "
            f"skipping this chunk"
        )
        return None

    try:
        parsed = parse_json_blob(raw)
        return normalize_record(parsed, entry, doc_text=full_text)
    except Exception:
        # Compact repair prompt — now uses chat template (FIX).
        repair_prompt = build_repair_prompt(entry, raw, tokenizer=tokenizer)
        try:
            repaired_raw = get_raw_generation(model, tokenizer, repair_prompt, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            free_gpu_memory()
            print(
                f"  Warning: chunk {chunk_idx}/{chunk_total} OOM during repair, "
                f"skipping this chunk"
            )
            return None
        try:
            repaired_parsed = parse_json_blob(repaired_raw)
            return normalize_record(repaired_parsed, entry, doc_text=full_text)
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
            model, tokenizer, entry, chunk, i, chunk_total, max_new_tokens=max_new_tokens,
            full_text=text
        )
        if chunk_record is not None:
            any_chunk_succeeded = True
            accumulated = merge_records(accumulated, chunk_record, COLUMNS)
        free_gpu_memory()

    if not any_chunk_succeeded:
        return blank_record(entry)

    # Single validation/correction pass on the merged record.
    # FIX (OOM): The validation prompt is the LARGEST in the pipeline
    # (full SYSTEM_PROMPT + 31 rules + extracted JSON ≈ 10,500 tokens
    # input).  On Colab-tier GPUs (≤16 GB) this is the most common OOM
    # trigger.  Skip it when free VRAM is below 3 GiB — the Python-side
    # normalize_record + _crossfield_validate already enforce the same
    # rules deterministically, so extraction quality is preserved.
    free_gib = _gpu_free_gib()
    if free_gib < 3.0:
        print(
            f"    note: skipping LLM validation pass (only {free_gib:.1f} GiB free, "
            f"need ~3 GiB) — Python-side normalization still applied"
        )
        return accumulated

    # Use the same max_new_tokens as the rest of the pipeline (single source
    # of truth).  A previous version hardcoded 1200 here, which silently
    # overrode the auto-raised MAX_NEW_TOKENS (1500) and caused inconsistent
    # "generation hit max_new_tokens" log values.
    validation_max = max_new_tokens or MAX_NEW_TOKENS
    validation_prompt = build_validation_prompt(entry, accumulated)
    try:
        validation_raw = get_raw_generation(model, tokenizer, validation_prompt, max_new_tokens=validation_max)
    except torch.cuda.OutOfMemoryError:
        free_gpu_memory()
        print(
            "    note: validation pass OOM — returning merged record "
            "(Python-side normalization still applied)"
        )
        return accumulated
    try:
        validated_parsed = parse_json_blob(validation_raw)
        return normalize_record(validated_parsed, entry, doc_text=text)
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
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_gib = total_gib - allocated_gib
        print(
            f"Post-load GPU memory: {allocated_gib:.2f} GiB allocated, "
            f"{reserved_gib:.2f} GiB reserved, "
            f"{free_gib:.2f} GiB free out of {total_gib:.2f} GiB total"
        )
        # FIX (diagnostics): Loud warning when the model is using far more
        # VRAM than expected for the claimed quantization level.  This is
        # the single most common misconfiguration — the .env has
        # HF_LOAD_IN_4BIT=false (default) so weights load in fp16.
        if LOAD_IN_4BIT and allocated_gib > 4.0:
            print(
                f"\n{'='*70}\n"
                f"WARNING: 4-bit quantization is ENABLED but the model is using "
                f"{allocated_gib:.1f} GiB — this is too high for a 4-bit 3B model\n"
                f"(expected ~2 GiB). Quantization may not be applied correctly.\n"
                f"Check that bitsandbytes is installed and working:\n"
                f"  pip install -U 'bitsandbytes>=0.46.1'\n"
                f"{'='*70}\n"
            )
        elif not LOAD_IN_4BIT and not LOAD_IN_8BIT and allocated_gib > 5.0:
            print(
                f"\n{'='*70}\n"
                f"WARNING: Model loaded in fp16 and using {allocated_gib:.1f} GiB.\n"
                f"On a {total_gib:.1f} GiB GPU this leaves only {free_gib:.1f} GiB\n"
                f"for KV-cache + generation — OOM during generate() is very likely.\n"
                f"\nSTRONGLY RECOMMENDED: Enable 4-bit quantization in your .env:\n"
                f"  HF_LOAD_IN_4BIT=true\n"
                f"This will reduce model memory from ~{allocated_gib:.1f} GiB to ~2 GiB\n"
                f"and eliminate most OOM errors.\n"
                f"{'='*70}\n"
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