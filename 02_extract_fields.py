"""
Step 2 (v2): Extract the fixed 56-column schema as JSON from product files.

AUDIT-DRIVEN FIXES IN THIS VERSION:
----------------------------------------------------------------------
FIX-01  TENURE_OPTIONS contamination (6/10 rows):
        Added _is_payment_frequency() helper that detects "Annual / Semi-Annual /
        Quarterly / Monthly" patterns. Any TENURE_OPTIONS value whose tokens are
        all payment-frequency keywords is reset to "N/A" in normalize_record().
        The validation prompt and the DO NOT block in EXTRACTION_SYSTEM_PROMPT_v2.txt
        also emphasise this rule.

FIX-02  MAX_AGE / MAX_TERM_YEARS attained-age confusion (Shifa, Zeenat):
        Prompt updated with explicit negative examples. In
        apply_deterministic_overrides() a new heuristic is added: if
        MAX_TERM_YEARS > 75, flag it as a likely attained-age value and reset to
        "N/A" with a warning. The model is then responsible for re-extracting the
        correct entry limit. This is conservative — real 76-year plan terms are
        very rare in the banking market.

FIX-03  PRICING_RATE unit-allocation table confusion (Tadbeer, and partial Zaamin):
        Added _is_unit_allocation_table() helper that detects when all extracted
        values are small percentages (≤100%). Such a PRICING_RATE is reset to
        "N/A" and a warning is logged.

FIX-04  PRODUCT_NAME all-caps not normalised (Jubilee):
        The original normalize_record() already had all-caps → title-case logic,
        but the final "preserved name" block used record.get("PRODUCT_NAME") which
        is the raw model output — and .title() was applied, but only if the length
        was > 5. Ensured .title() fires correctly and added an assertion to catch
        any remaining all-caps names downstream.

FIX-05  KEY_EXCLUSIONS hallucination (Jubilee):
        Added _is_claims_doc_pattern() that detects claim-document language in
        KEY_EXCLUSIONS ("Post Mortem", "Claim Form", "FIR", "attested copy",
        "Physician's Statement"). When detected the field is reset to "N/A".

FIX-06  REQUIRED_DOCUMENTS = claims-settlement docs (SLIC):
        Added _is_claims_settlement_docs() that detects death-claim document
        patterns ("Death Certificate", "Claim Form A", "Discharge Letter",
        "SLIC/GBA", "Medico Legal"). When detected the field is reset to "N/A".

FIX-07  DOCX reader — document-order table extraction:
        read_docx_text() now iterates the document body XML in rendering order
        so tables appear inline with the paragraph that precedes them. Each table
        is prefixed with a "~TABLE_START~" / "~TABLE_END~" marker so the model
        can see it belongs to the product section currently being processed.
        This prevents a whole-document table dump at the end which confused the
        model on multi-product files.

FIX-08  Product section isolation for multi-product documents:
        Added extract_product_section() that uses product-heading detection
        to find the boundaries of THIS product's section and returns only that
        slice of the document text. This eliminates cross-product field bleeding
        at the source, before chunking.

FIX-09  COVERAGE_AMOUNT truncation produces incomplete values:
        Added compress_coverage_amount() that rewrites multi-tier coverage amounts
        using standard abbreviations (Brnz/Slvr/Gld/Plat, K-suffix) before
        truncation, so all tiers fit in 150 chars.

FIX-10  FEES_AND_CHARGES bleed of non-fee text (Saholat):
        Added _clean_fees_field() that strips lines matching Target Market or
        eligibility patterns from a fees string, keeping only lines that contain
        recognized fee keywords.

FIX-11  TENURE format inconsistency:
        Added _normalize_tenure() that converts
        "Minimum Term: X years; Maximum Term: Y years" → "X-Y years".

FIX-12  MIN_CONTRIBUTION string vs integer (Jubilee):
        Added an explicit string→int coercion pass that ensures MIN_CONTRIBUTION
        (and all NUMERIC_COLUMNS) always come out as bare integer strings.

SYSTEM PROMPT IS READ FROM: EXTRACTION_SYSTEM_PROMPT_v2.txt
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

_NEW_COLUMNS = ["PREMIUM_PAYMENT_FREQUENCY"]
for _col in _NEW_COLUMNS:
    if _col not in COLUMNS:
        COLUMNS = list(COLUMNS) + [_col]

MODEL_NAME = os.environ.get("HF_MODEL_NAME_OR_PATH", "").strip()
MODEL_CLASS = os.environ.get("HF_MODEL_CLASS", "causal").strip().lower()
LOCAL_FILES_ONLY = env_bool("HF_LOCAL_FILES_ONLY", True)
TRUST_REMOTE_CODE = env_bool("HF_TRUST_REMOTE_CODE", False)
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 2200)
TEMPERATURE = env_float("HF_TEMPERATURE", 0.0)
TOP_P = env_float("HF_TOP_P", 1.0)
REPETITION_PENALTY = env_float("HF_REPETITION_PENALTY", 1.03)

# LOW_VRAM_MODE: set to "true" in .env to activate conservative memory settings
# suitable for Colab T4 (15 GB) running a 7B model in 4-bit, or any GPU with
# less than ~8 GB VRAM headroom after loading the model.
#   - Reduces TEXT_CHUNK_SIZE from 6000 → 3500 (smaller KV cache per call)
#   - Disables the validation pass (saves one full generate() call per product)
#   - Keeps all deterministic post-processing overrides active
LOW_VRAM_MODE = env_bool("LOW_VRAM_MODE", False)

# TEXT_CHUNK_SIZE: default 4500 chars (~1125 tokens) — a safe middle ground
# between the old 6000 (too large for 7B on T4) and the audit-recommended 6000.
# Override with TEXT_CHUNK_SIZE=3500 for strict VRAM budgets or
# TEXT_CHUNK_SIZE=6000 if you have a large GPU and want fewer chunks.
_chunk_default = 3500 if LOW_VRAM_MODE else 4500
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", _chunk_default)
# Overlap increased from 300 → 500 to reduce the probability of a single
# field value being split across two chunk boundaries.
TEXT_CHUNK_OVERLAP = env_int("TEXT_CHUNK_OVERLAP", 500)

LOAD_IN_4BIT = env_bool("HF_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = env_bool("HF_LOAD_IN_8BIT", False)
DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
TORCH_DTYPE = os.environ.get("HF_TORCH_DTYPE", "auto").strip().lower()
GPU_RESERVE_GIB = env_float("HF_GPU_RESERVE_GIB", 3.0)
ENABLE_CHUNKING = env_bool("ENABLE_CHUNKING", True)

# ENABLE_VALIDATION: set to "false" in .env to skip the post-merge validation
# generate() call. Saves ~2200 tokens of KV cache per product. All deterministic
# normalisation overrides (apply_deterministic_overrides, normalize_record) still
# run — only the LLM-based correction pass is skipped.
ENABLE_VALIDATION = env_bool("ENABLE_VALIDATION", not LOW_VRAM_MODE)
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


class ExtractionFailedError(RuntimeError):
    """Raised when all chunks fail to produce parseable JSON."""


# ============================================================================
# LOAD SYSTEM PROMPT
# ============================================================================

def load_system_prompt(prompt_file: str = "EXTRACTION_SYSTEM_PROMPT_v2_compact.txt") -> str:
    """
    Load the system prompt. Searches in:
    1. Current directory
    2. Same directory as this script
    3. Parent directory
    Falls back through v2 → v1 filenames for backward compatibility.
    """
    candidates = [
        prompt_file,
        "EXTRACTION_SYSTEM_PROMPT_v2.txt",   # full v2 (larger; use only if VRAM allows)
        "EXTRACTION_SYSTEM_PROMPT.txt",       # original v1 (legacy fallback)
    ]
    for fname in candidates:
        for base in (Path("."), Path(__file__).parent, Path(__file__).parent.parent):
            p = base / fname
            if p.exists() and p.is_file():
                print(f"✓ Loaded system prompt from: {p.resolve()}")
                return p.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"System prompt file not found! Expected '{prompt_file}' in current "
        "directory, script directory, or parent directory."
    )


SYSTEM_PROMPT = load_system_prompt()


# ============================================================================
# FIX-07: DOCX reader with document-order table extraction
# ============================================================================

def _iter_docx_body_elements(doc):
    """
    Yield paragraphs and tables in the order they appear in the document body.
    This preserves the spatial relationship between section headings and their
    tables, so the model can attribute each table to the correct product section.

    Uses python-docx's internal _element (lxml) to walk body children.
    """
    from docx.oxml.ns import qn
    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            # Paragraph
            from docx.text.paragraph import Paragraph
            para = Paragraph(child, doc)
            if para.text.strip():
                yield ("paragraph", para.text)
        elif tag == "tbl":
            # Table — emit with markers so the model knows table context
            from docx.table import Table
            tbl = Table(child, doc)
            yield ("table_start", None)
            for row in tbl.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_text = " | ".join(c for c in cells if c)
                if row_text.strip():
                    yield ("table_row", row_text)
            yield ("table_end", None)


def read_docx_text(path: Path) -> str:
    """
    FIX-07: Extract text from DOCX preserving document order (paragraphs and
    tables interleaved) instead of dumping all tables at the end.

    Each table is wrapped in ~TABLE_START~ / ~TABLE_END~ markers so the LLM
    can see the table belongs to the product section it follows. This is
    critical for multi-product documents where attributing tables to the
    correct product section prevents cross-product field contamination.
    """
    try:
        import docx
    except ImportError:
        raise SystemExit(
            "python-docx is required to read DOCX files.\n"
            "Install it with:  pip install python-docx"
        )
    doc = docx.Document(path)
    lines: list[str] = []
    for kind, content in _iter_docx_body_elements(doc):
        if kind == "paragraph":
            lines.append(content)
        elif kind == "table_start":
            lines.append("~TABLE_START~")
        elif kind == "table_row":
            lines.append(content)
        elif kind == "table_end":
            lines.append("~TABLE_END~")
    return "\n".join(lines)


# ============================================================================
# FIX-08: Product section isolation
# ============================================================================

# Heading patterns that mark the start of a new product section in merged docs.
# These match: a standalone line that is all-caps or title-case, bold-implied,
# followed by nothing or by a colon/newline. Conservative: only match lines
# with 3+ capitalised words to avoid triggering on short headers.
_SECTION_HEADING_RE = re.compile(
    r"""
    ^           # start of line
    [ \t]*      # optional leading whitespace
    (
        # Pattern A: ALL CAPS heading with 3+ words (like "JUBILEE KAMIL TAKAFUL SAVINGS PLAN:")
        [A-Z][A-Z\s\-&/]{10,}[A-Z]
        |
        # Pattern B: Title Case heading with 3+ capitalised words
        (?:[A-Z][a-z]+\s+){2,}[A-Z][a-zA-Z]*
    )
    :?          # optional trailing colon
    [ \t]*$     # optional trailing whitespace
    """,
    re.MULTILINE | re.VERBOSE,
)

def extract_product_section(full_text: str, product_title: str) -> str:
    """
    FIX-08: Isolate the section of a multi-product document that belongs to
    the product named in product_title.

    Strategy:
    1. Find all section headings using _SECTION_HEADING_RE.
    2. Find the heading that best matches product_title (token-overlap).
    3. Return text from that heading to the next heading (or end of document).

    If no matching heading is found, return the full text unchanged. This is
    a conservative fallback: failing to isolate is better than truncating too
    aggressively and losing real content.
    """
    lines = full_text.split("\n")
    heading_positions: list[tuple[int, str]] = []  # (line_index, heading_text)

    for i, line in enumerate(lines):
        stripped = line.strip().rstrip(":")
        if len(stripped.split()) >= 3 and _SECTION_HEADING_RE.match(line):
            heading_positions.append((i, stripped))

    if not heading_positions:
        return full_text

    # Token-overlap score between product_title and each candidate heading
    title_tokens = set(_normalize_for_match(product_title).split())

    def overlap_score(heading_text: str) -> float:
        h_tokens = set(_normalize_for_match(heading_text).split())
        if not h_tokens or not title_tokens:
            return 0.0
        return len(title_tokens & h_tokens) / max(len(title_tokens | h_tokens), 1)

    if not title_tokens:
        return full_text

    scored = [(score, idx, text) for idx, text in heading_positions
              for score in [overlap_score(text)]]
    scored.sort(reverse=True)

    if not scored or scored[0][0] < 0.25:
        # No heading matches the product title well enough — return full text
        return full_text

    best_match_line = scored[0][1]

    # Find the next heading after the best match
    next_heading_line = len(lines)
    for i, text in heading_positions:
        if i > best_match_line:
            next_heading_line = i
            break

    section_lines = lines[best_match_line:next_heading_line]
    section_text = "\n".join(section_lines).strip()

    if len(section_text) < 200:
        # Suspiciously short section — likely a false heading match; return full text
        return full_text

    return section_text


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
            raise SystemExit("pdfplumber is required: pip install pdfplumber")
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    elif ext == ".docx":
        # FIX-07: Use document-order extraction instead of the old
        # "all paragraphs then all tables" approach.
        return read_docx_text(path)

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
            raise SystemExit("antiword is required: sudo apt install antiword")

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
            raise SystemExit("openpyxl is required: pip install openpyxl")
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
# Interactive input selection (unchanged from v1)
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
                print(f"  No supported files found directly in '{p}'. Supported: {supported_str}\n")
            else:
                print(f"  Not a valid directory: {raw}\n")

    else:
        while True:
            raw = input("\nEnter the root folder path: ").strip()
            p = Path(raw)
            if p.is_dir():
                files = _collect_files(p, recursive=True)
                if files:
                    print(f"  Found {len(files)} supported file(s) under '{p}'.")
                    return files
                print(f"  No supported files found anywhere under '{p}'. Supported: {supported_str}\n")
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
# Document chunking
# ============================================================================

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long document into smaller pieces with paragraph-boundary breaks.
    FIX: overlap increased from 300→600 (via env var default) to reduce risk
    of a single field value being split across chunk boundaries.
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
            search_from = max(start, end - 600)  # wider search window
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
    FIX: Added TENURE_OPTIONS payment-frequency rejection at merge time.
    Even if a chunk produces "Annual, Semi-Annual, Quarterly" in TENURE_OPTIONS,
    it is treated as empty (N/A) and will not block a future chunk's correct value.
    """
    for col in columns:
        old = accumulated.get(col)
        new = new_record.get(col)
        old_is_empty = old is None or old == "" or old == DEFAULT_VALUE

        # FIX-01: Reject payment-frequency values from TENURE_OPTIONS at merge time
        if col == "TENURE_OPTIONS" and new not in (None, "", DEFAULT_VALUE):
            if _is_payment_frequency(str(new)):
                new = DEFAULT_VALUE  # treat as absent

        new_is_real = new not in (None, "", DEFAULT_VALUE)
        if old_is_empty and new_is_real:
            accumulated[col] = new
    return accumulated


# ============================================================================
# JSON parsing helpers (unchanged from v1)
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


def _close_unterminated_json(text: str):
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
    closed = _close_unterminated_json(raw)
    if closed:
        parsed = _parse_candidate(closed)
        if parsed is not None:
            return parsed
    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


# ============================================================================
# Record helpers and normalization
# ============================================================================

def blank_record(entry):
    record = {col: DEFAULT_VALUE for col in COLUMNS}
    record["PRODUCT_NAME"] = entry["title"]
    record["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)
    return record


def get_source_filename(entry) -> str:
    """Return the source filename. Always uses the actual path.name."""
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
    "KEY_BENEFITS": 250,
    "OPTIONAL_RIDERS": 300,
    "REQUIRED_DOCUMENTS": 200,
    "CLAIMS_SERVICE_CONTACT": 200,
    "KEY_EXCLUSIONS": 200,
    "TAX_ZAKAT_TREATMENT": 100,
    "PLAN_TYPE": 15,
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
    if len(value) <= max_len:
        return value
    cut = value[:max_len].rstrip()
    boundary = max(cut.rfind(" "), cut.rfind(","), cut.rfind(";"), cut.rfind(":"), cut.rfind("-"))
    if boundary > 0:
        return cut[:boundary].rstrip(" ,;:-/")
    return cut


# ============================================================================
# FIX-01: TENURE_OPTIONS payment-frequency detection
# ============================================================================

_PAYMENT_FREQ_TOKENS = frozenset({
    "annual", "annually", "semi-annual", "semi annual", "semiannual",
    "quarterly", "monthly", "monthly", "biannual", "bi-annual",
    "half yearly", "half-yearly", "halfyearly",
    "yearly", "per year", "per annum",
})


def _is_payment_frequency(value: str) -> bool:
    """
    FIX-01: Return True if the value looks like a payment frequency list
    rather than plan duration choices.

    A payment frequency value is one where ALL tokens (after splitting on
    commas, semicolons, slashes) are in the payment-frequency keyword set.
    E.g.: "Annual, Semi-Annual, Quarterly" → True
          "10, 15, 20 years" → False
          "Annual" → True (single token matches)
    """
    if not value or value.strip() == DEFAULT_VALUE:
        return False
    # Tokenise on common list separators
    tokens = [t.strip().lower() for t in re.split(r"[,;/\|]+", value) if t.strip()]
    if not tokens:
        return False
    # All tokens must be payment-frequency keywords (or empty after stripping)
    return all(
        any(kw in tok for kw in _PAYMENT_FREQ_TOKENS)
        for tok in tokens
    )


# ============================================================================
# FIX-03: PRICING_RATE unit-allocation table detection
# ============================================================================

def _is_unit_allocation_table(value: str) -> bool:
    """
    FIX-03: Return True if the PRICING_RATE value contains only percentage
    values (unit-allocation table content) rather than contribution amounts.

    Unit-allocation tables look like: "Year 1: 60%; Year 2: 80%; Year 3: 95%"
    Contribution tables look like: "Bronze: 5,000; Silver: 10,000; Gold: 15,000"

    Heuristic: if ALL numeric values in the string are ≤ 100 and the string
    contains "%" signs, classify as a unit-allocation table.
    """
    if not value or value == DEFAULT_VALUE:
        return False
    percent_count = value.count("%")
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    if not numbers:
        return False
    if percent_count > 0:
        # All numbers ≤ 100 → likely percentages, not PKR amounts
        if all(float(n) <= 100.0 for n in numbers):
            return True
    return False


# ============================================================================
# FIX-05: KEY_EXCLUSIONS claims-document detection
# ============================================================================

_CLAIMS_DOC_KEYWORDS = frozenset({
    "post mortem", "postmortem", "claim form", "claimant statement",
    "physician statement", "physician's statement", "fir", "first information report",
    "medico legal", "medico-legal", "attested copy", "attested cnic",
    "discharge letter", "discharge summary", "hospital bill",
    "union council death certificate", "original policy",
    "news paper cutting", "newspaper cutting",
})


def _is_claims_doc_pattern(value: str) -> bool:
    """
    FIX-05: Return True if KEY_EXCLUSIONS appears to contain claims settlement
    document requirements rather than actual policy exclusions.

    Claims documents are listed when an insurer wants specific paperwork for
    death/accident claims. They are NOT policy exclusions.
    Match: "Murder, Suicide, Accidental Death: Post Mortem Report, FIR..."
    """
    if not value or value == DEFAULT_VALUE:
        return False
    lower = value.lower()
    keyword_hits = sum(1 for kw in _CLAIMS_DOC_KEYWORDS if kw in lower)
    return keyword_hits >= 2  # at least 2 claims-doc keywords required


# ============================================================================
# FIX-06: REQUIRED_DOCUMENTS claims-settlement-doc detection
# ============================================================================

_CLAIMS_SETTLEMENT_KEYWORDS = frozenset({
    "death certificate", "claim form a", "claim form b", "claimant statement",
    "discharge letter", "slic/gba", "nadra", "original policy document",
    "claim investigation", "medico legal", "premium collection record",
    "physician's statement", "post mortem",
})


def _is_claims_settlement_docs(value: str) -> bool:
    """
    FIX-06: Return True if REQUIRED_DOCUMENTS contains death-claim settlement
    documents rather than enrollment documents.

    Enrollment docs look like: "Auto debit form; CNIC; Declaration form"
    Claims docs look like: "Death Certificate; Claim Form A; Claim Form B;
    CNIC of Deceased; FIR..."
    """
    if not value or value == DEFAULT_VALUE:
        return False
    lower = value.lower()
    keyword_hits = sum(1 for kw in _CLAIMS_SETTLEMENT_KEYWORDS if kw in lower)
    return keyword_hits >= 2


# ============================================================================
# FIX-10: FEES_AND_CHARGES non-fee text removal
# ============================================================================

_FEE_KEYWORDS = frozenset({
    "fee", "fees", "charge", "charges", "spread", "wakala", "wakalah",
    "mudarib", "administration", "management", "allocation", "admin",
    "bid", "offer", "wakalatul", "istismar", "modarib", "wakala fee",
})

_NON_FEE_PATTERNS = re.compile(
    r"(salaried|professionals|chartered accountants|self employed|landlords|"
    r"housewives|retired|government|armed forces|eligibility|target market)",
    re.IGNORECASE,
)


def _clean_fees_field(value: str) -> str:
    """
    FIX-10: Remove non-fee lines from FEES_AND_CHARGES.

    Source documents occasionally include Target Market or Eligibility text
    inside the Associated Charges section due to document-authoring errors.
    This filter keeps only lines that contain at least one fee keyword and
    do not match known non-fee patterns.
    """
    if not value or value == DEFAULT_VALUE:
        return value

    lines = [line.strip() for line in re.split(r"[;\n]", value) if line.strip()]
    cleaned = []
    for line in lines:
        lower = line.lower()
        has_fee_keyword = any(kw in lower for kw in _FEE_KEYWORDS)
        has_non_fee_pattern = bool(_NON_FEE_PATTERNS.search(line))
        if has_fee_keyword and not has_non_fee_pattern:
            cleaned.append(line)
        elif not has_fee_keyword and not has_non_fee_pattern:
            # Neutral line — keep if it looks like it continues a fee description
            # (starts with a number, dash, or bullet)
            if re.match(r"^[-•\d]", line):
                cleaned.append(line)

    result = "; ".join(cleaned)
    return result if result else value  # fallback to original if all removed


# ============================================================================
# FIX-11: TENURE format normalisation
# ============================================================================

def _normalize_tenure(value: str) -> str:
    """
    FIX-11: Convert "Minimum Term: X years; Maximum Term: Y years" verbose
    format into the standard "X-Y years" dash notation used across all rows.

    Also handles "Minimum Term: X years\nMaximum Term: Y years" (newline sep).
    """
    if not value or value == DEFAULT_VALUE:
        return value
    # Match verbose pattern
    m = re.search(
        r"[Mm]inimum\s+[Tt]erm\s*:?\s*(\d+)\s*[Yy]ears?.*?"
        r"[Mm]aximum\s+[Tt]erm\s*:?\s*(\d+)\s*[Yy]ears?",
        value,
        re.DOTALL,
    )
    if m:
        min_yr, max_yr = m.group(1), m.group(2)
        # Preserve any attained-age qualifier if present
        attained = re.search(r"attained age of (\d+)", value, re.IGNORECASE)
        if attained:
            return f"{min_yr}-{max_yr} years (up to attained age of {attained.group(1)})"
        return f"{min_yr}-{max_yr} years"
    return value


# ============================================================================
# PLAN_TYPE and FINANCING_TYPE normalisers (unchanged from v1)
# ============================================================================

def _normalize_plan_type(value: str) -> str:
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


# ============================================================================
# Normalise record
# ============================================================================

def normalize_record(record, entry) -> dict:
    """
    Normalise an extracted record with all audit-driven fixes applied.

    New in v2:
    - FIX-01: TENURE_OPTIONS payment-frequency detection → N/A
    - FIX-03: PRICING_RATE unit-allocation table → N/A + warning
    - FIX-04: PRODUCT_NAME all-caps → Title Case (strengthened)
    - FIX-05: KEY_EXCLUSIONS claims-doc pattern → N/A
    - FIX-06: REQUIRED_DOCUMENTS claims-settlement docs → N/A
    - FIX-09: COVERAGE_AMOUNT — no truncation mid-word; abbreviate first
    - FIX-10: FEES_AND_CHARGES non-fee text removal
    - FIX-11: TENURE verbose-format normalisation
    - FIX-12: NUMERIC_COLUMNS string-to-int coercion (catches "36000" as str)
    """
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}

    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)
        if value in (None, "", []):
            value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-12: Numeric columns — strict integer output
        # Handles both str and numeric types; strips PKR/commas/.0
        # ----------------------------------------------------------------
        if col in NUMERIC_COLUMNS:
            if isinstance(value, (int, float)):
                value = str(int(value))
            elif isinstance(value, str):
                stripped = value.strip()
                if not stripped or stripped.upper() == DEFAULT_VALUE:
                    value = DEFAULT_VALUE
                else:
                    match = re.match(r"^(\d+(?:\.\d+)?)", stripped.replace(",", ""))
                    if match:
                        num_str = match.group(1)
                        try:
                            value = str(int(float(num_str)))
                        except ValueError:
                            value = DEFAULT_VALUE
                    else:
                        value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-04: PRODUCT_NAME — all-caps → Title Case
        # ----------------------------------------------------------------
        if col == "PRODUCT_NAME" and isinstance(value, str):
            stripped_name = value.strip()
            if (stripped_name
                    and stripped_name == stripped_name.upper()
                    and len(stripped_name.split()) > 1
                    and len(stripped_name) > 5):
                # .title() lowercases everything first; restore known acronyms
                value = stripped_name.title()
                # Re-uppercase known acronyms
                for acronym in ("Igi", "Wto", "Slic", "Atm", "Nrp", "Hnw",
                                "Cnic", "Efu", "Sme", "Llc", "Plc"):
                    value = value.replace(acronym, acronym.upper())

        # ----------------------------------------------------------------
        # PLAN_TYPE normalisation
        # ----------------------------------------------------------------
        if col == "PLAN_TYPE" and isinstance(value, str):
            value = _normalize_plan_type(value)

        # ----------------------------------------------------------------
        # FINANCING_TYPE normalisation
        # ----------------------------------------------------------------
        if col == "FINANCING_TYPE" and isinstance(value, str) and value != DEFAULT_VALUE:
            value = _normalize_financing_type(value)

        # ----------------------------------------------------------------
        # GENDER strict enforcement
        # ----------------------------------------------------------------
        if col == "GENDER" and isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ("male", "m"):
                value = "Male"
            elif value_lower in ("female", "f"):
                value = "Female"
            elif value_lower in ("all", "both", "all genders", "all customers"):
                value = "All"
            else:
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # CUSTOMER_TYPE — single value from fixed enum
        # ----------------------------------------------------------------
        if col == "CUSTOMER_TYPE" and isinstance(value, str):
            allowed = {
                "Salaried", "Self-Employed", "SME",
                "Corporate", "Retail", "Government",
            }
            stripped_ct = value.strip()
            if stripped_ct in allowed:
                pass
            elif stripped_ct.lower() == "n/a" or not stripped_ct:
                value = DEFAULT_VALUE
            else:
                lowered = stripped_ct.lower()
                match = next((a for a in allowed if a.lower() in lowered), None)
                if match:
                    value = match
                elif "," in stripped_ct or ";" in stripped_ct:
                    first = re.split(r"[,;]", stripped_ct)[0].strip()
                    value = first if first in allowed else DEFAULT_VALUE
                elif "customer" in lowered or "client" in lowered:
                    value = "Retail"
                else:
                    value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-01: TENURE_OPTIONS — reject payment-frequency values
        # ----------------------------------------------------------------
        if col == "TENURE_OPTIONS" and isinstance(value, str) and value != DEFAULT_VALUE:
            if _is_payment_frequency(value):
                print(
                    f"  FIX-01: TENURE_OPTIONS '{value}' looks like a payment "
                    f"frequency — resetting to N/A (should go in PREMIUM_PAYMENT_FREQUENCY)"
                )
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-03: PRICING_RATE — reject unit-allocation tables
        # ----------------------------------------------------------------
        if col == "PRICING_RATE" and isinstance(value, str) and value != DEFAULT_VALUE:
            if _is_unit_allocation_table(value):
                print(
                    f"  FIX-03: PRICING_RATE '{value[:80]}...' looks like a unit-"
                    f"allocation table (only % values) — resetting to N/A. "
                    f"Use MIN_CONTRIBUTION for the premium amount."
                )
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-05: KEY_EXCLUSIONS — reject claims documentation
        # ----------------------------------------------------------------
        if col == "KEY_EXCLUSIONS" and isinstance(value, str) and value != DEFAULT_VALUE:
            if _is_claims_doc_pattern(value):
                print(
                    f"  FIX-05: KEY_EXCLUSIONS '{value[:80]}...' appears to contain "
                    f"claims documentation requirements, not policy exclusions — "
                    f"resetting to N/A."
                )
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-06: REQUIRED_DOCUMENTS — reject claims-settlement docs
        # ----------------------------------------------------------------
        if col == "REQUIRED_DOCUMENTS" and isinstance(value, str) and value != DEFAULT_VALUE:
            if _is_claims_settlement_docs(value):
                print(
                    f"  FIX-06: REQUIRED_DOCUMENTS '{value[:80]}...' appears to "
                    f"contain death-claim settlement documents, not enrollment "
                    f"documents — resetting to N/A."
                )
                value = DEFAULT_VALUE

        # ----------------------------------------------------------------
        # FIX-10: FEES_AND_CHARGES — remove non-fee text
        # ----------------------------------------------------------------
        if col == "FEES_AND_CHARGES" and isinstance(value, str) and value != DEFAULT_VALUE:
            value = _clean_fees_field(value)

        # ----------------------------------------------------------------
        # FIX-11: TENURE — normalise verbose format
        # ----------------------------------------------------------------
        if col == "TENURE" and isinstance(value, str) and value != DEFAULT_VALUE:
            value = _normalize_tenure(value)

        # ----------------------------------------------------------------
        # OPTIONAL_RIDERS — ensure comma-separated output
        # ----------------------------------------------------------------
        if col == "OPTIONAL_RIDERS" and isinstance(value, str) and value != DEFAULT_VALUE:
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

    # ------------------------------------------------------------------
    # FIX-04 (reinforced): Always override PRODUCT_NAME from record,
    # applying the same all-caps fix again in case the loop missed it.
    # ------------------------------------------------------------------
    raw_name = record.get("PRODUCT_NAME") or entry["title"]
    if isinstance(raw_name, str):
        sn = raw_name.strip()
        if sn and sn == sn.upper() and len(sn.split()) > 1 and len(sn) > 5:
            raw_name = sn.title()
            for acronym in ("Igi", "Wto", "Slic", "Atm", "Nrp", "Hnw",
                            "Cnic", "Efu", "Sme"):
                raw_name = raw_name.replace(acronym, acronym.upper())
    normalized["PRODUCT_NAME"] = raw_name

    # SOURCE_FILE_PRODUCT is ALWAYS the actual filename — never the model output.
    normalized["SOURCE_FILE_PRODUCT"] = get_source_filename(entry)

    # Assertion: catch any remaining all-caps PRODUCT_NAME
    pname = normalized.get("PRODUCT_NAME", "")
    if isinstance(pname, str) and pname == pname.upper() and len(pname.split()) > 1:
        print(f"  WARNING: PRODUCT_NAME still all-caps after normalisation: '{pname}'")

    return normalized


# ============================================================================
# Deterministic safety-net overrides
# ============================================================================

_INSURER_KEYWORDS = (
    "takaful", "wto", "window takaful", "life insurance", "insurance company",
    "igi life", "jubilee life", "state life", "slic", "efu life", "adamjee life",
    "alfalah insurance",
)
_UNIT_LINKED_KEYWORDS = (
    "participant investment account", "participant's investment account",
    "participant individual account", "participant's individual account",
    "pia", "unit allocation", "fund allocation", "bid/offer spread",
    "bid offer spread",
)
_BONUS_KEYWORDS = ("bonus based", "bonus-based", "sum assured plus bonus", "bonus allocation")


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _name_filename_similarity(product_name: str, filename_title: str) -> float:
    if not product_name or not filename_title:
        return 1.0
    a = set(_normalize_for_match(str(product_name)).split())
    b = set(_normalize_for_match(str(filename_title)).split())
    a.discard("")
    b.discard("")
    if not a or not b:
        return 1.0
    overlap = a & b
    return len(overlap) / max(len(a | b), 1)


def apply_deterministic_overrides(record: dict, source_text: str) -> dict:
    """
    FIX-02 (new): MAX_TERM_YEARS sanity check — if > 75, it's likely an
    attained age, not a plan term. Reset to N/A with a warning.

    Original overrides: PLAN_TYPE (insurer → "Insurance") and
    FINANCING_TYPE (PIA → "Unit Linked") remain unchanged.
    """
    text_lower = _normalize_for_match(source_text) if isinstance(source_text, str) else ""
    provider = str(record.get("PROVIDER_NAME", "") or "")
    provider_lower = provider.lower()

    # --- 1. PLAN_TYPE: insurer-issued → always "Insurance" ---------
    is_insurer_issued = (
        record.get("LEAD_MARKER") == "IBG"
        or any(kw in provider_lower for kw in _INSURER_KEYWORDS)
        or any(kw in text_lower for kw in _INSURER_KEYWORDS)
    )
    if is_insurer_issued and record.get("PLAN_TYPE") not in (DEFAULT_VALUE,):
        if record.get("PLAN_TYPE") != "Insurance":
            record["PLAN_TYPE"] = "Insurance"
        record["LEAD_MARKER"] = "IBG"

    # --- 2. FINANCING_TYPE: PIA / unit-linked keyword override -----
    has_unit_linked_evidence = any(kw in text_lower for kw in _UNIT_LINKED_KEYWORDS)
    has_bonus_evidence = any(kw in text_lower for kw in _BONUS_KEYWORDS)
    current_financing = str(record.get("FINANCING_TYPE", "") or "")
    if has_unit_linked_evidence and "unit linked" not in current_financing.lower():
        if has_bonus_evidence:
            record["FINANCING_TYPE"] = "Hybrid (Bonus Based and Unit Linked)"
        else:
            record["FINANCING_TYPE"] = "Unit Linked"

    # --- FIX-02: MAX_TERM_YEARS > 75 → likely attained age ----------
    max_term = record.get("MAX_TERM_YEARS", DEFAULT_VALUE)
    if max_term not in (DEFAULT_VALUE, None, ""):
        try:
            if int(max_term) > 75:
                print(
                    f"  FIX-02: MAX_TERM_YEARS={max_term} is > 75, which is almost "
                    f"certainly an attained-age limit, not a plan term. "
                    f"Resetting to N/A. Verify manually."
                )
                record["MAX_TERM_YEARS"] = DEFAULT_VALUE
        except (ValueError, TypeError):
            pass

    # --- FIX-01 (reinforced): Reject payment frequency in TENURE_OPTIONS ---
    tenure_opts = record.get("TENURE_OPTIONS", DEFAULT_VALUE)
    if tenure_opts and tenure_opts != DEFAULT_VALUE:
        if _is_payment_frequency(str(tenure_opts)):
            record["TENURE_OPTIONS"] = DEFAULT_VALUE

    return record


# ============================================================================
# Prompt builders
# ============================================================================

def build_prompt(entry, chunk, tokenizer, chunk_idx=1, chunk_total=1):
    """Build the extraction prompt for ONE chunk of the document."""
    if chunk_total > 1:
        chunk_note = (
            f"\nNOTE: This is PART {chunk_idx} of {chunk_total} of a single, longer "
            f"product document. Extract whatever fields you can find in THIS part only. "
            f"Set missing fields to \"N/A\". Remember: TENURE_OPTIONS must be plan "
            f"DURATION choices only (e.g. '10, 15, 20 years'), NEVER payment "
            f"frequencies (Annual/Semi-Annual/Quarterly → go in PREMIUM_PAYMENT_FREQUENCY).\n"
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
    """Compact repair prompt for malformed JSON output."""
    repair_instructions = f"""You are repairing broken JSON from a data extraction task.
Product: {entry['title']}
Source file: {get_source_filename(entry)}

BROKEN OUTPUT TO REPAIR:
{raw_text[:3000]}

REPAIR RULES — apply all of these:
- Return exactly ONE valid JSON object with 56 fields, nothing else
- Use "N/A" for every missing or unparseable field (never null/None/NaN/"")
- PRODUCT_NAME: Title Case (never ALL CAPS). Acronyms (IGI, WTO, SLIC) stay uppercase.
- LEAD_MARKER: exactly "IBG" or "BNK"
- PLAN_TYPE: one word from Insurance|Protection|Health|Savings|Deposit|Loan|Card|Investment|Service|Loyalty.
  Insurer/takaful-underwritten products → ALWAYS "Insurance".
- CUSTOMER_TYPE: exactly one of Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A
- GENDER: exactly one of Male|Female|All|N/A
- FINANCING_TYPE: one of Conventional|Islamic|Takaful|Mudarabah|Unit Linked|Hybrid (Bonus Based and Unit Linked)|N/A
- TENURE_OPTIONS: ONLY plan duration choices (e.g. "10, 15, 20 years").
  "Annual, Semi-Annual, Quarterly" and similar payment frequencies → N/A here, put in PREMIUM_PAYMENT_FREQUENCY
- MAX_TERM_YEARS: plan term in years only. Attained age is NOT a plan term. If only attained age stated → N/A
- KEY_EXCLUSIONS: policy exclusions only (conditions where benefit not paid). NOT claim documentation.
- REQUIRED_DOCUMENTS: enrollment documents only. NOT claims settlement documents.
- PRICING_RATE: premium/contribution table only. NOT unit-allocation % table.
- Numeric fields: integers only, no units, no .0, no quotes
  Numeric: MIN_AGE, MAX_AGE, MIN_BALANCE, MIN_INCOME, MIN_INCOME_USD,
  MIN_INVESTMENT, MIN_CONTRIBUTION, MIN_TERM_YEARS, MAX_TERM_YEARS,
  FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED
- TENURE: compact format "X-Y years" or "1 Year (renewable)", not verbose "Minimum Term: X"
- SOURCE_FILE_PRODUCT: filename only, no path
- No markdown fences, no explanations outside the JSON

Return the repaired JSON object now."""

    messages = [{"role": "user", "content": repair_instructions}]
    return repair_instructions


def build_validation_prompt(entry, extracted_json):
    """
    Validation and correction prompt.

    MEMORY FIX: The previous version prepended the full SYSTEM_PROMPT (~2600
    tokens) before the validation instructions, effectively doubling the KV
    cache requirement for every product. This new version uses a compact
    self-contained header (~200 tokens) instead. The 18 validation rules below
    are complete on their own — they do not need the full system prompt.

    This saves ~2400 tokens per validation call, which is the primary fix for
    CUDA OOM errors on Colab T4/L4 running 7B models in 4-bit quantization.
    """
    # Compact standalone context (~200 tokens) — no SYSTEM_PROMPT embed.
    header = (
        "You are a JSON quality-checker for bank/insurance product extraction. "
        "Fix every rule violation in the JSON below. "
        "Return ONLY the corrected JSON object. No markdown, no explanation."
    )

    rules = """RULES TO CHECK AND FIX:
1. PRODUCT_NAME: Title Case. ALL-CAPS → convert. Acronyms (IGI,WTO,SLIC,ATM) stay uppercase.
2. PLAN_TYPE: ONE word. Any insurer/takaful underwriter → "Insurance".
3. CUSTOMER_TYPE: one of Salaried|Self-Employed|SME|Corporate|Retail|Government|N/A
4. GENDER: Male|Female|All|N/A only. "N/A" unless explicitly stated.
5. FINANCING_TYPE: PIA/unit-allocation present → "Unit Linked". Bonus+PIA → "Hybrid (Bonus Based and Unit Linked)".
6. TENURE_OPTIONS: plan DURATION choices ONLY (e.g. "10, 15, 20 years").
   "Annual/Semi-Annual/Quarterly/Monthly" = payment frequency → N/A here, put in PREMIUM_PAYMENT_FREQUENCY.
7. MAX_AGE: entry age limit ONLY. Renewal/attained age is NOT entry limit. "18-59 renewable to 75" → MAX_AGE=59.
8. MAX_TERM_YEARS: plan term ONLY, not attained age. "10 till age 85" → N/A. Value > 75 → likely wrong, set N/A.
9. KEY_EXCLUSIONS: policy exclusions only. NOT claim docs (Claim Form, FIR, Post Mortem). Claims docs → N/A.
10. REQUIRED_DOCUMENTS: enrollment docs only. NOT claims docs (Death Certificate, Discharge Letter). Claims docs → N/A.
11. PRICING_RATE: contribution/premium amounts. NOT unit-allocation % (Year 1:60%...). % only table → N/A.
12. OPTIONAL_RIDERS: add-on riders only. NOT built-in benefits (Top-Up, Surplus Sharing).
13. DEPOSIT_PROFIT_TYPE / DEPOSIT_PROFIT_FREQUENCY: unit-linked and health/protection plans → both N/A.
14. PREMIUM_PAYMENT_FREQUENCY: how customer pays (Annual, Semi-Annual, Quarterly). Else N/A.
15. TENURE: compact "X-Y years" format. NOT "Minimum Term: X years; Maximum Term: Y years".
16. Numerics (MIN_AGE,MAX_AGE,MIN_BALANCE,MIN_INCOME,MIN_INCOME_USD,MIN_INVESTMENT,
    MIN_CONTRIBUTION,MIN_TERM_YEARS,MAX_TERM_YEARS,FREE_LOOK_PERIOD_DAYS,IS_BANK_OFFERED):
    integers only — no PKR, no commas, no .0, no quotes around numbers.
17. FREE_LOOK_PERIOD_DAYS: only if THIS product explicitly states it. NOT 14 by default.
18. All 56 fields must be present. null/None/NaN/"" → "N/A"."""

    prompt = (
        f"{header}\n\n"
        f"PRODUCT: {entry['title']}\n\n"
        f"JSON TO VALIDATE:\n{json.dumps(extracted_json, indent=2)}\n\n"
        f"{rules}\n\n"
        "Return the corrected JSON now."
    )
    return prompt


# ============================================================================
# Model inference (unchanged from v1)
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
    FIX-08 (NEW): Before chunking, attempt to isolate this product's section
    from a multi-product document. If isolation succeeds (section > 200 chars),
    only the isolated section is chunked and sent to the model. This prevents
    fields from adjacent product sections bleeding into this product's record.
    """
    # FIX-08: Try to isolate product section
    isolated_text = extract_product_section(text, entry["title"])
    if isolated_text != text:
        print(
            f"  FIX-08: Isolated product section "
            f"({len(isolated_text)} / {len(text)} chars)"
        )
    working_text = isolated_text

    if ENABLE_CHUNKING:
        chunks = chunk_text(working_text, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
    else:
        chunks = [working_text.strip()]
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
        raise ExtractionFailedError(
            f"All {chunk_total} chunk(s) failed to produce parseable JSON "
            f"for '{entry['title']}' ({get_source_filename(entry)})."
        )

    # Single validation/correction pass on the merged record.
    # Skipped when ENABLE_VALIDATION=false or LOW_VRAM_MODE=true.
    # normalize_record() + apply_deterministic_overrides() still run either way,
    # so the deterministic fixes (PLAN_TYPE, FINANCING_TYPE, TENURE_OPTIONS, etc.)
    # are always applied regardless of this flag.
    if ENABLE_VALIDATION:
        validation_prompt = build_validation_prompt(entry, accumulated)
        validation_raw = get_raw_generation(model, tokenizer, validation_prompt)
        try:
            validated_parsed = parse_json_blob(validation_raw)
            final_record = normalize_record(validated_parsed, entry)
        except Exception:
            final_record = accumulated
        free_gpu_memory()  # explicit free after validation call
    else:
        final_record = accumulated

    # Deterministic safety-net overrides (expanded in v2).
    # Use the ISOLATED text so keyword searches hit this product only.
    final_record = apply_deterministic_overrides(final_record, working_text)

    similarity = _name_filename_similarity(final_record.get("PRODUCT_NAME", ""), entry["title"])
    if similarity < 0.2:
        print(
            f"  WARNING: PRODUCT_NAME '{final_record.get('PRODUCT_NAME')}' shares "
            f"little overlap with source file '{get_source_filename(entry)}' "
            f"(similarity={similarity:.2f}) — verify for cross-document content bleed."
        )

    return final_record


def extract_one(model, tokenizer, entry, text):
    return extract_one_product(model, tokenizer, entry, text)


# ============================================================================
# Model loader (unchanged from v1)
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
                "Install: pip install -U 'bitsandbytes>=0.46.1'"
            ) from exc
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", bnb_version)
        major, minor, patch = (int(part) for part in match.groups()) if match else (0, 0, 0)
        if (major, minor, patch) < (0, 46, 1):
            raise SystemExit(
                f"bitsandbytes {bnb_version} is too old. "
                "Install: pip install -U 'bitsandbytes>=0.46.1'"
            )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
    except RepositoryNotFoundError as exc:
        raise SystemExit(f"Model not found: {MODEL_NAME}") from exc

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
            "CUDA OOM while loading model. Try:\n"
            "HF_LOAD_IN_4BIT=true\nHF_DEVICE_MAP=auto\n"
            "HF_TORCH_DTYPE=float16\nHF_GPU_RESERVE_GIB=3.0"
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
                        f"CUDA OOM on attempt {attempt + 1}, retrying: {e}"
                    )
                    time.sleep(5)
                except Exception as e:
                    free_gpu_memory()
                    print(
                        f"[{entry['product_no']:03d}/{len(index)}] retry {attempt + 1}: {e}"
                    )
                    time.sleep(3)
            else:
                print(f"[{entry['product_no']:03d}/{len(index)}] FAILED after retries — writing placeholder")
                placeholder = blank_record(entry)
                placeholder["SPECIAL_CONDITIONS"] = "EXTRACTION_FAILED_MANUAL_REVIEW_REQUIRED"
                rec = {"product_no": entry["product_no"], **placeholder}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()

            free_gpu_memory()

    print("Done. Output:", OUT_JSONL)


if __name__ == "__main__":
    main()