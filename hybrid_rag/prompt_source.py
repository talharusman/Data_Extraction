"""
prompt_source.py
=================

Parses the ORIGINAL EXTRACTION_SYSTEM_PROMPT.txt (your tested, working
56-field prompt) into structured, VERBATIM segments:

    HEADER              -- the intro lines before "COLUMNS:"
    GLOBAL_RULES         -- {"G1": "...", "G2": "...", ...} exact text
    COLUMN_RULES          -- {"PRODUCT_NAME": "...", ...} exact text,
                             one entry per field (combined rules like
                             "SERVICE_TYPE/REWARD_TYPE: ..." are split so
                             BOTH fields point at the identical verbatim
                             text -- nothing is reworded)
    WORKED_EXAMPLES        -- [{"title", "text", "fields": {...}}, ...]
                             verbatim example blocks, tagged with which
                             fields they demonstrate
    DO_NOT_LINES            -- [{"text", "fields": {...}}, ...] verbatim
                              "x ..." bullets, tagged with which fields
                              they reference (continuation lines merged)

Nothing here rewrites, simplifies, or reinterprets a single word of the
original prompt -- this module only locates section boundaries with
regex so prompt_builder.py can slice out exactly the rules relevant to
one field group. This is a ONE-TIME, dev-time parse (module load time),
matching the same "never regroup/reparse at runtime" constraint as
field_groups.py.

If EXTRACTION_SYSTEM_PROMPT.txt changes, this module re-parses it
automatically on next process start -- there is no hand-maintained
duplicate of your rules anywhere in this codebase.
"""
from __future__ import annotations

import re
from pathlib import Path

_FIELD_TOKEN = re.compile(r"^([A-Z][A-Z0-9_]*(?:/[A-Z][A-Z0-9_]*)*):", re.MULTILINE)
_GLOBAL_TOKEN = re.compile(r"^(G\d+)\.", re.MULTILINE)
_DO_NOT_LINE = re.compile(r"^\u00d7\s+")  # '×' bullet marker


def _find_prompt_file() -> Path:
    candidates = [
        Path("EXTRACTION_SYSTEM_PROMPT.txt"),
        Path(__file__).parent / "EXTRACTION_SYSTEM_PROMPT.txt",
        Path(__file__).parent.parent / "EXTRACTION_SYSTEM_PROMPT.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "prompt_source.py could not find EXTRACTION_SYSTEM_PROMPT.txt "
        "(searched CWD, hybrid_rag/, and project root)."
    )


def _slice_section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start) if end_marker else len(text)
    return text[start:end].strip("\n")


def _parse_header(text: str) -> str:
    return text[: text.index("COLUMNS:")].strip()


def _parse_global_rules(section: str) -> dict[str, str]:
    matches = list(_GLOBAL_TOKEN.finditer(section))
    rules: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        rules[key] = section[start:end].strip()
    return rules


def _split_field_tokens(token_group: str) -> list[str]:
    return token_group.split("/")


def _parse_column_rules(section: str) -> dict[str, str]:
    matches = list(_FIELD_TOKEN.finditer(section))
    rules: dict[str, str] = {}
    for i, m in enumerate(matches):
        token_group = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        block = section[start:end].strip()
        # Combined entries (e.g. "SERVICE_TYPE/REWARD_TYPE: ...") map every
        # listed field to the SAME verbatim block -- the rule text is not
        # duplicated/reworded per field, just referenced by both.
        for field in _split_field_tokens(token_group):
            rules[field] = block
    return rules


def _fields_mentioned(text: str, known_fields: set[str]) -> set[str]:
    """
    Detects which known fields a block of text references. Handles both a
    field appearing standalone (e.g. "GENDER=...") and prose slash-combos
    that only spell out a shared prefix once (e.g.
    "DEPOSIT_PROFIT_TYPE/FREQUENCY" meaning both DEPOSIT_PROFIT_TYPE and
    DEPOSIT_PROFIT_FREQUENCY).
    """
    found = set()
    for field in known_fields:
        if re.search(rf"\b{re.escape(field)}\b", text):
            found.add(field)
    for combo in re.findall(r"\b([A-Z][A-Z0-9_]*/[A-Z0-9_]+(?:/[A-Z0-9_]+)*)\b", text):
        parts = combo.split("/")
        base = parts[0]
        if "_" in base:
            prefix = base.rsplit("_", 1)[0]
            candidates = [base] + [f"{prefix}_{p}" for p in parts[1:]]
        else:
            candidates = parts
        for c in candidates:
            if c in known_fields:
                found.add(c)
    return found


def _parse_worked_examples(section: str, known_fields: set[str]) -> list[dict]:
    # Example titles are non-indented lines ending in ':' that are NOT a
    # pure FIELD_TOKEN line (i.e. contain lowercase letters -- real column
    # rule lines are always ALL_CAPS_WITH_UNDERSCORES before the colon).
    lines = section.split("\n")
    title_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9 /&,\-]*:$")
    field_line_pattern = re.compile(r"^[A-Z][A-Z0-9_]*:$")

    titles_idx = [
        i for i, line in enumerate(lines)
        if title_pattern.match(line.strip()) and not field_line_pattern.match(line.strip())
    ]

    examples = []
    for n, idx in enumerate(titles_idx):
        end = titles_idx[n + 1] if n + 1 < len(titles_idx) else len(lines)
        block = "\n".join(lines[idx:end]).strip()
        title = lines[idx].strip().rstrip(":")
        examples.append({
            "title": title,
            "text": block,
            "fields": _fields_mentioned(block, known_fields),
        })
    return examples


def _parse_do_not(section: str, known_fields: set[str]) -> list[dict]:
    lines = section.split("\n")
    bullets: list[dict] = []
    current: list[str] | None = None
    for line in lines:
        if _DO_NOT_LINE.match(line.strip()):
            if current is not None:
                text = " ".join(current).strip()
                bullets.append({"text": text, "fields": _fields_mentioned(text, known_fields)})
            current = [line.strip()]
        elif line.strip() and current is not None:
            # continuation of the previous wrapped bullet
            current.append(line.strip())
    if current is not None:
        text = " ".join(current).strip()
        bullets.append({"text": text, "fields": _fields_mentioned(text, known_fields)})
    return bullets


def _load() -> dict:
    raw = _find_prompt_file().read_text(encoding="utf-8")

    header = _parse_header(raw)

    global_section = _slice_section(raw, "===GLOBAL RULES===", "===COLUMN RULES===")
    global_rules = _parse_global_rules(global_section)

    column_section = _slice_section(
        raw, "===COLUMN RULES===", "===WORKED EXAMPLES (ground-truth style)==="
    )
    column_rules = _parse_column_rules(column_section)
    known_fields = set(column_rules.keys())

    examples_section = _slice_section(
        raw, "===WORKED EXAMPLES (ground-truth style)===", "===DO NOT==="
    )
    worked_examples = _parse_worked_examples(examples_section, known_fields)

    do_not_section = _slice_section(raw, "===DO NOT===", None)
    do_not_lines = _parse_do_not(do_not_section, known_fields)

    return {
        "header": header,
        "global_rules": global_rules,
        "column_rules": column_rules,
        "worked_examples": worked_examples,
        "do_not_lines": do_not_lines,
        "raw_text": raw,
    }


_PARSED = _load()

HEADER: str = _PARSED["header"]
GLOBAL_RULES: dict[str, str] = _PARSED["global_rules"]
COLUMN_RULES: dict[str, str] = _PARSED["column_rules"]
WORKED_EXAMPLES: list[dict] = _PARSED["worked_examples"]
DO_NOT_LINES: list[dict] = _PARSED["do_not_lines"]

# Global rules that apply to every group regardless of which fields it
# contains (document-level / formatting-level, not field-specific).
ALWAYS_APPLICABLE_GLOBAL_RULES = ["G1", "G4", "G6", "G9", "G10", "G11"]

# Global rules that only apply when the group contains the triggering field(s).
CONDITIONAL_GLOBAL_RULES = {
    "G2": {"MIN_AGE", "MAX_AGE", "MIN_TERM_YEARS", "MAX_TERM_YEARS",
           "FREE_LOOK_PERIOD_DAYS", "IS_BANK_OFFERED", "MIN_BALANCE",
           "AVG_BALANCE_REQUIREMENT", "MIN_INCOME", "MIN_INCOME_USD",
           "MIN_INVESTMENT", "MIN_CONTRIBUTION"},
    "G3": {"SOURCE_FILE_PRODUCT"},
    "G5": {"IS_BANK_OFFERED"},
    "G7": {"CUSTOMER_SEGMENT", "TARGET_SEGMENT", "SEGMENT_TIER",
           "SERVICE_TYPE", "CURRENCY_TYPE", "GENDER"},
    "G8": {"PRODUCT_NAME"},
}


def global_rules_for_fields(fields: set[str]) -> list[str]:
    """Returns the G-rule keys (in original order) relevant to this field set."""
    keys = list(ALWAYS_APPLICABLE_GLOBAL_RULES)
    for g_key, trigger_fields in CONDITIONAL_GLOBAL_RULES.items():
        if trigger_fields & fields:
            keys.append(g_key)
    order = sorted(keys, key=lambda k: int(k[1:]))
    return order


def examples_for_fields(fields: set[str]) -> list[dict]:
    return [ex for ex in WORKED_EXAMPLES if ex["fields"] & fields]


def do_not_lines_for_fields(fields: set[str]) -> list[dict]:
    """
    Field-specific DO NOT bullets relevant to this group, PLUS bullets with
    no field tag at all (pure formatting/process rules like the ';'
    separator or trailing-product rules) which apply to every group.
    """
    return [
        d for d in DO_NOT_LINES
        if (d["fields"] & fields) or not d["fields"]
    ]
