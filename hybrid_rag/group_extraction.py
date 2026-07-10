"""
group_extraction.py
====================

Runs extraction for ONE field group against its retrieved chunks. Decoupled
from the specific model-loading code in 02_extract_fields.py via a
`generate_fn(messages) -> raw_text` callable the orchestrator supplies (a
thin wrapper around the existing build_prompt-style chat templating +
get_raw_generation()).

Evidence is verified against the retrieved chunk text (not just trusted
blindly): if the model's claimed evidence string isn't actually a substring
of the chunk it cited, confidence is capped low so the merge strategy will
prefer other candidates / fall back to N/A.
"""
from __future__ import annotations

import ast
import json
import re

from .prompt_builder import build_group_messages


def _strip_wrappers(text: str) -> str:
    """Strips markdown code fences, a leading 'json' label, and Qwen-style
    <think>...</think> reasoning blocks that some quantized checkpoints emit
    before the actual JSON answer."""
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)</?think>", "", text)
    return text.strip()


def _parse_candidate(candidate: str) -> dict | None:
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


def _close_unterminated_json(candidate: str) -> str | None:
    """
    Last-resort repair for output that got cut off mid-object (hit
    max_new_tokens). Walks the string tracking open braces/brackets/quotes
    outside of strings and appends whatever closers are needed to make it
    parseable. Returns None if the text isn't salvageable this way.
    """
    depth_curly = depth_square = 0
    in_string = False
    escape = False
    trailing_comma = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        trailing_comma = ch == ","

    if depth_curly <= 0 and depth_square <= 0 and not in_string:
        return None  # already balanced (or malformed in some other way) -- nothing to close

    repaired = candidate.rstrip()
    if in_string:
        repaired += '"'
    if trailing_comma:
        repaired = repaired.rstrip(", \n\t")
    repaired += "]" * max(0, depth_square) + "}" * max(0, depth_curly)
    return repaired


def _iter_json_candidates(text: str):
    """Yields plausible JSON-object substrings in priority order: fenced
    blocks, the whole cleaned text, the naive first{...last} span, then
    every balanced-brace object found by scanning (handles the model
    emitting prose before/after the JSON, or multiple objects)."""
    text = _strip_wrappers(text)
    seen: set[str] = set()

    def emit(candidate: str):
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        yield from emit(block)

    yield from emit(text)

    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and last > first:
        yield from emit(text[first : last + 1])

    for brace_start in (i for i, ch in enumerate(text) if ch == "{"):
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
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield from emit(text[brace_start : idx + 1])
                    break

    # Last resort: the longest '{'-starting fragment, auto-closed, in case
    # generation was cut off by max_new_tokens before any '}' appeared.
    if first != -1:
        repaired = _close_unterminated_json(text[first:])
        if repaired:
            yield from emit(repaired)


def _extract_json_object(raw: str) -> dict | None:
    for candidate in _iter_json_candidates(raw):
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = _parse_candidate(cleaned)
        if parsed is not None:
            return parsed
    return None


def _is_table_like(text: str) -> bool:
    """Rough table-vs-paragraph heuristic used by the merge strategy."""
    return bool(re.search(r"(\t.+\t)|(\|.+\|)|(:\s*[\d,]+\s*(\||\n|$))", text))


def _verify_evidence(evidence, chunks_by_id: dict[int, dict]) -> tuple[bool, bool]:
    """Returns (evidence_found_in_cited_chunk, is_table_evidence)."""
    if not evidence or not isinstance(evidence, str):
        return False, False
    for chunk in chunks_by_id.values():
        if evidence.strip() and evidence.strip() in chunk["text"]:
            return True, _is_table_like(chunk["text"])
    return False, False


def extract_group(
    generate_fn,
    entry: dict,
    group: dict,
    chunks: list[dict],
    group_prompt_cache: dict | None = None,
) -> dict[str, dict]:
    """
    Returns {field_name: {"value", "confidence", "evidence", "chunk_id",
    "page", "is_table"}} for every field in this group. Fields the model
    omits or that fail evidence verification come back with value=None,
    confidence=0.0.

    Terminal logging: prints per-field ✓/✗ results and a group summary so
    you can see exactly what was extracted and why, without enabling
    EXTRACTION_DEBUG_MODE.
    """
    # Minimum confidence below which an unverified value is treated as
    # noise and discarded. Prevents hallucinated values (e.g.
    # COLLATERAL_TYPE="Debit Card" in a takaful plan) from contaminating
    # the flat record.
    _CONFIDENCE_FLOOR = 0.25

    total_fields = len(group["field_order"])
    result: dict[str, dict] = {
        name: {"value": None, "confidence": 0.0, "evidence": None, "chunk_id": None, "page": None, "is_table": False}
        for name in group["fields"]
    }
    if not chunks:
        print(f"    ⚠  No chunks retrieved for group '{group['name']}' — all fields N/A")
        return result

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    messages = build_group_messages(entry, group, chunks, group_prompt_cache)

    try:
        raw = generate_fn(messages)
    except Exception as e:
        print(f"    ✗ LLM call FAILED for group '{group['name']}': {e}")
        return result

    if not raw or not raw.strip():
        print(f"    ✗ EMPTY generation for group '{group['name']}'"
              f" — check chat template / stop tokens / max_new_tokens.")
        return result

    parsed = _extract_json_object(raw)
    if not parsed:
        preview = raw.strip().replace("\n", " ")[:200]
        print(f"    ✗ Unparseable JSON for group '{group['name']}'"
              f" ({len(raw)} chars). Preview: {preview!r}")
        return result

    # Detect the "single-envelope" failure mode: the model returned one
    # {value, confidence, evidence, chunk_id, page} object at the top level
    # instead of one per field. This happens when a small model misreads the
    # output contract. In this case none of the field names exist as keys.
    _ENVELOPE_KEYS = {"value", "confidence", "evidence", "chunk_id", "page"}
    top_keys = set(str(k).lower() for k in parsed.keys())
    if top_keys <= _ENVELOPE_KEYS and _ENVELOPE_KEYS & top_keys:
        preview = raw.strip().replace("\n", " ")[:300]
        print(f"    ✗ Model returned a single RAG envelope instead of one per field "
              f"for group '{group['name']}'. Raw preview: {preview!r}")
        return result

    # Build a case-insensitive lookup as a fallback for models that return
    # lowercase field names (e.g. 'product_name' instead of 'PRODUCT_NAME').
    parsed_lower = {k.lower(): v for k, v in parsed.items()}

    extracted_count = 0
    for name in group["field_order"]:
        # Try exact key first; fall back to lowercase match.
        if name in parsed:
            raw_field = parsed[name]
        elif name.lower() in parsed_lower:
            raw_field = parsed_lower[name.lower()]
            print(f"      ~ {name}: found via lowercase key lookup")
        else:
            print(f"      ✗ {name}: (not in model output)")
            continue

        if raw_field is None:
            print(f"      ✗ {name}: (null — model explicitly returned null)")
            continue

        # Tolerate the model returning a bare value instead of the full
        # {value, confidence, evidence, ...} object.
        if isinstance(raw_field, dict):
            value      = raw_field.get("value")
            confidence = raw_field.get("confidence", 0.5)
            evidence   = raw_field.get("evidence")
            chunk_id   = raw_field.get("chunk_id")
            page       = raw_field.get("page")
        else:
            value, confidence, evidence, chunk_id, page = raw_field, 0.4, None, None, None

        if value in (None, "", "null", "N/A", "n/a"):
            print(f"      ✗ {name}: (null/N/A)")
            continue

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.4
        confidence = max(0.0, min(1.0, confidence))

        evidence_ok, is_table = _verify_evidence(evidence, chunks_by_id)
        if evidence is not None and not evidence_ok:
            # Model cited evidence that isn't actually in the retrieved
            # text — cap confidence; if very low, discard as hallucination.
            confidence = min(confidence, 0.4)

        # BUG FIX: confidence floor to reject hallucinated values.
        # A 3B model on an unrelated chunk will sometimes assign a field
        # like COLLATERAL_TYPE="Debit Card" with low confidence. Without
        # this floor, that wrong value would make it into the flat record
        # and survive all the way to the final JSONL output.
        if confidence < _CONFIDENCE_FLOOR and not evidence_ok:
            print(f"      ✗ {name}: '{str(value)[:50]}' "
                  f"(conf={confidence:.2f} < floor={_CONFIDENCE_FLOOR}, no evidence → discarded)")
            continue

        # BUG FIX: chunk_id may be None (NoneType not hashable) — the old
        # code did `chunk_id in chunks_by_id` which raises TypeError when
        # chunk_id is None. Guard with an explicit None check.
        resolved_chunk = None
        if chunk_id is not None and chunk_id in chunks_by_id:
            resolved_chunk = chunks_by_id[chunk_id]
        if resolved_chunk is None and evidence_ok:
            for c in chunks_by_id.values():
                if evidence and evidence.strip() in c["text"]:
                    resolved_chunk = c
                    break

        result[name] = {
            "value":      value,
            "confidence": confidence,
            "evidence":   evidence,
            "chunk_id":   resolved_chunk["chunk_id"] if resolved_chunk else chunk_id,
            "page":       resolved_chunk["page_number"] if resolved_chunk else page,
            "is_table":   is_table,
        }
        val_preview = str(value)[:60].replace("\n", " ")
        ev_flag = "✓ev" if evidence_ok else ("?ev" if evidence else "no-ev")
        print(f"      ✓ {name}: \"{val_preview}\" (conf={confidence:.2f}, {ev_flag})")
        extracted_count += 1

    status = "✓" if extracted_count > 0 else "✗"
    print(f"    {status} Group '{group['name']}': {extracted_count}/{total_fields} fields extracted")
    return result