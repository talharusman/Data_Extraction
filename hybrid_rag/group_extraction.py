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

import json
import re

from .prompt_builder import build_group_messages


def _extract_json_object(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    first, last = raw.find("{"), raw.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = raw[first : last + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
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
    """
    result: dict[str, dict] = {
        name: {"value": None, "confidence": 0.0, "evidence": None, "chunk_id": None, "page": None, "is_table": False}
        for name in group["fields"]
    }
    if not chunks:
        return result

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    messages = build_group_messages(entry, group, chunks, group_prompt_cache)

    try:
        raw = generate_fn(messages)
    except Exception as e:
        print(f"    [group_extraction] LLM call failed for group '{group['key']}': {e}")
        return result

    parsed = _extract_json_object(raw)
    if not parsed:
        print(f"    [group_extraction] unparseable output for group '{group['key']}'")
        return result

    for name in group["fields"]:
        raw_field = parsed.get(name)
        if raw_field is None:
            continue
        # Tolerate the model returning a bare value instead of the full
        # {value, confidence, evidence, ...} object.
        if isinstance(raw_field, dict):
            value = raw_field.get("value")
            confidence = raw_field.get("confidence", 0.5)
            evidence = raw_field.get("evidence")
            chunk_id = raw_field.get("chunk_id")
            page = raw_field.get("page")
        else:
            value, confidence, evidence, chunk_id, page = raw_field, 0.4, None, None, None

        if value in (None, "", "null", "N/A", "n/a"):
            continue

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.4
        confidence = max(0.0, min(1.0, confidence))

        evidence_ok, is_table = _verify_evidence(evidence, chunks_by_id)
        if evidence is not None and not evidence_ok:
            # Model cited evidence that isn't actually in the retrieved
            # text -- don't trust it fully, but don't discard the value
            # either (it may still be correct, just mis-cited).
            confidence = min(confidence, 0.4)

        # Resolve page/chunk_id from the actual cited chunk when possible,
        # rather than trusting the model's own (sometimes wrong) numbers.
        resolved_chunk = chunks_by_id.get(chunk_id) if chunk_id in chunks_by_id else None
        if resolved_chunk is None and evidence_ok:
            for c in chunks_by_id.values():
                if evidence and evidence.strip() in c["text"]:
                    resolved_chunk = c
                    break

        result[name] = {
            "value": value,
            "confidence": confidence,
            "evidence": evidence,
            "chunk_id": resolved_chunk["chunk_id"] if resolved_chunk else chunk_id,
            "page": resolved_chunk["page_number"] if resolved_chunk else page,
            "is_table": is_table,
        }

    return result
