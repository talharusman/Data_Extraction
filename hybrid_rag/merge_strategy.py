"""
merge_strategy.py
==================

Replaces the old "first non-null wins" merge in 02_extract_fields.py's
merge_records(). Given one or more extraction *candidates* for the same
field (e.g. from an optional multi-pass ensemble, or from multiple groups
that both happened to touch a field), pick the winner using, in order:

    1. Highest confidence
    2. Explicit evidence preferred (evidence present + verified > none)
    3. Table values preferred over paragraph values
    4. Majority agreement (if >=2 candidates agree on a normalized value,
       that value wins even over a single higher-confidence outlier)
    5. Most recent page, as the final tiebreaker

A "candidate" dict has the shape produced by group_extraction.extract_group:
    {"value", "confidence", "evidence", "chunk_id", "page", "is_table"}
"""
from __future__ import annotations

from collections import Counter


def _normalize_for_agreement(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _sort_key(cand: dict):
    # Higher is better on every axis below (agreement handled separately).
    has_value = 1 if cand.get("value") not in (None, "", "N/A") else 0
    confidence = cand.get("confidence") or 0.0
    has_evidence = 1 if cand.get("evidence") else 0
    is_table = 1 if cand.get("is_table") else 0
    page = cand.get("page") or 0
    return (has_value, confidence, has_evidence, is_table, page)


def pick_best_candidate(candidates: list[dict]) -> dict | None:
    """
    Applies the 5-step merge strategy to a list of candidates for ONE field
    and returns the winning candidate dict (or None if all are empty).
    """
    real = [c for c in candidates if c and c.get("value") not in (None, "", "N/A")]
    if not real:
        return None
    if len(real) == 1:
        return real[0]

    # Step 4: majority agreement check first, since it can override a lone
    # higher-confidence outlier per spec ordering (checked before falling
    # back to pure confidence sort when there's a genuine tie/near-tie).
    counts = Counter(_normalize_for_agreement(c["value"]) for c in real)
    top_value, top_count = counts.most_common(1)[0]
    if top_count >= 2 and top_count > 1:
        agreeing = [c for c in real if _normalize_for_agreement(c["value"]) == top_value]
        agreeing.sort(key=_sort_key, reverse=True)
        best_non_agreeing = max(
            (c for c in real if _normalize_for_agreement(c["value"]) != top_value),
            key=_sort_key, default=None,
        )
        # Majority wins UNLESS a single outlier is decisively more confident
        # AND better evidenced (avoids majority of low-quality guesses
        # beating one strongly-verified extraction).
        if best_non_agreeing is not None:
            majority_score = _sort_key(agreeing[0])
            outlier_score = _sort_key(best_non_agreeing)
            if outlier_score > majority_score and best_non_agreeing.get("confidence", 0) >= 0.85 \
                    and best_non_agreeing.get("evidence"):
                return best_non_agreeing
        return agreeing[0]

    # Steps 1-3 + 5, via the composite sort key.
    real.sort(key=_sort_key, reverse=True)
    return real[0]


def merge_group_results(group_result_sets: list[dict[str, dict]]) -> dict[str, dict]:
    """
    Merges one or more full group-result dicts (field -> candidate) that
    cover the SAME set of fields (e.g. two ensemble passes over the same
    group) into a single field -> winning-candidate dict.
    """
    if not group_result_sets:
        return {}
    if len(group_result_sets) == 1:
        return group_result_sets[0]

    all_fields = set()
    for rs in group_result_sets:
        all_fields |= set(rs.keys())

    merged: dict[str, dict] = {}
    for field in all_fields:
        candidates = [rs.get(field) for rs in group_result_sets if rs.get(field)]
        best = pick_best_candidate(candidates)
        merged[field] = best or {
            "value": None, "confidence": 0.0, "evidence": None,
            "chunk_id": None, "page": None, "is_table": False,
        }
    return merged


def candidates_to_flat_record(all_group_results: dict[str, dict[str, dict]], default_value: str = "N/A") -> dict:
    """
    Flattens {group_key: {field: candidate}} into a plain {field: value}
    record ready to hand to normalize_record() in 02_extract_fields.py,
    substituting default_value ("N/A") for any field with no surviving
    candidate -- preserving the original schema/output contract.
    """
    flat: dict = {}
    for group_key, field_results in all_group_results.items():
        for field, cand in field_results.items():
            value = cand.get("value") if cand else None
            flat[field] = value if value not in (None, "") else default_value
    return flat


def evidence_debug_view(all_group_results: dict[str, dict[str, dict]]) -> dict:
    """
    Returns {field: {"confidence", "evidence", "chunk_id", "page"}} for
    optional Debug Mode output -- kept OUT of the final JSONL record unless
    explicitly requested by the caller (see EXTRACTION_DEBUG_MODE in the
    orchestrator).
    """
    debug: dict = {}
    for field_results in all_group_results.values():
        for field, cand in field_results.items():
            if not cand:
                continue
            debug[field] = {
                "confidence": cand.get("confidence"),
                "evidence": cand.get("evidence"),
                "chunk_id": cand.get("chunk_id"),
                "page": cand.get("page"),
            }
    return debug
