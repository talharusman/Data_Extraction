"""
prompt_builder.py
==================

REVISED per "Existing Extraction Prompt Migration": group prompts are now
assembled entirely from VERBATIM slices of your original, tested
EXTRACTION_SYSTEM_PROMPT.txt (via prompt_source.py) -- header wording,
global rules G1-G11, per-field column rules, relevant worked examples,
and relevant DO NOT bullets are all reused unchanged. Nothing is
paraphrased, simplified, or reworded. Only fields/rules/examples that
don't belong to the current group are left out.

The only text NOT taken from your original prompt is the small RAG output
envelope (base_prompt.RAG_OUTPUT_CONTRACT) telling the model to wrap each
value in {value, confidence, evidence, chunk_id, page} -- required so
merge_strategy.py can do confidence/evidence-based merging. This is an
output-format instruction, not a new extraction/business rule.
"""
from __future__ import annotations

import json

from .base_prompt import RAG_OUTPUT_CONTRACT
from . import prompt_source as ps


def _expected_json_skeleton(field_names: list[str]) -> str:
    skeleton = {
        name: {"value": "...", "confidence": 0.0, "evidence": "...", "chunk_id": 0, "page": 0}
        for name in field_names
    }
    return json.dumps(skeleton, indent=2)


def build_group_system_prompt(group: dict) -> str:
    """
    Assembles the group's system prompt from verbatim original-prompt
    slices:
        1. Original header ("You are a data extraction engine...").
        2. Only the COLUMNS relevant to this group (not all 56).
        3. Only the G-rules relevant to this group (verbatim G1..G11 text).
        4. Only this group's COLUMN RULES entries (verbatim).
        5. Only WORKED EXAMPLES that demonstrate a field in this group
           (verbatim, full example block).
        6. Only DO NOT bullets that reference a field in this group, plus
           the format-only bullets that apply everywhere (verbatim).
        7. The RAG output contract (value/confidence/evidence wrapper).

    Cached per group by the caller (see build_group_messages) so this is
    built once per group per process, not per call.
    """
    field_names = group["field_order"]
    field_set = set(field_names)

    # The original header's first line ("Extract ALL 56 columns below.")
    # would directly contradict a group-scoped prompt, so -- and only for
    # this one structural/procedural line, not any field-specific rule --
    # it's adapted to name this group's column count instead. Every other
    # word of the header is kept verbatim.
    header = ps.HEADER.replace(
        "Extract ALL 56 columns below. Return ONLY one valid JSON object",
        f'Extract ONLY the {len(field_names)} columns in the "{group["name"]}" '
        f"group below (part of a larger 56-column schema handled across "
        f"separate calls). Return ONLY one valid JSON object",
    )
    parts: list[str] = [header]

    parts.append(f"COLUMNS IN THIS GROUP: {', '.join(field_names)}")

    # --- Global rules (verbatim), only the ones relevant to this group ---
    g_keys = ps.global_rules_for_fields(field_set)
    if g_keys:
        global_block = "\n".join(ps.GLOBAL_RULES[k] for k in g_keys)
        parts.append(f"===GLOBAL RULES (subset relevant to this group)===\n{global_block}")

    # --- Column rules (verbatim), only for this group's fields, in schema order ---
    seen_blocks: set[str] = set()
    column_blocks: list[str] = []
    for name in field_names:
        block = ps.COLUMN_RULES[name]
        if block not in seen_blocks:  # combined entries (e.g. SERVICE_TYPE/REWARD_TYPE) printed once
            seen_blocks.add(block)
            column_blocks.append(block)
    parts.append("===COLUMN RULES (this group only)===\n" + "\n".join(column_blocks))

    # --- Worked examples (verbatim), only ones touching this group's fields ---
    examples = ps.examples_for_fields(field_set)
    if examples:
        ex_block = "\n\n".join(ex["text"] for ex in examples)  # verbatim, title + body
        parts.append("===WORKED EXAMPLES (ground-truth style, relevant to this group)===\n" + ex_block)

    # --- DO NOT bullets (verbatim), only ones touching this group's fields
    #     + format-only bullets that apply to every group ---
    do_not = ps.do_not_lines_for_fields(field_set)
    if do_not:
        do_not_block = "\n".join(d["text"] for d in do_not)
        parts.append("===DO NOT (relevant to this group)===\n" + do_not_block)

    # --- RAG output contract (new, format-only, not a business rule) ---
    parts.append(RAG_OUTPUT_CONTRACT)
    parts.append(
        f"Return a single JSON object with EXACTLY this shape "
        f"(keys = the {len(field_names)} columns above):\n"
        f"{_expected_json_skeleton(field_names)}"
    )

    return "\n\n".join(parts)


def build_group_user_prompt(entry: dict, group: dict, chunks: list[dict]) -> str:
    """Builds the user-turn content: product identity + retrieved chunks."""
    chunk_blocks = []
    for c in chunks:
        page = c.get("page_number", "N/A")
        chunk_blocks.append(
            f"--- CHUNK {c['chunk_id']} (page {page}, score={c.get('score', 0):.3f}) ---\n"
            f"{c['text']}\n"
            f"--- END CHUNK {c['chunk_id']} ---"
        )
    chunks_text = "\n\n".join(chunk_blocks) if chunk_blocks else "(no relevant chunks retrieved)"

    return (
        f"Product title: {entry.get('title', 'N/A')}\n"
        f"Source file: {entry.get('filename', entry.get('title', 'N/A'))}\n\n"
        f"--- PRODUCT TEXT START (retrieved top-{len(chunks)} chunks for this group) ---\n"
        f"{chunks_text}\n"
        f"--- PRODUCT TEXT END ---\n\n"
        f"Return only one JSON object with the {group['name']} fields as defined above."
    )


def build_group_messages(entry: dict, group: dict, chunks: list[dict],
                          group_prompt_cache: dict | None = None) -> list[dict]:
    """
    Returns chat-format messages ready for tokenizer.apply_chat_template(),
    matching the pattern used by build_prompt() in 02_extract_fields.py.
    """
    if group_prompt_cache is not None and group["key"] in group_prompt_cache:
        system_prompt = group_prompt_cache[group["key"]]
    else:
        system_prompt = build_group_system_prompt(group)
        if group_prompt_cache is not None:
            group_prompt_cache[group["key"]] = system_prompt

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_group_user_prompt(entry, group, chunks)},
    ]
