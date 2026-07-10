"""
Step 2 (Hybrid RAG upgrade): Extract the fixed 56-column schema using
token chunking + FAISS/BM25 hybrid retrieval + field-group extraction +
confidence-based merge, instead of the old "send every char-chunk, ask for
all 56 fields, first-non-null-wins" approach in 02_extract_fields.py.

THIS SCRIPT DOES NOT REPLACE 02_extract_fields.py.
It REUSES it: model loading (make_generator), file reading
(read_file_text), input selection (select_input_files,
build_index_from_files), resumability (load_done_ids), normalization
(normalize_record), validation (build_validation_prompt), and the exact
same JSON/JSONL output schema all come straight from 02_extract_fields.py
via import, per the "preserve backward compatibility" requirement.

What's new here (see hybrid_rag/ package):
    - Token-based chunking (700-1000 tokens, 100-200 overlap) with
      chunk_id/page_number/token_count metadata, instead of character
      chunking.
    - Embeddings generated ONCE per document and cached, reused across
      every field group's retrieval.
    - FAISS vector index + BM25 keyword index, built once per document.
    - Hybrid retrieval (FAISS + BM25, merged/deduped/ranked) per field
      group, top_k=4 by default.
    - Field GROUPS (9 groups covering all 55 LLM-extracted fields; the
      56th, SOURCE_FILE_PRODUCT, is filled deterministically) computed
      ONE TIME in hybrid_rag/field_groups.py, never at runtime.
    - One LLM call per field group (only that group's fields + only its
      top-K retrieved chunks), instead of one call per char-chunk asking
      for all 56 fields -- fewer, smaller, more focused prompts.
    - Confidence + evidence + chunk_id + page returned per field
      internally; merge strategy (confidence > evidence > table-preferred
      > majority agreement > recency) picks the final value per field,
      replacing "first non-null wins".
    - Per-group + one final full-record validation pass (the final pass
      reuses 02_extract_fields.py's existing validation prompt verbatim).
    - The per-document FAISS/BM25/embedding index is torn down
      (DocumentIndex.clear()) after each product file finishes, before
      the next document's index is built.
    - Debug Mode (EXTRACTION_DEBUG_MODE=true in .env) additionally writes
      confidence/evidence/chunk_id/page per field to a sidecar
      *.debug.jsonl file; the primary OUT_JSONL schema is unchanged.

Configure via .env (in addition to the existing HF_* variables consumed by
02_extract_fields.py):
    RAG_TARGET_TOKENS=850          # 700-1000
    RAG_OVERLAP_TOKENS=150         # 100-200
    RAG_TOP_K=4                    # chunks retrieved per field group
    RAG_ALPHA=0.5                  # FAISS weight vs BM25 (0-1)
    RAG_ENSEMBLE_PASSES=1          # >1 enables genuine majority-vote merge
    EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    EXTRACTION_DEBUG_MODE=false
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Reuse EVERYTHING from the existing, working script instead of
# reimplementing it. This is the "incrementally improve, don't rewrite"
# requirement -- model loading, I/O, normalization, and validation are all
# imported, not copied.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "extract_fields_v2", str(Path(__file__).parent / "02_extract_fields.py")
)
_v2 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v2)  # noqa: this runs 02's module-level setup (loads SYSTEM_PROMPT, env, etc.)

from pipeline_config import COLUMNS, OUT_JSONL, env_bool, env_float, env_int, load_local_env  # noqa: E402

from hybrid_rag import field_groups as fg  # noqa: E402
from hybrid_rag.token_chunking import chunk_document  # noqa: E402
from hybrid_rag.embeddings_index import DocumentIndex, make_default_embedder  # noqa: E402
from hybrid_rag.retrieval import retrieve_for_group  # noqa: E402
from hybrid_rag.group_extraction import extract_group  # noqa: E402
from hybrid_rag.merge_strategy import (  # noqa: E402
    merge_group_results,
    candidates_to_flat_record,
    evidence_debug_view,
)

load_local_env()

RAG_TARGET_TOKENS = env_int("RAG_TARGET_TOKENS", 850)
RAG_OVERLAP_TOKENS = env_int("RAG_OVERLAP_TOKENS", 150)
RAG_TOP_K = env_int("RAG_TOP_K", 4)
RAG_ALPHA = env_float("RAG_ALPHA", 0.5)
RAG_ENSEMBLE_PASSES = env_int("RAG_ENSEMBLE_PASSES", 1)
DEBUG_MODE = env_bool("EXTRACTION_DEBUG_MODE", False)
DEBUG_JSONL = OUT_JSONL.with_suffix(".debug.jsonl")

# The validation pass must reproduce ALL 56 fields verbatim in one shot
# (unlike a single field group, which only needs a handful). Summing this
# project's own field_max_lengths gives ~4,380 chars of possible field
# *content* alone, before field names/JSON punctuation -- reusing a
# group-sized or even the plain extraction MAX_NEW_TOKENS budget here was
# what caused validation output to truncate and (before the merge-on-top
# fix above) wipe out already-good fields. Give it real headroom,
# independent of whatever MAX_NEW_TOKENS is set to elsewhere.
# INCREASED from 3000 → 4500: validation was still truncating at 3000
# tokens because the model occasionally emits reasoning preamble before
# the JSON, eating into the output budget.
RAG_VALIDATION_MAX_NEW_TOKENS = env_int("RAG_VALIDATION_MAX_NEW_TOKENS", 4500)

# Fail fast at startup if the schema and the one-time field grouping have
# drifted apart (per field_groups.py's own docstring: re-run/edit that file
# by hand if COLUMNS changes -- never regroup at runtime).
fg.validate_against_columns(list(COLUMNS))

# Build each group's system-prompt text exactly once per process (dedupes
# prompt text across every chunk/document that uses this group).
_GROUP_PROMPT_CACHE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Generation wrapper: adapts 02_extract_fields.py's get_raw_generation() +
# tokenizer chat templating into the generate_fn(messages) -> str callable
# that hybrid_rag.group_extraction.extract_group expects.
# ---------------------------------------------------------------------------

def build_lean_validation_prompt(entry, extracted_json: dict) -> str:
    """
    Same validation checklist as 02_extract_fields.py's
    build_validation_prompt(), WITHOUT prepending the full
    EXTRACTION_SYSTEM_PROMPT.txt contents (~8-9K chars covering rules for
    all 56 fields).

    That prepend made sense in the old pipeline, where a single monolithic
    prompt was the only source of truth the validator had ever seen. In the
    hybrid pipeline, every field was already extracted against its own
    focused group prompt (hybrid_rag/base_prompt.py + field_groups.py), so
    re-sending the entire 56-field ruleset just to validate a JSON blob is
    redundant context that inflates the input length -- and therefore
    activation/VRAM usage -- of every single document's validation call for
    no accuracy benefit. This keeps the same checklist, just without the
    redundant prefix.
    """
    validation_instructions = f"""You extracted this JSON. Validate and fix any issues:

EXTRACTED JSON:
{json.dumps(extracted_json)}

VALIDATION RULES -- check each and FIX if violated:

1. PRODUCT_NAME: Title Case, never ALL CAPS.
2. PLAN_TYPE: for any insurer/takaful-underwritten product, contains the
   word "Insurance" (optionally with a short qualifier). For a bank-only
   product: one of Deposit|Loan|Card|Service|Loyalty|Investment|Savings.
3. TARGET_GOAL: standardized short value (Protection, Savings, Education,
   Health, Marriage, etc.).
4. CUSTOMER_TYPE: only Salaried|Self-Employed|SME|Corporate|Retail|
   Government, " | "-joined if multiple. No free text/bank names.
5. CUSTOMER_SEGMENT / TARGET_SEGMENT / SEGMENT_TIER: preserve any existing
   non-N/A value; only set "N/A" if currently empty/null. Do not move
   ELIGIBILITY_TYPE content into these fields or vice versa.
6. CHANNEL: "Bank Branch" if branches mentioned.
7. ELIGIBILITY_TYPE: concise operational eligibility (age, CNIC, one-policy
   rules) -- not a marketing/positioning statement.
8. GENDER: exactly one of Male|Female|All|N/A. "All" only if eligibility
   info is present and no gender restriction is stated.
9. FINANCING_TYPE: "Unit Linked" for PIA/fund-allocation plans; "Hybrid
   (Bonus Based and Unit Linked)" for bonus+unit-linked; never "N/A" when
   unit-linked structure is explicitly mentioned.
10. DEPOSIT_PROFIT_TYPE / DEPOSIT_PROFIT_FREQUENCY: both "N/A" for
    unit-linked or health/protection plans. Never "At Maturity" for
    unit-linked plans.
11. TENURE_OPTIONS: plan DURATION choices only (e.g. "5 | 10 | 15 | 20
    years"); payment-frequency words (Annual/Quarterly/Monthly/
    Semi-Annual) belong in PREMIUM_PAYMENT_FREQUENCY instead.
12. PREMIUM_PAYMENT_FREQUENCY: how the customer pays, " | "-joined if
    multiple, else "N/A".
13. Every multi-value field uses " | " as separator (never ";" or "," as a
    list separator).
14. Numeric fields (MIN_AGE, MAX_AGE, MIN_TERM_YEARS, MAX_TERM_YEARS,
    FREE_LOOK_PERIOD_DAYS, IS_BANK_OFFERED): plain integers only, no PKR/
    commas/.0. Tiered fields (MIN_BALANCE, MIN_INCOME, MIN_INVESTMENT,
    MIN_CONTRIBUTION) may keep a genuine tiered table.
15. FREE_LOOK_PERIOD_DAYS: only if explicitly stated; never default to 14.
16. MIN_CONTRIBUTION vs PRICING_RATE vs MIN_INVESTMENT: tiered premium/
    contribution amounts belong in MIN_CONTRIBUTION, not PRICING_RATE or
    MIN_INVESTMENT. For IBG products, rescue a contribution-style amount
    sitting in MIN_INVESTMENT into MIN_CONTRIBUTION.
17. OPTIONAL_RIDERS vs KEY_BENEFITS: OPTIONAL_RIDERS only for
    explicitly-labeled optional add-ons, not core/default benefits.
18. SOURCE_FILE_PRODUCT: filename only, no path.
19. All 56 fields present, no null/None/NaN/empty string -- use "N/A".

If ANY rule is violated, return CORRECTED JSON. Otherwise return JSON unchanged.
Fix ONLY the violations, preserve everything else.
Return ONLY valid JSON, no explanations."""

    return validation_instructions


def make_generate_fn(model, tokenizer, max_new_tokens):
    def _generate(messages: list[dict]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            except Exception:
                prompt = "\n\n".join(m["content"] for m in messages)
        else:
            prompt = "\n\n".join(m["content"] for m in messages)
        return _v2.get_raw_generation(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

    return _generate


def group_max_new_tokens(group: dict, ceiling: int) -> int:
    """
    Right-sizes the token budget per field group instead of reusing the
    full 56-field ceiling (e.g. 2200) for every one of the 9 group calls.
    Groups only ask for a handful of fields, each wrapped in the RAG
    {value, confidence, evidence, chunk_id, page} envelope -- ~250 tokens
    covers a typical field's worth of JSON + evidence quote (the evidence
    string alone can be 30-80 tokens, plus key names, confidence float,
    chunk_id, page, and JSON punctuation). The fixed overhead covers the
    JSON skeleton (outer braces, field separators). Capped at `ceiling`
    so this never asks for MORE than the original budget, only less where
    it's safe to.

    BUG FIX: the original 130-token estimate was far too low for the RAG
    envelope, causing truncation on most groups (the 830-token truncation
    warning in logs proved this). Raised to 250 tokens/field and floor
    from 300 to 500.
    """
    n_fields = len(group["field_order"])
    budget = 250 + 250 * n_fields
    return max(500, min(budget, ceiling))


# ---------------------------------------------------------------------------
# Page-aware text extraction (best-effort). Falls back to 02's
# read_file_text() (single blob, page 1) for formats without native pages.
# ---------------------------------------------------------------------------

def read_pages(abs_path: Path) -> tuple[str, list[str] | None]:
    """
    Returns (full_text, page_texts_or_None).

    Uses PyMuPDF (fitz) for page-level PDF text, matching the project's
    existing requirements.txt (pymupdf) instead of introducing a second
    PDF library. Falls back to 02_extract_fields.py's read_file_text()
    (single blob, no page numbers) for non-PDF formats or if PyMuPDF fails.
    """
    if abs_path.suffix.lower() == ".pdf":
        try:
            # pyrefly: ignore [missing-import]
            import fitz  # PyMuPDF
            pages = []
            with fitz.open(abs_path) as doc:
                for page in doc:
                    pages.append(page.get_text() or "")
            full_text = "\n".join(pages)
            return full_text, pages
        except Exception:
            pass
    full_text = _v2.read_file_text(abs_path)
    return full_text, None


# ---------------------------------------------------------------------------
# Core per-document extraction
# ---------------------------------------------------------------------------

def extract_one_product_hybrid(model, tokenizer, entry, text, page_texts, max_new_tokens, embedder) -> dict:
    """
    Token-chunk -> embed once -> FAISS+BM25 index -> per-group hybrid
    retrieval + extraction -> confidence-based merge -> final validation.
    """
    chunks = chunk_document(
        text,
        hf_tokenizer=tokenizer,
        target_tokens=RAG_TARGET_TOKENS,
        overlap_tokens=RAG_OVERLAP_TOKENS,
        page_texts=page_texts,
    )
    print(f"  \u2500\u2500 [{entry['title'][:55]}] \u2500\u2500")
    print(f"  \u2192 {len(chunks)} token-chunks created (target={RAG_TARGET_TOKENS} tok, overlap={RAG_OVERLAP_TOKENS} tok)")

    doc_index = DocumentIndex(embedder=embedder)
    try:
        doc_index.build(chunks)  # embeddings + FAISS + BM25 built ONCE here
        print(f"  \u2192 FAISS + BM25 index built. Starting {len(fg.FIELD_GROUPS)}-group extraction...\n")

        all_group_results: dict[str, dict[str, dict]] = {}
        n_groups = len(fg.FIELD_GROUPS)
        for grp_idx, group in enumerate(fg.FIELD_GROUPS):
            budget = group_max_new_tokens(group, max_new_tokens)
            print(f"  \u25b6 [{grp_idx+1}/{n_groups}] {group['name']} "
                  f"({len(group['field_order'])} fields, budget={budget} tokens, top_k={RAG_TOP_K})")

            retrieved = retrieve_for_group(doc_index, group, top_k=RAG_TOP_K, alpha=RAG_ALPHA)

            # For the Identity group always include the very first chunk
            # (document header/title page) so PRODUCT_NAME, PLAN_TYPE, and
            # PROVIDER_NAME are never missed due to retrieval scoring.
            if group["key"] == "identity" and chunks:
                first = dict(chunks[0])
                first.setdefault("score", 1.0)
                if not any(c["chunk_id"] == first["chunk_id"] for c in retrieved):
                    retrieved = [first] + retrieved[:RAG_TOP_K - 1]
                    print(f"    + Injected chunk #1 (document header) into identity retrieval")

            # Right-sized per-group budget (see group_max_new_tokens).
            generate_fn = make_generate_fn(model, tokenizer, budget)

            passes = []
            for _ in range(max(1, RAG_ENSEMBLE_PASSES)):
                passes.append(
                    extract_group(generate_fn, entry, group, retrieved, group_prompt_cache=_GROUP_PROMPT_CACHE)
                )
                _v2.free_gpu_memory()

            all_group_results[group["key"]] = merge_group_results(passes)

        flat_record = candidates_to_flat_record(all_group_results, default_value=_v2.DEFAULT_VALUE)

        # Deterministic field (never asked of the LLM).
        flat_record["SOURCE_FILE_PRODUCT"] = _v2.get_source_filename(entry)

        # --- Extraction summary before validation ---
        n_extracted = sum(1 for v in flat_record.values() if v != _v2.DEFAULT_VALUE)
        print(f"\n  \u2500\u2500 Extraction summary: {n_extracted}/{len(flat_record)} fields populated (non-N/A) \u2500\u2500")

        # Merged-record normalization pass (reused verbatim from 02).
        normalized = _v2.normalize_record(flat_record, entry)

        # Final single full-record validation pass.
        print(f"  \u25b6 Validation pass (budget={RAG_VALIDATION_MAX_NEW_TOKENS} tokens)...")
        validation_prompt = build_lean_validation_prompt(entry, normalized)

        # BUG FIX: the validation prompt was passed as a raw string to
        # get_raw_generation() WITHOUT applying the chat template. For
        # Qwen2.5-Instruct, the chat template (<|im_start|>system/user/
        # assistant<|im_end|>) is essential for the model to understand it
        # should produce a JSON response. Without it, the model generates
        # random continuations or malformed output, which either fails JSON
        # parsing (acceptable — falls back to `normalized`) or produces a
        # truncated/partial JSON whose merge-on-top logic can overwrite good
        # values with bad ones.
        validation_messages = [
            {"role": "system", "content": "You are a JSON validation assistant. Return only valid JSON, no explanations."},
            {"role": "user", "content": validation_prompt},
        ]
        try:
            try:
                templated_prompt = tokenizer.apply_chat_template(
                    validation_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                templated_prompt = tokenizer.apply_chat_template(
                    validation_messages, tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            templated_prompt = validation_prompt

        validation_raw = _v2.get_raw_generation(
            model, tokenizer, templated_prompt,
            max_new_tokens=RAG_VALIDATION_MAX_NEW_TOKENS,
        )

        # BUG FIX: the validation call asks the model to re-emit the FULL
        # 56-field JSON. If that output is truncated (hits max_new_tokens)
        # or partially malformed, _v2.normalize_record() used to fill in
        # DEFAULT_VALUE ("N/A") for every field missing from the truncated
        # output -- silently wiping out fields that were already correctly
        # extracted in `normalized`, even though the model never intended to
        # clear them (it just never got that far before being cut off). That
        # was the actual cause of rows coming back with "no data extracted."
        #
        # Fix: treat validated_parsed as a set of CORRECTIONS layered on top
        # of `normalized`, not a wholesale replacement. Any field the
        # validation pass didn't return (or returned empty) falls back to
        # the value `normalized` already had, instead of "N/A".
        try:
            validated_parsed = _v2.parse_json_blob(validation_raw)
            if not isinstance(validated_parsed, dict):
                raise ValueError("validation output was not a JSON object")
            merged_after_validation = dict(normalized)
            corrections = 0
            for col in COLUMNS:
                v = validated_parsed.get(col)
                if v not in (None, "", "null", "None"):
                    if v != normalized.get(col):
                        corrections += 1
                    merged_after_validation[col] = v
                # else: keep normalized[col] -- field wasn't reached/returned.
            final_record = _v2.normalize_record(merged_after_validation, entry)
            print(f"  \u2713 Validation applied ({corrections} field(s) corrected)")
        except Exception as e:
            print(f"  \u2717 Validation failed ({e}) \u2014 keeping pre-validation extraction.")
            final_record = normalized

        if DEBUG_MODE:
            _write_debug_record(entry, all_group_results)

        return final_record
    finally:
        # REQUIRED: drop FAISS index, BM25 index, chunk store, and cached
        # embeddings for this document before moving to the next one.
        doc_index.clear()


def _write_debug_record(entry, all_group_results) -> None:
    debug_rec = {
        "product_no": entry["product_no"],
        "title": entry["title"],
        "fields": evidence_debug_view(all_group_results),
    }
    with open(DEBUG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(debug_rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main (mirrors 02_extract_fields.py's main() loop/retry/OOM handling)
# ---------------------------------------------------------------------------

def main():
    input_files = _v2.select_input_files()
    index = _v2.build_index_from_files(input_files)

    print(f"\n  {len(index)} file(s) queued for extraction (Hybrid RAG mode).")
    print(f"  Output will be appended to: {OUT_JSONL}")
    if DEBUG_MODE:
        print(f"  Debug evidence/confidence will be appended to: {DEBUG_JSONL}")
    print()

    try:
        model, tokenizer = _v2.make_generator()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA ran out of memory while loading the model.\n"
            "For Colab T4/L4 GPUs, use these .env settings:\n"
            "HF_LOAD_IN_4BIT=true\nHF_DEVICE_MAP=auto\nHF_TORCH_DTYPE=float16\nHF_GPU_RESERVE_GIB=3.0\n"
        ) from exc

    # Embedder resolved once and reused across documents (the model itself
    # isn't per-document state -- only the FAISS/BM25 indexes built from it
    # are, and those go through DocumentIndex.clear() per document).
    embedder = make_default_embedder()

    done = _v2.load_done_ids()
    print(f"{len(index)} products total, {len(done)} already extracted")

    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        for entry in index:
            if entry["product_no"] in done:
                continue

            abs_path: Path = entry["_abs_path"]
            try:
                text, page_texts = read_pages(abs_path)
            except Exception as read_err:
                print(f"[{entry['product_no']:03d}/{len(index)}] SKIP  Could not read '{abs_path.name}': {read_err}")
                continue

            if not text.strip():
                print(f"[{entry['product_no']:03d}/{len(index)}] SKIP  '{abs_path.name}' produced no text.")
                continue

            for attempt in range(3):
                # BUG FIX: the old code progressively shrank the token budget
                # on each retry (2200 → 1320 → 792), which was a holdover
                # from 02's monolithic pipeline where reducing output size
                # helped avoid OOM. In the hybrid pipeline, group budgets are
                # already right-sized by group_max_new_tokens() — shrinking
                # the ceiling only causes MORE truncation and WORSE results
                # on each retry. Use the full budget on every attempt.
                attempt_max_new_tokens = _v2.MAX_NEW_TOKENS
                try:
                    data = extract_one_product_hybrid(
                        model, tokenizer, entry, text, page_texts,
                        max_new_tokens=attempt_max_new_tokens, embedder=embedder,
                    )
                    rec = {"product_no": entry["product_no"], **data}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"[{entry['product_no']:03d}/{len(index)}] OK  {entry['title'][:60]}  ({abs_path.name})")
                    break
                except torch.cuda.OutOfMemoryError as e:
                    _v2.free_gpu_memory()
                    print(f"[{entry['product_no']:03d}/{len(index)}] CUDA OOM attempt {attempt + 1}, retrying: {e}")
                    time.sleep(5)
                except Exception as e:
                    _v2.free_gpu_memory()
                    print(f"[{entry['product_no']:03d}/{len(index)}] retry {attempt + 1}: {e}")
                    time.sleep(3)
            else:
                print(f"[{entry['product_no']:03d}/{len(index)}] FAILED after retries")

            _v2.free_gpu_memory()

    print("Done. Output:", OUT_JSONL)


if __name__ == "__main__":
    main()