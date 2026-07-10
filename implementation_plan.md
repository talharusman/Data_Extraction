# Fix N/A Output Bug — Hybrid RAG Extraction Pipeline

## Problem Summary

The pipeline runs without errors (prints "OK") but the extracted JSON has nearly every field set to `"N/A"`. Only a handful of fields (`CUSTOMER_SEGMENT`, `MIN_AGE`, `CARD_TYPE`, `CHANNEL`, `MIN_CONTRIBUTION`, `COVERAGE_AMOUNT`, `KEY_BENEFITS`, `FREE_LOOK_PERIOD_DAYS`) get real values — all 40+ other fields come out `"N/A"`.

The user's Colab log shows two `max_new_tokens` truncation warnings:
```
note: generation hit max_new_tokens=830
note: generation hit max_new_tokens=3000
```

## Root Cause Analysis — 4 Bugs Found

---

### Bug 1 (CRITICAL): `group_max_new_tokens()` severely underestimates token budget per group

**File**: [03_extract_fields_hybrid.py](file:///c:/Users/Dell/Desktop/BankAlfala/hybrid_rag_upgrade/03_extract_fields_hybrid.py#L218-L232)

The function estimates `180 + 130 * n_fields` tokens per group. But the RAG_OUTPUT_CONTRACT asks the model to return a **5-key nested object** per field:
```json
{
  "FIELD": {
    "value": "...",
    "confidence": 0.85,
    "evidence": "exact quote from the retrieved text...",
    "chunk_id": 2,
    "page": 1
  }
}
```

The `evidence` field alone is a **verbatim quote** from the document, typically 30-80 tokens. With JSON key names, punctuation, and string content, each field's envelope easily consumes **200-300 tokens**, not 130.

For the Identity group (8 fields), the budget is `180 + 130*8 = 1220`, but the model needs ~2,400+ tokens to return all 8 fields with evidence strings. The output gets truncated mid-JSON, the auto-close repair can only salvage the first few fields, and the rest come back as null → `"N/A"`.

> [!CAUTION]
> This is the #1 cause of the all-N/A problem. The **830-token truncation warning** in the user's log directly proves this — the `group_max_new_tokens=830` ceiling (for a small group like Pricing & Fees with 2 fields: `180 + 130*2 = 440`, capped at min 300) was hit.

**Fix**: Increase per-field estimate from 130 to 250 tokens, and raise the floor from 300 to 500.

---

### Bug 2 (CRITICAL): Retry loop progressively *shrinks* the token budget, making things worse

**File**: [03_extract_fields_hybrid.py](file:///c:/Users/Dell/Desktop/BankAlfala/hybrid_rag_upgrade/03_extract_fields_hybrid.py#L416-L417)

```python
attempt_max_new_tokens = max(400, int(_v2.MAX_NEW_TOKENS * (0.6 ** attempt)))
```

This OOM retry logic was copied from 02's standalone mode where shrinking the budget helps avoid GPU memory errors. But in the hybrid pipeline, **the token budget is already right-sized per group** via `group_max_new_tokens()`. Shrinking `max_new_tokens` on retry:
- Attempt 0: `2200 * 1.0 = 2200` (passed as ceiling to `group_max_new_tokens`)
- Attempt 1: `2200 * 0.6 = 1320`
- Attempt 2: `2200 * 0.36 = 792`

Since `group_max_new_tokens` caps at the `ceiling`, the second and third retries have an even *lower* budget, guaranteeing worse truncation. This retry logic should only shrink the budget for actual OOM errors, not for all exceptions.

**Fix**: Remove the progressive budget reduction. Keep retries for OOM but don't shrink the per-group budget (it's already right-sized). Use a fixed ceiling on all attempts.

---

### Bug 3 (MODERATE): Validation prompt bypasses chat template

**File**: [03_extract_fields_hybrid.py](file:///c:/Users/Dell/Desktop/BankAlfala/hybrid_rag_upgrade/03_extract_fields_hybrid.py#L314-L318)

```python
validation_prompt = build_lean_validation_prompt(entry, normalized)
validation_raw = _v2.get_raw_generation(
    model, tokenizer, validation_prompt,
    max_new_tokens=max(max_new_tokens, RAG_VALIDATION_MAX_NEW_TOKENS),
)
```

`build_lean_validation_prompt()` returns a plain string. `get_raw_generation()` tokenizes this raw string directly without applying the chat template. But for Qwen2.5-Instruct models, the chat template (`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`) is **essential** for the model to understand it should produce a JSON response.

Without the chat template, Qwen2.5 may generate random continuations, think tokens, or malformed output — which then either fails JSON parsing (falling back to `normalized` — acceptable) or produces a truncated/partial JSON that the merge-on-top logic (line 338-341) can actually make things *worse* by overwriting good values with bad ones.

**Fix**: Apply `tokenizer.apply_chat_template()` to the validation prompt before passing it to `get_raw_generation()`, matching how group extraction prompts are handled via `make_generate_fn()`.

---

### Bug 4 (MINOR): Validation `max_new_tokens` is inflated by `max(max_new_tokens, 3000)`

**File**: [03_extract_fields_hybrid.py](file:///c:/Users/Dell/Desktop/BankAlfala/hybrid_rag_upgrade/03_extract_fields_hybrid.py#L317)

```python
max_new_tokens=max(max_new_tokens, RAG_VALIDATION_MAX_NEW_TOKENS),
```

`max_new_tokens` here is the *per-group ceiling* (2200), and `RAG_VALIDATION_MAX_NEW_TOKENS` is 3000. So the validation pass always uses 3000 tokens. On a 4-bit 3B model on Colab, requesting 3000 new tokens with the full 56-field JSON already in the prompt input burns significant GPU memory, and the 3000-token truncation warning in the log confirms it's still not enough. However, the validation pass asking for 56 fields with full values genuinely needs this budget. The real fix is Bug 3 (chat template) — with a properly framed prompt, the model will emit valid JSON faster.

---

## Proposed Changes

### [MODIFY] [03_extract_fields_hybrid.py](file:///c:/Users/Dell/Desktop/BankAlfala/hybrid_rag_upgrade/03_extract_fields_hybrid.py)

#### Change 1: Fix `group_max_new_tokens()` — increase per-field token estimate

```diff
 def group_max_new_tokens(group: dict, ceiling: int) -> int:
     n_fields = len(group["field_order"])
-    budget = 180 + 130 * n_fields
-    return max(300, min(budget, ceiling))
+    budget = 250 + 250 * n_fields
+    return max(500, min(budget, ceiling))
```

Rationale: The RAG envelope requires ~200-300 tokens per field (value + confidence float + evidence quote + chunk_id + page + JSON punctuation). 250 tokens/field is conservative. Floor of 500 ensures even 2-field groups have room for evidence strings.

#### Change 2: Fix retry loop — don't shrink token budget progressively

```diff
             for attempt in range(3):
-                attempt_max_new_tokens = max(400, int(_v2.MAX_NEW_TOKENS * (0.6 ** attempt)))
+                attempt_max_new_tokens = _v2.MAX_NEW_TOKENS
                 try:
```

The progressive shrinking was a holdover from 02's monolithic pipeline where reducing output size helped avoid OOM. In the hybrid pipeline, group budgets are already right-sized — shrinking the ceiling only causes more truncation.

#### Change 3: Fix validation prompt — apply chat template

Wrap the validation prompt in proper chat-template messages before calling `get_raw_generation()`:

```diff
         validation_prompt = build_lean_validation_prompt(entry, normalized)
-        validation_raw = _v2.get_raw_generation(
-            model, tokenizer, validation_prompt,
-            max_new_tokens=max(max_new_tokens, RAG_VALIDATION_MAX_NEW_TOKENS),
-        )
+        validation_messages = [
+            {"role": "system", "content": "You are a JSON validation assistant. Return only valid JSON, no explanations."},
+            {"role": "user", "content": validation_prompt},
+        ]
+        try:
+            try:
+                templated_prompt = tokenizer.apply_chat_template(
+                    validation_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
+                )
+            except TypeError:
+                templated_prompt = tokenizer.apply_chat_template(
+                    validation_messages, tokenize=False, add_generation_prompt=True,
+                )
+        except Exception:
+            templated_prompt = validation_prompt
+
+        validation_raw = _v2.get_raw_generation(
+            model, tokenizer, templated_prompt,
+            max_new_tokens=RAG_VALIDATION_MAX_NEW_TOKENS,
+        )
```

---

## Summary of Impact

| Bug | Severity | Impact | Fix |
|-----|----------|--------|-----|
| #1 Token budget | **CRITICAL** | Groups truncated → most fields null → N/A | Raise per-field estimate 130→250, floor 300→500 |
| #2 Retry shrink | **CRITICAL** | Retries make extraction worse not better | Use fixed ceiling, no progressive reduction |
| #3 Chat template | **MODERATE** | Validation produces bad/partial JSON | Apply `apply_chat_template()` to validation |
| #4 Max tokens | **MINOR** | Validation prompt slightly oversized | Use `RAG_VALIDATION_MAX_NEW_TOKENS` directly |

## Verification Plan

### Manual Verification
- User runs the fixed code in Colab against the same Alfalah Insurance Zaamin Takaful Plan document
- Check that the output JSONL has substantially fewer N/A fields
- Verify the `max_new_tokens` truncation warnings are reduced or eliminated
