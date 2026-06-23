"""
Step 2: For every product text file produced by 01_segment_products.py,
use a local Hugging Face model to extract the fixed 41-column schema as JSON.

Configure the model through .env so you can compare different local models:
  HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
  HF_MODEL_CLASS=causal
  HF_LOCAL_FILES_ONLY=true

Resumable: already-extracted products (present in OUT_JSONL) are skipped,
so you can safely re-run after an interruption.
"""
from __future__ import annotations

import json
import os
import re
import time
from importlib.metadata import PackageNotFoundError, version

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from pipeline_config import (
    COLUMNS,
    INDEX_PATH,
    OUT_JSONL,
    PRODUCTS_DIR,
    env_bool,
    env_float,
    env_int,
    load_local_env,
)

load_local_env()

MODEL_NAME = os.environ.get("HF_MODEL_NAME_OR_PATH", "").strip()
MODEL_CLASS = os.environ.get("HF_MODEL_CLASS", "causal").strip().lower()
LOCAL_FILES_ONLY = env_bool("HF_LOCAL_FILES_ONLY", True)
TRUST_REMOTE_CODE = env_bool("HF_TRUST_REMOTE_CODE", False)
MAX_NEW_TOKENS = env_int("HF_MAX_NEW_TOKENS", 1200)
TEMPERATURE = env_float("HF_TEMPERATURE", 0.0)
TOP_P = env_float("HF_TOP_P", 1.0)
REPETITION_PENALTY = env_float("HF_REPETITION_PENALTY", 1.03)
TEXT_CHUNK_SIZE = env_int("TEXT_CHUNK_SIZE", 15000)
LOAD_IN_4BIT = env_bool("HF_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = env_bool("HF_LOAD_IN_8BIT", False)
DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
TORCH_DTYPE = os.environ.get("HF_TORCH_DTYPE", "auto").strip().lower()

SYSTEM_PROMPT = f"""You are a data-extraction engine for a bank product catalogue.
You will be given the raw text of ONE product's section, extracted from a PDF
that merges many Pakistani bank/insurance product brochures (savings accounts,
credit/debit cards, auto/home/personal loans, takaful/insurance plans, mutual
funds, etc).

Extract values for EXACTLY these {len(COLUMNS)} columns and return ONLY a single
JSON object (no markdown fences, no commentary) with these exact keys:
{json.dumps(COLUMNS)}

Rules:
- If a field is not mentioned or not applicable to this product type, set its
  value to the JSON string "N/A" (not null, not empty string).
- Never invent numbers or facts that are not stated or clearly implied in the
  text.
- Keep values short and structured (e.g. "18-60" for an age range field if a
  single field must hold a range, or split into MIN_AGE / MAX_AGE as numbers
  when the schema has separate fields for that, which it does here).
- MIN_AGE / MAX_AGE: numbers only (no "years" suffix), or "N/A".
- MIN_TERM_YEARS / MAX_TERM_YEARS: numbers only, or "N/A".
- LOAN_AMOUNT_RANGE / COVERAGE_AMOUNT / MIN_BALANCE / MIN_INCOME / etc:
  include currency and figure as written, e.g. "PKR 25,000" or
  "500,000 - 1,750,000".
- SPECIAL_CONDITIONS: a brief free-text summary (<=200 chars) of any notable
  conditions not captured elsewhere (e.g. waiting periods, exclusions,
  rollover rules).
- PLAN_TYPE / CUSTOMER_TYPE / SEGMENT etc: use short controlled phrases drawn
  from the document's own wording (e.g. "Salaried", "Self-Employed",
  "Savings Account", "Term Deposit", "Takaful", "Auto Loan", "Credit Card").
- LEAD_CO_MNE: the bank or company offering/underwriting the product (e.g.
  "Bank Alfalah", "IGI Life", "Jubilee Life", "State Life").
"""


def load_index():
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_done_ids():
    done = set()
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


def build_prompt(entry, text, tokenizer):
    user_msg = f"""Product title (from document heading): {entry['title']}
Source file: {entry['file']}

--- PRODUCT TEXT START ---
{text[:TEXT_CHUNK_SIZE]}
--- PRODUCT TEXT END ---

Return the JSON object now."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"{SYSTEM_PROMPT}\n\n{user_msg}"


def parse_json_blob(raw):
    text = raw.strip()
    text = text.strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    model_kwargs = {
        "local_files_only": LOCAL_FILES_ONLY,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if DEVICE_MAP.lower() != "none":
        model_kwargs["device_map"] = DEVICE_MAP

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
    print(f"Model class: {MODEL_CLASS}")
    print(f"Device map: {DEVICE_MAP}")
    print(f"Torch dtype: {TORCH_DTYPE}")
    print(f"4-bit quantization: {LOAD_IN_4BIT}")
    print(f"8-bit quantization: {LOAD_IN_8BIT}")

    if MODEL_CLASS == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **model_kwargs)
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        task = "text-generation"

    generator = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device_map=DEVICE_MAP if DEVICE_MAP.lower() != "none" else None,
    )
    return generator, tokenizer


def extract_one(generator, tokenizer, entry, text):
    prompt = build_prompt(entry, text, tokenizer)
    outputs = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        return_full_text=False if generator.task == "text-generation" else True,
    )

    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        raw = first.get("generated_text") or first.get("summary_text") or first.get("text") or ""
    elif isinstance(outputs, dict):
        raw = outputs.get("generated_text") or outputs.get("summary_text") or outputs.get("text") or ""
    else:
        raw = str(outputs)

    data = parse_json_blob(raw)
    for col in COLUMNS:
        data.setdefault(col, "N/A")
    data["PRODUCT_NAME"] = data.get("PRODUCT_NAME") or entry["title"]
    data["SOURCE_FILE_PRODUCT"] = entry["file"]
    return data


def main():
    try:
        generator, tokenizer = make_generator()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA ran out of memory while loading the model.\n"
            "For Colab T4/L4 GPUs, use these .env settings:\n"
            "HF_LOAD_IN_4BIT=true\n"
            "HF_DEVICE_MAP=auto\n"
            "HF_TORCH_DTYPE=float16\n"
            "You can also use a smaller model like Qwen/Qwen2.5-7B-Instruct."
        ) from exc

    index = load_index()
    done = load_done_ids()
    print(f"{len(index)} products total, {len(done)} already extracted")

    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        for entry in index:
            if entry["product_no"] in done:
                continue
            path = PRODUCTS_DIR / entry["file"]
            with open(path, encoding="utf-8") as f:
                text = f.read()

            for attempt in range(3):
                try:
                    data = extract_one(generator, tokenizer, entry, text)
                    rec = {"product_no": entry["product_no"], **data}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"[{entry['product_no']:03d}/{len(index)}] OK  {entry['title'][:60]}")
                    break
                except Exception as e:
                    print(f"[{entry['product_no']:03d}/{len(index)}] retry {attempt+1}: {e}")
                    time.sleep(3)
            else:
                print(f"[{entry['product_no']:03d}/{len(index)}] FAILED after retries")

    print("Done. Output:", OUT_JSONL)


if __name__ == "__main__":
    main()
