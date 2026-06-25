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

import ast
import json
import os
import re
import time
from importlib.metadata import PackageNotFoundError, version

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
DEFAULT_VALUE = "N/A"
NUMERIC_COLUMNS = {
    "MIN_AGE",
    "MAX_AGE",
    "BANK_CUSTOMER",
    "MIN_BALANCE",
    "AVG_BALANCE_REQUIREMENT",
    "MIN_INCOME",
    "MIN_INCOME_USD",
    "MIN_INVESTMENT",
    "MIN_CONTRIBUTION",
    "MIN_TERM_YEARS",
    "MAX_TERM_YEARS",
}

SYSTEM_PROMPT = f"""You are a data-extraction engine for a bank product catalogue.
You will be given the raw text of ONE product's section, extracted from a PDF
that merges many Pakistani bank/insurance product brochures (savings accounts,
credit/debit cards, auto/home/personal loans, takaful/insurance plans, mutual
funds, etc).

Extract values for EXACTLY these {len(COLUMNS)} columns and return ONLY a single
JSON object (no markdown fences, no commentary) with these exact keys:
{json.dumps(COLUMNS)}

Use the following data dictionary to interpret each column:

PRODUCT_NAME: Normalized product or service name. Example: "Alfalah Car Ijarah". Type: text/categorical.
LEAD_CO_MNE: Lead marker from user (IBG or BNK). Example: "IBG". Type: text/categorical.
SOURCE_FILE_PRODUCT: Batch/source group name provided by user. Example: "Isl Consumer". Type: text/categorical.
PLAN_TYPE: High-level family such as Loan, Deposit, Savings, Card, Investment, Insurance, Service or Loyalty. Example: "Loan". Type: text/categorical.
TARGET_GOAL: Primary customer need/use case. Example: "Housing". Type: text/categorical.
CUSTOMER_TYPE: Broad eligible customer class. Example: "SME / Corporate". Type: text/categorical.
EMPLOYMENT_TYPE: Employment eligibility where applicable. Example: "Salaried/SEP". Type: text/categorical.
CUSTOMER_SEGMENT: Behavioral/demographic segment when explicitly identified. Example: "NRP". Type: text/categorical.
TARGET_SEGMENT: More specific positioning segment where stated. Example: "Financial Inclusion". Type: text/categorical.
SEGMENT: Priority/HNW/premium segment tag. Example: "Premium (HNW)". Type: text/categorical.
MIN_AGE: Minimum eligible age. Example: 18. Type: numeric.
MAX_AGE: Maximum eligible age. Example: 65. Type: numeric.
GENDER: Gender eligibility/focus. Example: "Female". Type: text/categorical.
BANK_CUSTOMER: 1 for bank-offered/relationship products in this normalized dataset. Example: 1. Type: numeric.
ACCOUNT_TYPE: Current, Savings, Wallet, Digital, etc. Example: "Current". Type: text/categorical.
CARD_TYPE: Debit, Credit, Virtual Debit, Premium etc. Example: "Credit (Premium)". Type: text/categorical.
CHANNEL: Primary servicing/onboarding channel. Example: "Mobile App". Type: text/categorical.
ELIGIBILITY_TYPE: Short eligibility logic text. Example: "CNIC + Biometric". Type: text/categorical.
SERVICE_TYPE: Operational/service type for non-core products. Example: "Payments". Type: text/categorical.
REWARD_TYPE: Reward mechanism for loyalty rows. Example: "Points". Type: text/categorical.
CURRENCY: Currency or set of currencies. Example: "PKR + FCY". Type: text/categorical.
CURRENCY_TYPE: PKR/FCY style label if used. Example: "FCY". Type: text/categorical.
MIN_BALANCE: Minimum opening or operating balance. Example: 1000. Type: numeric.
AVG_BALANCE_REQUIREMENT: Average balance requirement. Example: 50000. Type: numeric.
MIN_INCOME: Minimum PKR income where stated. Example: 50000. Type: numeric.
MIN_INCOME_USD: Minimum USD income where stated. Example: 3000. Type: numeric.
MIN_INVESTMENT: Minimum investment/placement amount. Example: 100K. Type: numeric.
MIN_CONTRIBUTION: Minimum premium/contribution. Example: 250000. Type: numeric.
LOAN_AMOUNT_RANGE: Loan size or facility range. Example: 200K-3M. Type: text/categorical.
COVERAGE_AMOUNT: Coverage amount or insured amount. Example: 50K-150K coverage. Type: text/categorical.
FINANCING_TYPE: Conventional/Islamic financing structure or instrument type. Example: "Mudarabah". Type: text/categorical.
PROFIT_TYPE: Profit basis or mode. Example: "Tier-based". Type: text/categorical.
PROFIT_FREQUENCY: Monthly, semi-annual, maturity etc. Example: "Monthly". Type: text/categorical.
TENURE: Readable tenor text. Example: "1-5 years". Type: text/categorical.
TENURE_OPTIONS: Structured tenor menu text. Example: "1M -> 5Y". Type: text/categorical.
MIN_TERM_YEARS: Minimum term in years when directly available. Example: 10. Type: numeric.
MAX_TERM_YEARS: Maximum term in years when directly available. Example: 25. Type: numeric.
BUSINESS_TENURE: Required business age/operating history. Example: ">=3 years". Type: text/categorical.
COLLATERAL_TYPE: Security/collateral type. Example: "Property Mortgage". Type: text/categorical.
EQUITY_REQUIREMENT: Borrower equity or margin requirement. Example: "30%". Type: text/categorical.
DBR_LIMIT: Debt burden ratio limit. Example: "<=40%". Type: text/categorical.
TRANSACTION_LIMIT: Usage/balance/transaction cap. Example: "1M monthly". Type: text/categorical.
SPECIAL_CONDITIONS: Residual qualifiers or important caveats. Example: "RDA required". Type: text/categorical.

Rules:
- If a field is not mentioned or not applicable to this product type, set its
  value to the JSON string "N/A" (not null, not empty string).
- Prefer copying the source wording when a field is categorical.
- For numeric fields, return only the number when possible.
- For fields like LOAN_AMOUNT_RANGE, COVERAGE_AMOUNT, TENURE, and SPECIAL_CONDITIONS,
  keep the wording compact but faithful to the source text.
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


def blank_record(entry):
    record = {col: DEFAULT_VALUE for col in COLUMNS}
    record["PRODUCT_NAME"] = entry["title"]
    record["SOURCE_FILE_PRODUCT"] = entry["file"]
    return record


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
    seen = set()

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


def parse_json_blob(raw):
    for candidate in _iter_json_candidates(raw):
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = _parse_candidate(cleaned)
        if parsed is not None:
            return parsed
    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


def normalize_record(record, entry):
    if not isinstance(record, dict):
        return blank_record(entry)

    normalized = {col: DEFAULT_VALUE for col in COLUMNS}
    for col in COLUMNS:
        value = record.get(col, DEFAULT_VALUE)
        if value in (None, "", []):
            value = DEFAULT_VALUE
        if col in NUMERIC_COLUMNS and isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.upper() == DEFAULT_VALUE:
                value = DEFAULT_VALUE
        normalized[col] = value

    normalized["PRODUCT_NAME"] = record.get("PRODUCT_NAME") or entry["title"]
    normalized["SOURCE_FILE_PRODUCT"] = entry["file"]
    return normalized


def build_prompt(entry, text, tokenizer):
    user_msg = f"""Product title (from document heading): {entry['title']}
Source file: {entry['file']}

--- PRODUCT TEXT START ---
{text[:TEXT_CHUNK_SIZE]}
--- PRODUCT TEXT END ---

Return the JSON object now.

Important:
- Your reply must begin with `{{` and end with `}}`.
- Do not output any reasoning, analysis, or <think> blocks.
- Do not wrap the JSON in markdown fences.
- Output only one valid JSON object."""

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
    return f"""You are repairing a failed extraction.

Return exactly one valid JSON object and nothing else.
The JSON must use these keys in any order:
{json.dumps(COLUMNS)}

Rules:
- Use double quotes for all strings.
- Use "N/A" for missing or unknown fields.
- Keep numeric fields as numbers when the source gives a number.
- Do not add markdown fences, explanations, or analysis.

Product title: {entry['title']}
Source file: {entry['file']}

Broken model output to repair:
{raw_text[:5000]}
"""


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

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def extract_with_repair(model, tokenizer, entry, text):
    prompt = build_prompt(entry, text, tokenizer)
    raw = get_raw_generation(model, tokenizer, prompt)

    try:
        return normalize_record(parse_json_blob(raw), entry)
    except Exception as first_error:
        repair_prompt = build_repair_prompt(entry, raw)
        repaired_raw = get_raw_generation(model, tokenizer, repair_prompt)
        try:
            return normalize_record(parse_json_blob(repaired_raw), entry)
        except Exception as second_error:
            print(
                f"Warning: falling back to blank record for {entry['product_no']:03d} "
                f"after parse failures: {first_error}; {second_error}"
            )
            return blank_record(entry)


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


def extract_one(model, tokenizer, entry, text):
    return extract_with_repair(model, tokenizer, entry, text)


def main():
    try:
        model, tokenizer = make_generator()
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
                    data = extract_one(model, tokenizer, entry, text)
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
