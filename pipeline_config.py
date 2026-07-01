from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_local_env(dotenv_path: Path | None = None) -> None:
    """
    Load a simple .env file from the project root without an extra dependency.

    Existing environment variables always win.
    """
    path = dotenv_path or (BASE_DIR / ".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


def resolve_path(env_name: str, default: str | Path) -> Path:
    raw = os.environ.get(env_name)
    path = Path(raw) if raw else Path(default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


COLUMNS = [
    "PRODUCT_NAME", "LEAD_MARKER", "SOURCE_FILE_PRODUCT", "PLAN_TYPE",
    "TARGET_GOAL", "CUSTOMER_TYPE", "EMPLOYMENT_TYPE", "CUSTOMER_SEGMENT",
    "TARGET_SEGMENT", "SEGMENT_TIER", "MIN_AGE", "MAX_AGE", "GENDER",
    "IS_BANK_OFFERED", "ACCOUNT_TYPE", "CARD_TYPE", "CHANNEL",
    "ELIGIBILITY_TYPE", "SERVICE_TYPE", "REWARD_TYPE", "CURRENCY",
    "CURRENCY_TYPE", "MIN_BALANCE", "AVG_BALANCE_REQUIREMENT", "MIN_INCOME",
    "MIN_INCOME_USD", "MIN_INVESTMENT", "MIN_CONTRIBUTION",
    "LOAN_AMOUNT_RANGE", "COVERAGE_AMOUNT", "FINANCING_TYPE", "DEPOSIT_PROFIT_TYPE",
    "DEPOSIT_PROFIT_FREQUENCY", "TENURE", "TENURE_OPTIONS", "MIN_TERM_YEARS",
    "MAX_TERM_YEARS", "BUSINESS_TENURE", "COLLATERAL_TYPE",
    "EQUITY_REQUIREMENT", "DBR_LIMIT", "TRANSACTION_LIMIT",
    "SPECIAL_CONDITIONS",
]


PRODUCTS_DIR = resolve_path("PRODUCTS_DIR", "products")
INDEX_PATH = resolve_path("INDEX_PATH", "products_index.json")
OUT_JSONL = resolve_path("OUT_JSONL", "extracted_data.jsonl")
OUT_XLSX = resolve_path("OUT_XLSX", "bank_products_extracted.xlsx")
PDF_PATH = resolve_path("PDF_PATH", "final_report_merged.pdf")
REGENERATE_STAGE1 = env_bool("REGENERATE_STAGE1", False)
