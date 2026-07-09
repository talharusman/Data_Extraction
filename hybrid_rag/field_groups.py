"""
field_groups.py
================

*** THIS FILE IS THE ONE-TIME, DEVELOPMENT-TIME SCHEMA ANALYSIS. ***

Grouping (9 groups, 55 LLM-extracted fields + SOURCE_FILE_PRODUCT filled
deterministically) is unchanged from the original analysis. What changed
in this revision: every field's description/type/example now comes
verbatim from the DATA_DICTIONARY you provided (COLUMN_NAME / DESCRIPTION
/ TYPE_HINT / EXAMPLE), instead of hand-written descriptions -- this is
the authoritative source for retrieval-query generation and for the
"expected type/example" hints shown in each group's prompt.

The runtime pipeline imports FIELD_GROUPS as a constant. It NEVER asks an
LLM to invent, reorder, or resize groups. If the 56-column schema OR the
data dictionary changes, a developer re-runs/edits this file -- grouping
is not recomputed at extraction time, per project requirements.

Each group carries:
    key         -- stable machine key (also the FAISS/BM25 "namespace" tag)
    name        -- human-readable label
    fields      -- {COLUMN_NAME: DATA_DICTIONARY description}, used for
                   retrieval-query generation
    field_order -- field names in original schema order (for prompt/output)
    query       -- auto-generated retrieval query (see
                   generate_retrieval_query) -- built once at import time
                   from field names + descriptions + examples, NOT
                   hand-written and NOT produced by an LLM.

SOURCE_FILE_PRODUCT is intentionally excluded from every group: it is
filled deterministically from the input filename (see
get_source_filename() in 02_extract_fields.py), never extracted by the LLM.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA DICTIONARY (authoritative, as provided) -- COLUMN_NAME -> {
#     description, type_hint, example
# }
# ---------------------------------------------------------------------------
DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "PRODUCT_NAME": {"description": "Normalized product or service name.", "type_hint": "Text / categorical", "example": "Alfalah Car Ijarah"},
    "LEAD_MARKER": {"description": "Lead marker from user (IBG or BNK).", "type_hint": "Text / categorical", "example": "IBG"},
    "SOURCE_FILE_PRODUCT": {"description": "Batch/source group name provided by user.", "type_hint": "Text / categorical", "example": "Isl Consumer"},
    "PLAN_TYPE": {"description": "High-level family such as Loan, Deposit, Savings, Card, Investment, Insurance, Service or Loyalty.", "type_hint": "Text / categorical", "example": "Loan"},
    "TARGET_GOAL": {"description": "Primary customer need/use case.", "type_hint": "Text / categorical", "example": "Housing"},
    "CUSTOMER_TYPE": {"description": "Broad eligible customer class.", "type_hint": "Text / categorical", "example": "SME / Corporate"},
    "EMPLOYMENT_TYPE": {"description": "Employment eligibility where applicable.", "type_hint": "Text / categorical", "example": "Salaried/SEP"},
    "CUSTOMER_SEGMENT": {"description": "Behavioral/demographic segment when explicitly identified.", "type_hint": "Text / categorical", "example": "NRP"},
    "TARGET_SEGMENT": {"description": "More specific positioning segment where stated.", "type_hint": "Text / categorical", "example": "Financial Inclusion"},
    "SEGMENT_TIER": {"description": "Priority/HNW/premium segment tag.", "type_hint": "Text / categorical", "example": "Premium (HNW)"},
    "MIN_AGE": {"description": "Minimum eligible age.", "type_hint": "Numeric", "example": "18"},
    "MAX_AGE": {"description": "Maximum eligible age.", "type_hint": "Numeric", "example": "65"},
    "GENDER": {"description": "Gender eligibility/focus.", "type_hint": "Text / categorical", "example": "Female"},
    "IS_BANK_OFFERED": {"description": "1 for bank-offered/relationship products in this normalized dataset.", "type_hint": "Numeric", "example": "1"},
    "ACCOUNT_TYPE": {"description": "Current, Savings, Wallet, Digital, etc.", "type_hint": "Text / categorical", "example": "Current"},
    "CARD_TYPE": {"description": "Debit, Credit, Virtual Debit, Premium etc.", "type_hint": "Text / categorical", "example": "Credit (Premium)"},
    "CHANNEL": {"description": "Primary servicing/onboarding channel.", "type_hint": "Text / categorical", "example": "Mobile App"},
    "ELIGIBILITY_TYPE": {"description": "Short eligibility logic text.", "type_hint": "Text / categorical", "example": "CNIC + Biometric"},
    "SERVICE_TYPE": {"description": "Operational/service type for non-core products.", "type_hint": "Text / categorical", "example": "Payments"},
    "REWARD_TYPE": {"description": "Reward mechanism for loyalty rows.", "type_hint": "Text / categorical", "example": "Points"},
    "CURRENCY": {"description": "Currency or set of currencies.", "type_hint": "Text / categorical", "example": "PKR + FCY"},
    "CURRENCY_TYPE": {"description": "PKR/FCY style label if used.", "type_hint": "Text / categorical", "example": "FCY"},
    "MIN_BALANCE": {"description": "Minimum opening or operating balance.", "type_hint": "Numeric", "example": "1000"},
    "AVG_BALANCE_REQUIREMENT": {"description": "Average balance requirement.", "type_hint": "Numeric", "example": "50000"},
    "MIN_INCOME": {"description": "Minimum PKR income where stated.", "type_hint": "Numeric", "example": "50000"},
    "MIN_INCOME_USD": {"description": "Minimum USD income where stated.", "type_hint": "Numeric", "example": "3000"},
    "MIN_INVESTMENT": {"description": "Minimum investment/placement amount.", "type_hint": "Numeric", "example": "100K"},
    "MIN_CONTRIBUTION": {"description": "Minimum premium/contribution.", "type_hint": "Numeric", "example": "250000"},
    "LOAN_AMOUNT_RANGE": {"description": "Loan size or facility range.", "type_hint": "Text / categorical", "example": "200K-3M"},
    "COVERAGE_AMOUNT": {"description": "Coverage amount or insured amount.", "type_hint": "Text / categorical", "example": "50K-150K coverage"},
    "FINANCING_TYPE": {"description": "Conventional/Islamic financing structure or instrument type.", "type_hint": "Text / categorical", "example": "Mudarabah"},
    "DEPOSIT_PROFIT_TYPE": {"description": "Profit basis or mode.", "type_hint": "Text / categorical", "example": "Tier-based"},
    "DEPOSIT_PROFIT_FREQUENCY": {"description": "Monthly, semi-annual, maturity etc.", "type_hint": "Text / categorical", "example": "Monthly"},
    "TENURE": {"description": "Readable tenor text.", "type_hint": "Text / categorical", "example": "1-5 years"},
    "TENURE_OPTIONS": {"description": "Structured tenor menu text.", "type_hint": "Text / categorical", "example": "1M -> 5Y"},
    "MIN_TERM_YEARS": {"description": "Minimum term in years when directly available.", "type_hint": "Numeric", "example": "10"},
    "MAX_TERM_YEARS": {"description": "Maximum term in years when directly available.", "type_hint": "Numeric", "example": "25"},
    "BUSINESS_TENURE": {"description": "Required business age/operating history.", "type_hint": "Text / categorical", "example": ">=3 years"},
    "COLLATERAL_TYPE": {"description": "Security/collateral type.", "type_hint": "Text / categorical", "example": "Property Mortgage"},
    "EQUITY_REQUIREMENT": {"description": "Borrower equity or margin requirement.", "type_hint": "Text / categorical", "example": "30%"},
    "DBR_LIMIT": {"description": "Debt burden ratio limit.", "type_hint": "Text / categorical", "example": "<=40%"},
    "TRANSACTION_LIMIT": {"description": "Usage/balance/transaction cap.", "type_hint": "Text / categorical", "example": "1M monthly"},
    "SPECIAL_CONDITIONS": {"description": "Residual qualifiers or important caveats.", "type_hint": "Text / categorical", "example": "RDA required"},
    "PRODUCT_DESCRIPTION": {"description": "Short summary of the product or plan.", "type_hint": "Text / categorical", "example": "Savings plan with flexible deposits"},
    "PROVIDER_NAME": {"description": "Bank, insurer, or product provider name.", "type_hint": "Text / categorical", "example": "Bank Alfalah"},
    "PRODUCT_VARIANT_TIER": {"description": "Tier, variant, or package label.", "type_hint": "Text / categorical", "example": "Premier"},
    "PRICING_RATE": {"description": "Rate, markup, margin, or pricing summary.", "type_hint": "Text / categorical", "example": "6.5% p.a."},
    "FEES_AND_CHARGES": {"description": "Concise fees and charges summary.", "type_hint": "Text / categorical", "example": "Issuance fee 500 PKR"},
    "KEY_BENEFITS": {"description": "Key benefits or value points.", "type_hint": "Text / categorical", "example": "Free withdrawals, digital access"},
    "OPTIONAL_RIDERS": {"description": "Optional riders or add-ons.", "type_hint": "Text / categorical", "example": "Accidental cover rider"},
    "FREE_LOOK_PERIOD_DAYS": {"description": "Free-look period in days.", "type_hint": "Numeric", "example": "14"},
    "REQUIRED_DOCUMENTS": {"description": "Required application documents.", "type_hint": "Text / categorical", "example": "CNIC, income proof"},
    "CLAIMS_SERVICE_CONTACT": {"description": "Claims or service contact details.", "type_hint": "Text / categorical", "example": "Call center 111-111-111"},
    "KEY_EXCLUSIONS": {"description": "Main exclusions or limitations.", "type_hint": "Text / categorical", "example": "Pre-existing conditions excluded"},
    "TAX_ZAKAT_TREATMENT": {"description": "Tax or zakat treatment note.", "type_hint": "Text / categorical", "example": "Zakat applicable"},
    "PREMIUM_PAYMENT_FREQUENCY": {"description": "Frequency at which premiums/contributions may be paid, when explicitly stated (e.g., Annual, Semi-Annual, Quarterly, Monthly).", "type_hint": "Text / categorical", "example": "Annual"},
}

# ---------------------------------------------------------------------------
# Group membership -- UNCHANGED from the original analysis. Only the field
# descriptions sourced above changed; which group each field belongs to
# did not.
# ---------------------------------------------------------------------------
_GROUP_FIELD_LISTS: list[tuple[str, str, list[str]]] = [
    ("identity", "Identity & Classification", [
        "PRODUCT_NAME", "LEAD_MARKER", "PLAN_TYPE", "TARGET_GOAL",
        "PROVIDER_NAME", "PRODUCT_DESCRIPTION", "PRODUCT_VARIANT_TIER", "FINANCING_TYPE",
    ]),
    ("customer_eligibility", "Customer & Eligibility", [
        "CUSTOMER_TYPE", "EMPLOYMENT_TYPE", "CUSTOMER_SEGMENT", "TARGET_SEGMENT",
        "SEGMENT_TIER", "MIN_AGE", "MAX_AGE", "GENDER", "ELIGIBILITY_TYPE", "IS_BANK_OFFERED",
    ]),
    ("mechanics_channel", "Product Mechanics & Channel", [
        "ACCOUNT_TYPE", "CARD_TYPE", "CHANNEL", "SERVICE_TYPE", "REWARD_TYPE",
    ]),
    ("currency_thresholds", "Currency & Financial Thresholds", [
        "CURRENCY", "CURRENCY_TYPE", "MIN_BALANCE", "AVG_BALANCE_REQUIREMENT",
        "MIN_INCOME", "MIN_INCOME_USD", "MIN_INVESTMENT", "MIN_CONTRIBUTION",
        "LOAN_AMOUNT_RANGE", "COVERAGE_AMOUNT",
    ]),
    ("profit_tenure_terms", "Profit, Tenure & Payment Terms", [
        "DEPOSIT_PROFIT_TYPE", "DEPOSIT_PROFIT_FREQUENCY", "TENURE", "TENURE_OPTIONS",
        "MIN_TERM_YEARS", "MAX_TERM_YEARS", "BUSINESS_TENURE", "PREMIUM_PAYMENT_FREQUENCY",
    ]),
    ("loan_risk_conditions", "Loan Risk & Conditions", [
        "COLLATERAL_TYPE", "EQUITY_REQUIREMENT", "DBR_LIMIT", "TRANSACTION_LIMIT", "SPECIAL_CONDITIONS",
    ]),
    ("pricing_fees", "Pricing & Fees", [
        "PRICING_RATE", "FEES_AND_CHARGES",
    ]),
    ("benefits_riders", "Benefits & Riders", [
        "KEY_BENEFITS", "OPTIONAL_RIDERS", "FREE_LOOK_PERIOD_DAYS",
    ]),
    ("docs_claims_compliance", "Documentation, Claims & Compliance", [
        "REQUIRED_DOCUMENTS", "CLAIMS_SERVICE_CONTACT", "KEY_EXCLUSIONS", "TAX_ZAKAT_TREATMENT",
    ]),
]


def generate_retrieval_query(fields: dict[str, str], field_names: list[str]) -> str:
    """
    Deterministically build a retrieval query from field names + their
    DATA_DICTIONARY descriptions + example values. Plain string
    composition, computed once at import time -- never by an LLM, never
    at runtime.
    """
    terms: list[str] = []
    for name in field_names:
        humanized = name.replace("_", " ").title()
        desc = fields[name]
        first_clause = desc.split(",")[0].split(";")[0]
        example = DATA_DICTIONARY.get(name, {}).get("example", "")
        terms.append(humanized)
        terms.append(first_clause)
        if example:
            terms.append(str(example))
    seen: set[str] = set()
    deduped = []
    for t in terms:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)
    return " ".join(deduped)


def _make_group(key: str, name: str, field_names: list[str]) -> dict:
    fields = {f: DATA_DICTIONARY[f]["description"] for f in field_names}
    return {
        "key": key,
        "name": name,
        "fields": fields,
        "field_order": field_names,
        "query": generate_retrieval_query(fields, field_names),
    }


FIELD_GROUPS: list[dict] = [_make_group(key, name, fields) for key, name, fields in _GROUP_FIELD_LISTS]

DETERMINISTIC_FIELDS = {"SOURCE_FILE_PRODUCT"}

ALL_GROUPED_FIELDS: set[str] = set()
for _g in FIELD_GROUPS:
    ALL_GROUPED_FIELDS |= set(_g["fields"].keys())


def field_to_group_key(field: str) -> str | None:
    for g in FIELD_GROUPS:
        if field in g["fields"]:
            return g["key"]
    return None


def validate_against_columns(columns: list[str]) -> None:
    """
    Sanity check to run once at startup: every schema column must be either
    grouped or explicitly marked deterministic, AND every grouped field
    must exist in DATA_DICTIONARY.
    """
    covered = ALL_GROUPED_FIELDS | DETERMINISTIC_FIELDS
    missing = [c for c in columns if c not in covered]
    extra = [f for f in ALL_GROUPED_FIELDS if f not in columns]
    no_dict_entry = [f for f in ALL_GROUPED_FIELDS if f not in DATA_DICTIONARY]
    if missing:
        raise ValueError(
            f"field_groups.py is out of date: these schema columns are not "
            f"assigned to any group and are not deterministic: {missing}. "
            f"Update field_groups.py (dev-time task, not runtime)."
        )
    if extra:
        raise ValueError(
            f"field_groups.py references columns that no longer exist in "
            f"the schema: {extra}. Update field_groups.py."
        )
    if no_dict_entry:
        raise ValueError(
            f"DATA_DICTIONARY is missing entries for: {no_dict_entry}. "
            f"Update DATA_DICTIONARY in field_groups.py."
        )
