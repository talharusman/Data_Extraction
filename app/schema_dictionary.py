from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldDefinition:
    description: str
    type_hint: str
    example: Any


FIELD_DEFINITIONS: dict[str, FieldDefinition] = {
    "PRODUCT_NAME": FieldDefinition("Normalized product or service name.", "Text / categorical", "Alfalah Car Ijarah"),
    "LEAD_CO_MNE": FieldDefinition("Lead marker from user (IBG or BNK).", "Text / categorical", "IBG"),
    "SOURCE_FILE_PRODUCT": FieldDefinition("Batch/source group name provided by user.", "Text / categorical", "Isl Consumer"),
    "PLAN_TYPE": FieldDefinition("High-level family such as Loan, Deposit, Savings,", "Text / categorical", "Loan"),
    "TARGET_GOAL": FieldDefinition("Primary customer need/use case.", "Text / categorical", "Housing"),
    "CUSTOMER_TYPE": FieldDefinition("Broad eligible customer class.", "Text / categorical", "SME / Corporate"),
    "EMPLOYMENT_TYPE": FieldDefinition("Employment eligibility where applicable.", "Text / categorical", "Salaried/SEP"),
    "CUSTOMER_SEGMENT": FieldDefinition("Behavioral/demographic segment when explicitly", "Text / categorical", "NRP"),
    "TARGET_SEGMENT": FieldDefinition("More specific positioning segment where stated.", "Text / categorical", "Financial Inclusion"),
    "SEGMENT": FieldDefinition("Priority/HNW/premium segment tag.", "Text / categorical", "Premium (HNW)"),
    "MIN_AGE": FieldDefinition("Minimum eligible age.", "Numeric", 18),
    "MAX_AGE": FieldDefinition("Maximum eligible age.", "Numeric", 65),
    "GENDER": FieldDefinition("Gender eligibility/focus.", "Text / categorical", "Female"),
    "BANK_CUSTOMER": FieldDefinition("1 for bank-offered/relationship products in this", "Numeric", 1),
    "ACCOUNT_TYPE": FieldDefinition("Current, Saving, Wallet, Digital etc", "Text / categorical", "Current"),
    "CARD_TYPE": FieldDefinition("Debit, Credit, Virtual Debit, Premium etc.", "Text / categorical", "Credit (Premium)"),
    "CHANNEL": FieldDefinition("Primary servicing/onboarding channel.", "Text / categorical", "Mobile App"),
    "ELIGIBILITY_TYPE": FieldDefinition("Short eligibility logic text.", "Text / categorical", "CNIC + Biometric"),
    "SERVICE_TYPE": FieldDefinition("Operational/service type for non-core products.", "Text / categorical", "Payments"),
    "REWARD_TYPE": FieldDefinition("Reward mechanism for loyalty rows.", "Text / categorical", "Points"),
    "CURRENCY": FieldDefinition("Currency or set of currencies.", "Text / categorical", "PKR + FCY"),
    "CURRENCY_TYPE": FieldDefinition("PKR/FCY style label if used.", "Text / categorical", "FCY"),
    "MIN_BALANCE": FieldDefinition("Minimum opening or operating balance.", "Numeric", 1000),
    "AVG_BALANCE_REQUIREMENT": FieldDefinition("Average balance requirement.", "Numeric", 50000),
    "MIN_INCOME": FieldDefinition("Minimum PKR income where stated.", "Numeric", 50000),
    "MIN_INCOME_USD": FieldDefinition("Minimum USD income where stated.", "Numeric", 3000),
    "MIN_INVESTMENT": FieldDefinition("Minimum investment/placement amount.", "Numeric", "100K"),
    "MIN_CONTRIBUTION": FieldDefinition("Minimum premium/contribution.", "Numeric", 250000),
    "LOAN_AMOUNT_RANGE": FieldDefinition("Loan size or facility range.", "Text / categorical", "200K-3M"),
    "COVERAGE_AMOUNT": FieldDefinition("Coverage amount or insured amount.", "Text / categorical", "50K-150K coverage"),
    "FINANCING_TYPE": FieldDefinition("Conventional/Islamic financing structure or", "Text / categorical", "Mudarabah"),
    "PROFIT_TYPE": FieldDefinition("Profit basis or mode.", "Text / categorical", "Tier-based"),
    "PROFIT_FREQUENCY": FieldDefinition("Monthly, semi-annual, maturity etc.", "Text / categorical", "Monthly"),
    "TENURE": FieldDefinition("Readable tenor text.", "Text / categorical", "1-5 years"),
    "TENURE_OPTIONS": FieldDefinition("Structured tenor menu text.", "Text / categorical", "1M -> 5Y"),
    "MIN_TERM_YEARS": FieldDefinition("Minimum term in years when directly available.", "Numeric", 10),
    "MAX_TERM_YEARS": FieldDefinition("Maximum term in years when directly available.", "Numeric", 25),
    "BUSINESS_TENURE": FieldDefinition("Required business age/operating history.", "Text / categorical", ">=3 years"),
    "COLLATERAL_TYPE": FieldDefinition("Security/collateral type.", "Text / categorical", "Property Mortgage"),
    "EQUITY_REQUIREMENT": FieldDefinition("Borrower equity or margin requirement.", "Text / categorical", "30%"),
    "DBR_LIMIT": FieldDefinition("Debt burden ratio limit.", "Text / categorical", "<=40%"),
    "TRANSACTION_LIMIT": FieldDefinition("Usage/balance/transaction cap.", "Text / categorical", "1M monthly"),
    "SPECIAL_CONDITIONS": FieldDefinition("Residual qualifiers or important caveats.", "Text / categorical", "RDA required"),
}


def get_field_definition(column: str) -> FieldDefinition:
    return FIELD_DEFINITIONS.get(
        column,
        FieldDefinition(column.replace("_", " ").title(), "Text / categorical", None),
    )


def column_meta(column: str) -> tuple[str, str, Any]:
    definition = get_field_definition(column)
    return definition.description, definition.type_hint, definition.example


def render_field_dictionary(fields: list[str]) -> str:
    lines = []
    for field in fields:
        definition = get_field_definition(field)
        lines.append(
            f"- {field}\n"
            f"  description: {definition.description}\n"
            f"  type_hint: {definition.type_hint}\n"
            f"  example: {definition.example}"
        )
    return "\n".join(lines)
