"""Canonical schema hints for IBM AML transaction datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CanonicalField:
    """Maps a canonical semantic field to likely dataset column names."""

    name: str
    aliases: tuple[str, ...]
    required: bool = False


@dataclass(slots=True)
class IBMAmlSchema:
    """Resolved schema mapping for an IBM AML-style CSV."""

    dataset_name: str = "ibm_aml"
    source_file: str = ""
    transaction_id: str | None = None
    timestamp: str | None = None
    src_account: str | None = None
    dst_account: str | None = None
    amount: str | None = None
    currency: str | None = None
    payment_format: str | None = None
    type: str | None = None
    src_bank: str | None = None
    dst_bank: str | None = None
    label: str | None = None
    pattern_id: str | None = None
    columns: dict[str, str] = field(default_factory=dict)


TIMESTAMP_FIELD = CanonicalField(
    name="timestamp",
    aliases=(
        "timestamp",
        "time_step",
        "time step",
        "step",
        "datetime",
        "date",
        "transaction_date",
        "transaction_time",
    ),
    required=True,
)

SOURCE_ACCOUNT_FIELD = CanonicalField(
    name="source_account",
    aliases=(
        "source_account",
        "source account",
        "from_account",
        "from account",
        "sender_account",
        "sender account",
        "origin_account",
        "originator",
        "account",
        "account_id",
    ),
    required=True,
)

DESTINATION_ACCOUNT_FIELD = CanonicalField(
    name="destination_account",
    aliases=(
        "destination_account",
        "destination account",
        "to_account",
        "to account",
        "receiver_account",
        "receiver account",
        "beneficiary_account",
        "counterparty_account",
        "recipient_account",
    ),
    required=True,
)

AMOUNT_FIELD = CanonicalField(
    name="amount",
    aliases=(
        "amount",
        "amount_paid",
        "amount_received",
        "payment_amount",
        "transaction_amount",
        "value",
    ),
    required=True,
)

CURRENCY_FIELD = CanonicalField(
    name="currency_or_payment_format",
    aliases=(
        "currency",
        "payment_currency",
        "receiving_currency",
        "payment format",
        "payment_format",
        "format",
    ),
)

TRANSACTION_TYPE_FIELD = CanonicalField(
    name="transaction_type",
    aliases=(
        "transaction_type",
        "transaction type",
        "type",
        "payment_type",
        "transfer_type",
    ),
)

BANK_FIELD = CanonicalField(
    name="bank",
    aliases=(
        "bank",
        "source_bank",
        "source bank",
        "from_bank",
        "from bank",
        "destination_bank",
        "destination bank",
        "to_bank",
        "to bank",
    ),
)

LABEL_FIELD = CanonicalField(
    name="laundering_label",
    aliases=(
        "is_laundering",
        "is laundering",
        "label",
        "target",
        "suspicious",
        "sar_flag",
        "fraud_label",
    ),
)

PATTERN_FIELD = CanonicalField(
    name="laundering_pattern",
    aliases=(
        "laundering_pattern",
        "laundering pattern",
        "typology",
        "pattern",
        "aml_typology",
        "scheme",
    ),
)

CANONICAL_FIELDS: tuple[CanonicalField, ...] = (
    TIMESTAMP_FIELD,
    SOURCE_ACCOUNT_FIELD,
    DESTINATION_ACCOUNT_FIELD,
    AMOUNT_FIELD,
    CURRENCY_FIELD,
    TRANSACTION_TYPE_FIELD,
    BANK_FIELD,
    LABEL_FIELD,
    PATTERN_FIELD,
)


def normalize_column_name(value: str) -> str:
    """Normalize a column name for fuzzy alias matching."""

    cleaned = value.strip().lower()
    for char in ("-", "/", "(", ")", ".", ","):
        cleaned = cleaned.replace(char, " ")
    return "_".join(cleaned.split())


def iter_aliases(field: CanonicalField) -> Iterable[str]:
    """Yield normalized aliases for a canonical field."""

    for alias in field.aliases:
        yield normalize_column_name(alias)


def detect_schema(columns: Iterable[str], source_file: str = "", dataset_name: str = "ibm_aml") -> IBMAmlSchema:
    """Resolve likely IBM AML columns into a reusable schema object."""

    original_columns = list(columns)
    lookup = {normalize_column_name(column): column for column in original_columns}

    def resolve(field: CanonicalField) -> str | None:
        for alias in iter_aliases(field):
            if alias in lookup:
                return lookup[alias]
        return None

    schema = IBMAmlSchema(
        dataset_name=dataset_name,
        source_file=source_file,
        timestamp=resolve(TIMESTAMP_FIELD),
        src_account=resolve(SOURCE_ACCOUNT_FIELD),
        dst_account=resolve(DESTINATION_ACCOUNT_FIELD),
        amount=resolve(AMOUNT_FIELD),
        type=resolve(TRANSACTION_TYPE_FIELD),
        label=resolve(LABEL_FIELD),
    )
    schema.currency = _resolve_first(lookup, ("payment_currency", "receiving_currency", "currency"))
    schema.payment_format = _resolve_first(lookup, ("payment_format", "payment_system", "format"))
    schema.src_bank = _resolve_first(lookup, ("source_bank", "from_bank", "bankorig"))
    schema.dst_bank = _resolve_first(lookup, ("destination_bank", "to_bank", "bankdest"))
    schema.pattern_id = _resolve_first(lookup, ("pattern_id", "typology", "pattern", "alert_id", "group_id"))
    schema.transaction_id = _resolve_first(lookup, ("transaction_id", "tx_id", "event_id", "id"))
    schema.columns = {
        key: value
        for key, value in {
            "transaction_id": schema.transaction_id,
            "timestamp": schema.timestamp,
            "src_account": schema.src_account,
            "dst_account": schema.dst_account,
            "amount": schema.amount,
            "currency": schema.currency,
            "payment_format": schema.payment_format,
            "type": schema.type,
            "src_bank": schema.src_bank,
            "dst_bank": schema.dst_bank,
            "label": schema.label,
            "pattern_id": schema.pattern_id,
        }.items()
        if value
    }
    return schema


def _resolve_first(lookup: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        normalized = normalize_column_name(alias)
        if normalized in lookup:
            return lookup[normalized]
    return None
