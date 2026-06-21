"""Event representation extraction for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Iterable
import math
import re
from typing import Any

import numpy as np

from .config import ExtractorConfig
from .event import Event, EventRecord

_ENTITY_FIELD_TOKENS = ("id", "account", "user", "host", "service", "device")
_CONTEXT_FIELD_TOKENS = (
    "ctx",
    "context",
    "ip",
    "region",
    "country",
    "env",
    "environment",
    "browser",
    "session",
    "channel",
    "merchant",
    "currency",
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_SOURCE_ENDPOINT_FIELDS = (
    "account_id",
    "src_account",
    "source_account",
    "from_account",
    "origin_account",
    "sender_account",
    "src_user",
    "source_user",
)
_DESTINATION_ENDPOINT_FIELDS = (
    "dst_account",
    "destination_account",
    "to_account",
    "beneficiary_account",
    "beneficiary_id",
    "counterparty_account",
    "recipient_account",
    "receiver_account",
    "target_account",
    "dst_user",
    "target_user",
)


def extract_keys(event: Event, config: ExtractorConfig | None = None) -> set[str]:
    """Extract deterministic lookup keys from an event."""

    extractor_config = config or ExtractorConfig()
    keys: set[str] = set()
    keys.add(f"type:{event.event_type}")
    keys.add(f"time_block:{_time_block(event.time, extractor_config.time_bucket_width)}")
    source_endpoints: set[str] = set()
    destination_endpoints: set[str] = set()

    for field_name, value in event.attrs.items():
        if value is None:
            continue
        normalized_name = field_name.lower()
        normalized_value = _normalize_scalar(value)
        role = _flow_role(normalized_name)
        if role is not None and _is_endpoint_field(normalized_name):
            keys.add(f"entity:{normalized_name}={normalized_value}")
            if role == "source":
                source_endpoints.add(normalized_value)
                keys.add(f"participant:{normalized_value}")
                keys.add(f"flow_src:{normalized_value}")
            elif role == "destination":
                destination_endpoints.add(normalized_value)
                keys.add(f"participant:{normalized_value}")
                keys.add(f"flow_dst:{normalized_value}")
            continue
        if _is_entity_field(normalized_name):
            keys.add(f"entity:{normalized_name}={normalized_value}")
            continue
        if _is_context_field(normalized_name):
            keys.add(f"ctx:{normalized_name}={normalized_value}")
            continue
        if isinstance(value, bool):
            keys.add(f"attr:{normalized_name}={str(value).lower()}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            keys.add(f"attr_bin:{normalized_name}={_numeric_bin(value)}")
        else:
            keys.add(f"attr:{normalized_name}={normalized_value}")

    for field_name, value in event.ctx.items():
        if value is None:
            continue
        normalized_name = field_name.lower()
        keys.add(f"ctx:{normalized_name}={_normalize_scalar(value)}")

    for source_value in sorted(source_endpoints):
        for destination_value in sorted(destination_endpoints):
            if source_value == destination_value:
                continue
            keys.add(f"flow_pair:{source_value}->{destination_value}")

    return keys


def compute_vector(event: Event, keys: Iterable[str], dim: int = 128) -> np.ndarray:
    """Build a hashed bag-of-fields vector and L2 normalize it."""

    vector = np.zeros(dim, dtype=np.float64)
    for key in sorted(set(keys)):
        vector[_stable_index(key, dim)] += 1.0

    if event.text:
        for token in _tokenize(event.text):
            vector[_stable_index(f"text:{token}", dim)] += 0.2

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def extract_aspects(event: Event) -> set[str]:
    """Extract heuristic evidence aspects from transaction-like and log-like events."""

    aspects: set[str] = set()
    attrs = {key.lower(): value for key, value in event.attrs.items()}
    text = (event.text or "").lower()
    event_type = event.event_type.lower()

    amount = _get_numeric(attrs, "amount", "value", "transfer_amount")
    balance = _get_numeric(attrs, "balance", "balance_after", "remaining_balance")
    prior_balance = _get_numeric(attrs, "prior_balance", "balance_before", "available_balance")
    routine = _get_bool(attrs, "is_routine", "routine")
    new_beneficiary = _get_bool(attrs, "is_new_beneficiary", "new_beneficiary")
    device_changed = _get_bool(attrs, "device_changed", "is_device_shift")
    burst_count = _get_numeric(attrs, "burst_count", "recent_event_count")
    amount_ratio = _safe_ratio(amount, prior_balance)
    balance_ratio = _safe_ratio(amount, balance)

    if amount is not None and amount >= 1_000:
        aspects.add("large_transfer")
    if amount is not None and prior_balance is not None and amount_ratio >= 0.8:
        aspects.add("source_accumulation")
    if amount is not None and balance is not None and balance_ratio >= 0.95:
        aspects.add("full_balance_transfer")
    if new_beneficiary or "new beneficiary" in text:
        aspects.add("beneficiary_novelty")
    if device_changed or "new device" in text or "device shift" in text:
        aspects.add("device_shift")
    if burst_count is not None and burst_count >= 3:
        aspects.add("temporal_burst")

    if "deploy" in event_type or "deployment" in text or _get_bool(attrs, "deployment_changed"):
        aspects.add("deployment_change")
    if "upstream" in text or _matches_any(text, ("dependency error", "gateway error", "upstream error")):
        aspects.add("upstream_error")
    utilization = _get_numeric(attrs, "cpu_utilization", "memory_utilization", "utilization", "saturation")
    if (utilization is not None and utilization >= 0.9) or "saturation" in text:
        aspects.add("resource_saturation")
    timeout_count = _get_numeric(attrs, "timeout_count", "repeated_timeout_count")
    if (timeout_count is not None and timeout_count >= 2) or "repeated timeout" in text:
        aspects.add("repeated_timeout")
    metric_delta = _get_numeric(attrs, "metric_delta", "error_rate_delta", "latency_delta")
    if (metric_delta is not None and abs(metric_delta) >= 0.25) or "metric shift" in text:
        aspects.add("metric_shift")

    if routine:
        aspects.discard("beneficiary_novelty")

    if not aspects:
        aspects.add("generic_evidence")
    return aspects


def featurize_event(event: Event, config: ExtractorConfig) -> EventRecord:
    """Convert a raw event into an event record used by the prototype."""

    keys = extract_keys(event, config)
    sketch = compute_vector(event, keys, dim=config.sketch_dim)
    aspects = extract_aspects(event)
    return EventRecord(
        event=event,
        lookup_keys=keys,
        sketch=sketch,
        aspects=aspects,
    )


class EventRepresentationExtractor:
    """Extractor for lookup keys, sketches, and evidence aspects."""

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def extract(self, event: Event) -> EventRecord:
        return featurize_event(event, self.config)


def _is_entity_field(field_name: str) -> bool:
    return any(token in field_name for token in _ENTITY_FIELD_TOKENS)


def _is_context_field(field_name: str) -> bool:
    return any(token in field_name for token in _CONTEXT_FIELD_TOKENS)


def _is_endpoint_field(field_name: str) -> bool:
    return field_name in _SOURCE_ENDPOINT_FIELDS or field_name in _DESTINATION_ENDPOINT_FIELDS


def _flow_role(field_name: str) -> str | None:
    if field_name in _SOURCE_ENDPOINT_FIELDS or field_name.startswith(("src_", "source_", "from_", "origin_", "sender_")):
        return "source"
    if field_name in _DESTINATION_ENDPOINT_FIELDS or field_name.startswith(
        ("dst_", "destination_", "to_", "beneficiary_", "counterparty_", "receiver_", "recipient_", "target_")
    ):
        return "destination"
    return None


def _time_block(value: str | int | float, width: int) -> int:
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        numeric_value = int(digits) if digits else 0
    else:
        numeric_value = int(float(value))
    safe_width = max(width, 1)
    return numeric_value // safe_width


def _normalize_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip().lower()


def _numeric_bin(value: int | float) -> str:
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return "0"
    bucket = int(math.floor(math.log10(magnitude)))
    if value < 0:
        return f"neg10^{bucket}"
    return f"10^{bucket}"


def _stable_index(value: str, dim: int) -> int:
    total = 0
    for index, character in enumerate(value):
        total += (index + 1) * ord(character)
    return total % dim


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _get_numeric(attrs: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = attrs.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _get_bool(attrs: dict[str, Any], *names: str) -> bool:
    for name in names:
        value = attrs.get(name)
        if isinstance(value, bool):
            return value
    return False


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator == 0:
        return 0.0
    return numerator / denominator


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)
