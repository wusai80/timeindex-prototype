"""Lightweight verification stage for LANL authentication judgments.

The verifier is intentionally narrower than a full classifier.
It focuses on whether a positive claim is structurally corroborated by the
retrieved evidence, rather than re-deciding the whole task from scratch.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from benchmarks.ibm_aml.deepseek_agent import DeepSeekAgentDecision
from timeindex.event import Event


_PATTERN_RE = re.compile(r"pattern=([^;]+)")
_FIELD_RE_TEMPLATE = r"{name}=([^;]+)"
_WEAK_EVENT_PATTERNS = {
    "lanl_query_host_touch",
    "lanl_user_continuity_detached",
}
_HIGH_BASELINE_EVENT_COUNT = 200
_HIGH_BASELINE_HOST_COUNT = 40


@dataclass(slots=True)
class VerificationResult:
    """Final prediction after an optional second-stage verifier."""

    predicted_positive: bool
    confidence: float
    rationale: str
    verifier_mode: str
    verifier_applied: bool
    verifier_overrode: bool
    verifier_reason: str
    verifier_features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_lanl_decision(
    query_event: Event,
    llm_decision: DeepSeekAgentDecision,
    retrieved_object_payloads: Sequence[Mapping[str, Any]],
    *,
    mode: str = "none",
) -> VerificationResult:
    """Optionally post-process an LLM decision with a deterministic verifier."""

    normalized_mode = str(mode or "none").strip().lower()
    if normalized_mode in {"", "none", "off", "disabled"}:
        return VerificationResult(
            predicted_positive=bool(llm_decision.predicted_positive),
            confidence=float(llm_decision.confidence),
            rationale=str(llm_decision.rationale),
            verifier_mode="none",
            verifier_applied=False,
            verifier_overrode=False,
            verifier_reason="verifier disabled",
            verifier_features={},
        )
    if normalized_mode != "lanl_rule":
        raise ValueError(f"Unsupported verifier mode: {mode}")
    return _verify_with_lanl_rule(query_event, llm_decision, retrieved_object_payloads)


def _verify_with_lanl_rule(
    query_event: Event,
    llm_decision: DeepSeekAgentDecision,
    retrieved_object_payloads: Sequence[Mapping[str, Any]],
) -> VerificationResult:
    features = _lanl_features(query_event, retrieved_object_payloads)
    predicted_positive = bool(llm_decision.predicted_positive)
    confidence = float(llm_decision.confidence)
    rationale = str(llm_decision.rationale)
    override = False
    reason = "llm verdict kept"

    if predicted_positive:
        strong_positive_bridge = (
            features["has_query_dst_precursor"]
            or features["chain_count"] > 0
            or (
                features["has_temporal_bridge"]
                and features["has_multi_step_context"]
                and features["event_detached_count"] >= 2
            )
            or (
                features["has_temporal_bridge"]
                and features["event_detached_count"] >= 2
                and features["event_query_host_touch_count"] >= 1
            )
        )
        weak_pattern_dominance = (
            features["event_count"] > 0
            and features["event_weak_pattern_count"] == features["event_count"]
        )
        weak_touch_only = (
            features["event_count"] >= 4
            and features["event_query_host_touch_count"] == features["event_count"]
            and not features["has_query_dst_precursor"]
            and features["chain_count"] == 0
        )
        skip_dominated_fanout = (
            features["skip_source_host_fanout_count"] >= 1
            and features["event_count"] >= 2
            and features["event_weak_pattern_count"] == features["event_count"]
            and features["event_query_host_touch_count"] >= 1
            and not features["has_query_dst_precursor"]
            and features["chain_count"] == 0
        )
        detached_skip_without_bridge = (
            features["skip_source_host_fanout_count"] >= 1
            and features["event_count"] >= 2
            and features["event_detached_count"] == features["event_count"]
            and features["event_query_host_touch_count"] == 0
            and not features["has_query_dst_precursor"]
            and not features["has_temporal_bridge"]
            and features["chain_count"] == 0
        )
        high_baseline_routine_corridor = (
            features["high_baseline_user_activity"]
            and features["skip_source_host_fanout_count"] == 0
            and features["chain_count"] == 0
            and not features["has_query_dst_precursor"]
            and weak_pattern_dominance
            and features["direct_query_touch_count"] >= 2
            and (
                not features["has_temporal_bridge"]
                or features["event_detached_count"] == 0
                or features["max_span_seconds"] <= 180.0
            )
        )
        if weak_touch_only and not strong_positive_bridge:
            predicted_positive = False
            confidence = min(confidence, 0.24)
            override = True
            reason = "weak host-touch corridor without precursor or chain support"
        elif skip_dominated_fanout and not strong_positive_bridge:
            predicted_positive = False
            confidence = min(confidence, 0.32)
            override = True
            reason = "skip-fanout story lacks precursor or chain corroboration"
        elif detached_skip_without_bridge and not strong_positive_bridge:
            predicted_positive = False
            confidence = min(confidence, 0.28)
            override = True
            reason = "detached skip story lacks a direct bridged path into the query"
        elif high_baseline_routine_corridor and not strong_positive_bridge:
            predicted_positive = False
            confidence = min(confidence, 0.26)
            override = True
            reason = "high-baseline touch corridor lacks structural evidence beyond routine reuse"

    final_rationale = rationale
    if override:
        final_rationale = (
            f"{rationale} Verifier override: historical context is dominated by {reason}."
        ).strip()

    return VerificationResult(
        predicted_positive=predicted_positive,
        confidence=confidence,
        rationale=final_rationale,
        verifier_mode="lanl_rule",
        verifier_applied=True,
        verifier_overrode=override,
        verifier_reason=reason,
        verifier_features=features,
    )


def _lanl_features(query_event: Event, retrieved_object_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    query_attrs = dict(query_event.attrs)
    object_count = 0
    skip_count = 0
    chain_count = 0
    event_count = 0
    has_query_dst_precursor = False
    has_temporal_bridge = False
    has_multi_step_context = False
    skip_source_host_fanout_count = 0
    event_query_host_touch_count = 0
    event_detached_count = 0
    event_weak_pattern_count = 0
    direct_query_touch_count = 0
    max_span_seconds = 0.0
    patterns: list[str] = []
    object_types: list[str] = []
    prior_user_event_count = _coerce_int(query_attrs.get("prior_user_event_count"))
    prior_user_host_count = _coerce_int(query_attrs.get("prior_user_host_count"))
    high_baseline_user_activity = (
        prior_user_event_count >= _HIGH_BASELINE_EVENT_COUNT
        or prior_user_host_count >= _HIGH_BASELINE_HOST_COUNT
    )

    for raw_object in retrieved_object_payloads:
        object_count += 1
        object_type = str(raw_object.get("type", "")).strip().lower() or "event"
        object_types.append(object_type)
        aspects = {str(aspect) for aspect in raw_object.get("aspects", []) if str(aspect)}
        summary = str(raw_object.get("summary", "")).strip()
        pattern = _extract_pattern(summary)
        query_host_touch = _extract_bool_field(summary, "query_host_touch")
        span_seconds = _extract_float_field(summary, "span_seconds")
        if pattern:
            patterns.append(pattern)
        max_span_seconds = max(max_span_seconds, span_seconds)

        if "lanl_query_dst_precursor" in aspects or "query_dst_precursor=true" in summary:
            has_query_dst_precursor = True
        if "lanl_temporal_bridge" in aspects:
            has_temporal_bridge = True
        if "lanl_multi_step_context" in aspects:
            has_multi_step_context = True
        if object_type == "skip":
            skip_count += 1
            if pattern == "lanl_source_host_fanout":
                skip_source_host_fanout_count += 1
        elif object_type == "chain":
            chain_count += 1
        else:
            event_count += 1
            if pattern == "lanl_query_host_touch":
                event_query_host_touch_count += 1
            if pattern == "lanl_user_continuity_detached":
                event_detached_count += 1
            if pattern in _WEAK_EVENT_PATTERNS:
                event_weak_pattern_count += 1
            if query_host_touch:
                direct_query_touch_count += 1

    return {
        "query_event_id": str(query_event.event_id),
        "object_count": object_count,
        "object_types": object_types,
        "patterns": patterns,
        "skip_count": skip_count,
        "chain_count": chain_count,
        "event_count": event_count,
        "has_query_dst_precursor": has_query_dst_precursor,
        "has_temporal_bridge": has_temporal_bridge,
        "has_multi_step_context": has_multi_step_context,
        "skip_source_host_fanout_count": skip_source_host_fanout_count,
        "event_query_host_touch_count": event_query_host_touch_count,
        "event_detached_count": event_detached_count,
        "event_weak_pattern_count": event_weak_pattern_count,
        "direct_query_touch_count": direct_query_touch_count,
        "max_span_seconds": max_span_seconds,
        "prior_user_event_count": prior_user_event_count,
        "prior_user_host_count": prior_user_host_count,
        "high_baseline_user_activity": high_baseline_user_activity,
    }


def _extract_pattern(summary: str) -> str:
    match = _PATTERN_RE.search(summary)
    if not match:
        return ""
    return match.group(1).strip().lower()


def _extract_bool_field(summary: str, name: str) -> bool:
    match = re.search(_FIELD_RE_TEMPLATE.format(name=re.escape(name)), summary)
    if not match:
        return False
    return match.group(1).strip().lower() == "true"


def _extract_float_field(summary: str, name: str) -> float:
    match = re.search(_FIELD_RE_TEMPLATE.format(name=re.escape(name)), summary)
    if not match:
        return 0.0
    try:
        return float(match.group(1).strip())
    except ValueError:
        return 0.0


def _coerce_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0
