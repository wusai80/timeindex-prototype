"""Two-stage LLM detector for IBM AML temporal evidence review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.ibm_aml.deepseek_agent import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_PATH,
    DEFAULT_MODEL,
    _build_evidence_overview,
    _build_interpretation_hints,
    _build_structure_summary,
    _clamp_zero_one,
    _event_amount,
    _extract_message_content,
    _format_event_summary,
    _format_timeline_line,
    _parse_json_response,
    _post_chat_completion,
    _time_value,
    load_api_key,
)
from timeindex.event import Event


ChatTransport = Callable[[str, str, list[dict[str, str]], str, float], dict[str, Any]]
_MECHANISM_VOCABULARY = (
    "accumulation_before_outflow",
    "bridge_relay",
    "repeated_pair_escalation",
    "circular_flow",
    "novelty_plus_escalation",
)
_MECHANISM_ALIASES = {
    "accumulation before outflow": "accumulation_before_outflow",
    "source accumulation": "accumulation_before_outflow",
    "origin_buildup_then_forwarding": "accumulation_before_outflow",
    "origin buildup then forwarding": "accumulation_before_outflow",
    "bridge relay": "bridge_relay",
    "destination bridge": "bridge_relay",
    "bridging/relay movement": "bridge_relay",
    "destination concentration": "bridge_relay",
    "destination buildup continuation": "bridge_relay",
    "repeated transfers forming a coherent path": "repeated_pair_escalation",
    "repeated pair escalation": "repeated_pair_escalation",
    "repeated flow pair": "repeated_pair_escalation",
    "same-flow-pair repeat": "repeated_pair_escalation",
    "circular flow": "circular_flow",
    "round tripping": "circular_flow",
    "round-tripping": "circular_flow",
    "round_tripping": "circular_flow",
    "novelty plus escalation": "novelty_plus_escalation",
    "beneficiary novelty": "novelty_plus_escalation",
    "source dispersion": "novelty_plus_escalation",
    "outward dispersion pattern": "novelty_plus_escalation",
}


@dataclass(slots=True)
class AnalystCaseReport:
    """Structured evidence summary produced by the analysis agent."""

    query_event_id: str
    query_label: str | None
    model: str
    timeline_summary: str
    suspicious_patterns: list[str]
    routine_explanations: list[str]
    key_entities: list[str]
    supporting_event_ids: list[str]
    confidence_signals: list[str]
    missing_evidence: list[str]
    analyst_confidence: float
    retrieved_event_ids: list[str]
    retrieved_aspects: list[str]
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TwoAgentDecision:
    """Final skeptical decision built on top of the analyst case file."""

    query_event_id: str
    query_label: str | None
    analyst_report: AnalystCaseReport
    judge_model: str
    predicted_positive: bool
    confidence: float
    rationale: str
    accepted_supporting_event_ids: list[str]
    accepted_mechanisms: list[str]
    rejected_claims: list[str]
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["analyst_report"] = self.analyst_report.to_dict()
        return payload


def build_analysis_messages(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
) -> list[dict[str, str]]:
    """Build the analysis-agent prompt."""

    aspects = sorted(str(aspect) for aspect in retrieved_aspects if str(aspect))
    sorted_events = sorted(retrieved_events, key=lambda event: (_time_value(event.time), event.event_id))
    allowed_ids = [event.event_id for event in sorted_events]
    event_lines = [
        _format_timeline_line(position + 1, event, query_event)
        for position, event in enumerate(sorted_events)
    ]
    user_sections = [
        "Task",
        "Build a structured case file for the query using only the prior retrieved evidence.",
        "Do not make the final suspicious/not-suspicious decision.",
        "",
        "Query Event",
        _format_event_summary(query_event, query_event, include_relations=False),
        "",
        "Evidence Overview",
        _build_evidence_overview(query_event, sorted_events),
        "",
        "Structural Story Summary",
        _build_structure_summary(query_event, sorted_events),
        "",
        "Evidence Interpretation Hints",
        _build_interpretation_hints(query_event, sorted_events),
        "",
        "Retrieved Aspect Hints",
        ", ".join(aspects) if aspects else "none",
        "",
        "Prior Evidence Timeline",
        *(event_lines if event_lines else ["No prior retrieved evidence."]),
        "",
        "Instructions",
        "- Extract concrete temporal patterns only when they are supported by the retrieved evidence.",
        "- Distinguish suspicious mechanisms from routine explanations.",
        "- If the evidence is weak, say so explicitly in missing_evidence and routine_explanations.",
        "- supporting_event_ids must include only event ids that directly support the strongest available story.",
        "- Treat all retrieved evidence as historical context only; do not infer or mention any event after the query.",
        "",
        "Output Requirements",
        "- Return valid JSON only.",
        "- Use keys: timeline_summary, suspicious_patterns, routine_explanations, key_entities, supporting_event_ids, confidence_signals, missing_evidence, analyst_confidence.",
        f"- supporting_event_ids must be a subset of: {allowed_ids}.",
        "- analyst_confidence must be between 0 and 1.",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are the Evidence Analyst in a two-agent financial event review system. "
                "Your role is to summarize temporal evidence carefully and conservatively. "
                "Do not decide whether the query is suspicious. "
                "Instead, produce a structured case file that separates suspicious mechanisms from routine explanations "
                "and identifies which retrieved events directly support each claim. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": "\n".join(user_sections)},
    ]


def analyze_case_with_llm(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
    *,
    provider: str = "deepseek",
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
    transport: ChatTransport | None = None,
) -> AnalystCaseReport:
    """Run the analysis agent and normalize its case file."""

    resolved_key = api_key or load_api_key(provider, key_path=key_path)
    resolved_transport = transport or _post_chat_completion
    messages = build_analysis_messages(query_event, retrieved_events, retrieved_aspects)
    response = resolved_transport(base_url, resolved_key, messages, model, timeout_s)
    parsed = _parse_json_response(_extract_message_content(response))
    allowed_ids = {event.event_id for event in retrieved_events}
    supporting_ids = _normalize_event_ids(parsed.get("supporting_event_ids", []), allowed_ids)
    retrieved_ids = [event.event_id for event in retrieved_events]
    return AnalystCaseReport(
        query_event_id=query_event.event_id,
        query_label=query_event.label,
        model=model,
        timeline_summary=_clean_text(parsed.get("timeline_summary")),
        suspicious_patterns=_clean_list(parsed.get("suspicious_patterns")),
        routine_explanations=_clean_list(parsed.get("routine_explanations")),
        key_entities=_clean_list(parsed.get("key_entities")),
        supporting_event_ids=supporting_ids,
        confidence_signals=_clean_list(parsed.get("confidence_signals")),
        missing_evidence=_clean_list(parsed.get("missing_evidence")),
        analyst_confidence=_clamp_zero_one(parsed.get("analyst_confidence", 0.0)),
        retrieved_event_ids=retrieved_ids,
        retrieved_aspects=sorted(str(aspect) for aspect in retrieved_aspects if str(aspect)),
        raw_response=response,
    )


def build_judge_messages(
    query_event: Event,
    analyst_report: AnalystCaseReport,
    retrieved_events: list[Event],
) -> list[dict[str, str]]:
    """Build the skeptical judge prompt."""

    sorted_events = sorted(retrieved_events, key=lambda event: (_time_value(event.time), event.event_id))
    allowed_ids = [event.event_id for event in sorted_events]
    event_lines = [
        _format_timeline_line(position + 1, event, query_event)
        for position, event in enumerate(sorted_events)
    ]
    analyst_sections = [
        f"timeline_summary={analyst_report.timeline_summary or 'none'}",
        f"suspicious_patterns={', '.join(analyst_report.suspicious_patterns) or 'none'}",
        f"routine_explanations={', '.join(analyst_report.routine_explanations) or 'none'}",
        f"key_entities={', '.join(analyst_report.key_entities) or 'none'}",
        f"supporting_event_ids={analyst_report.supporting_event_ids}",
        f"confidence_signals={', '.join(analyst_report.confidence_signals) or 'none'}",
        f"missing_evidence={', '.join(analyst_report.missing_evidence) or 'none'}",
        f"analyst_confidence={analyst_report.analyst_confidence:.2f}",
    ]
    user_sections = [
        "Task",
        "Review the analyst case file skeptically and decide whether the query event is suspicious.",
        "Return false unless the evidence supports at least one concrete temporal mechanism from the allowed list below.",
        "",
        "Query Event",
        _format_event_summary(query_event, query_event, include_relations=False),
        "",
        "Analyst Case File",
        *analyst_sections,
        "",
        "Prior Evidence Timeline",
        *(event_lines if event_lines else ["No prior retrieved evidence."]),
        "",
        "Review Rules",
        "- Accept a suspicious finding only if one or more concrete mechanisms are supported by the timeline.",
        "- Use only these mechanism labels: accumulation_before_outflow, bridge_relay, repeated_pair_escalation, circular_flow, novelty_plus_escalation.",
        "- Treat accumulation before outflow as valid when the source receives funds before sending outward and the retrieved timeline supports that ordering, even if the amounts are not perfectly matched.",
        "- Treat bridge_relay as valid for either explicit relay movement or concentrated many-to-one buildup into the destination-like entity when the query plausibly continues that bridge-arrival pattern.",
        "- Treat repeated pair escalation as valid when the same pair repeats and the query is materially larger or part of a short structured sequence.",
        "- Treat novelty_plus_escalation as valid when the query extends an unusual outward-dispersion or destination-concentration pattern, even if the visible history is only a partial slice of the full chain.",
        "- Treat circular_flow as valid when funds move back toward a prior counterparty or a bidirectional loop is visible in the retrieved timeline.",
        "- Never use or mention any event after the query. If a claim depends on future behavior, reject that claim.",
        "- Reject claims that are generic, routine, disconnected, or unsupported by the timeline, but do not require a perfect end-to-end laundering story when a coherent suspicious prefix is already visible.",
        "- If no concrete mechanism survives skepticism, predicted_positive must be false.",
        "",
        "Output Requirements",
        "- Return valid JSON only.",
        "- Use keys: predicted_positive, confidence, accepted_supporting_event_ids, accepted_mechanisms, rejected_claims, rationale.",
        f"- accepted_supporting_event_ids must be a subset of: {allowed_ids}.",
        "- confidence must be between 0 and 1.",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are the Cross-Examiner in a two-agent financial event review system. "
                "Your job is to question the analyst's claims, reject weak evidence, and make the final decision. "
                "Be skeptical of routine repeated behavior, disconnected history, and unsupported leaps. "
                "A positive judgment requires at least one concrete temporal mechanism supported by the retrieved timeline. "
                "Use only the allowed mechanism labels and never rely on events after the query. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": "\n".join(user_sections)},
    ]


def judge_case_with_llm(
    query_event: Event,
    analyst_report: AnalystCaseReport,
    retrieved_events: list[Event],
    *,
    provider: str = "deepseek",
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
    transport: ChatTransport | None = None,
) -> TwoAgentDecision:
    """Run the skeptical judge and enforce a concrete-mechanism requirement."""

    resolved_key = api_key or load_api_key(provider, key_path=key_path)
    resolved_transport = transport or _post_chat_completion
    messages = build_judge_messages(query_event, analyst_report, retrieved_events)
    response = resolved_transport(base_url, resolved_key, messages, model, timeout_s)
    parsed = _parse_json_response(_extract_message_content(response))
    allowed_ids = {event.event_id for event in retrieved_events}
    accepted_ids = _normalize_event_ids(parsed.get("accepted_supporting_event_ids", []), allowed_ids)
    accepted_mechanisms = _normalize_mechanisms(parsed.get("accepted_mechanisms"))
    rejected_claims = _clean_list(parsed.get("rejected_claims"))
    rejected_claims = _drop_forward_looking_claims(rejected_claims)
    validated_mechanisms, dropped_mechanisms = _validate_mechanisms(
        query_event,
        retrieved_events,
        accepted_mechanisms,
    )
    accepted_mechanisms = validated_mechanisms
    for dropped in dropped_mechanisms:
        claim = f"mechanism_not_validated:{dropped}"
        if claim not in rejected_claims:
            rejected_claims.append(claim)
    predicted_positive = bool(parsed.get("predicted_positive", False))
    confidence = _clamp_zero_one(parsed.get("confidence", 0.0))

    if not accepted_mechanisms:
        predicted_positive = False
        accepted_ids = []
        confidence = min(confidence, 0.49)
        if "no_concrete_temporal_mechanism" not in rejected_claims:
            rejected_claims.append("no_concrete_temporal_mechanism")

    return TwoAgentDecision(
        query_event_id=query_event.event_id,
        query_label=query_event.label,
        analyst_report=analyst_report,
        judge_model=model,
        predicted_positive=predicted_positive,
        confidence=confidence,
        rationale=_clean_text(parsed.get("rationale")),
        accepted_supporting_event_ids=accepted_ids,
        accepted_mechanisms=accepted_mechanisms,
        rejected_claims=rejected_claims,
        raw_response=response,
    )


def classify_query_with_two_agents(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
    *,
    provider: str = "deepseek",
    analyzer_model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_MODEL,
    analyzer_api_key: str | None = None,
    judge_api_key: str | None = None,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
    analyzer_transport: ChatTransport | None = None,
    judge_transport: ChatTransport | None = None,
) -> TwoAgentDecision:
    """Run the full two-agent pipeline."""

    analyst_report = analyze_case_with_llm(
        query_event,
        retrieved_events,
        retrieved_aspects,
        provider=provider,
        model=analyzer_model,
        api_key=analyzer_api_key,
        key_path=key_path,
        base_url=base_url,
        timeout_s=timeout_s,
        transport=analyzer_transport,
    )
    return judge_case_with_llm(
        query_event,
        analyst_report,
        retrieved_events,
        provider=provider,
        model=judge_model,
        api_key=judge_api_key,
        key_path=key_path,
        base_url=base_url,
        timeout_s=timeout_s,
        transport=judge_transport,
    )


def summarize_two_agent_decisions(decisions: list[TwoAgentDecision]) -> dict[str, float]:
    """Aggregate a two-agent run."""

    if not decisions:
        return {
            "queries": 0.0,
            "predicted_positive_rate": 0.0,
            "mean_confidence": 0.0,
            "mean_analyst_confidence": 0.0,
            "mean_accepted_mechanisms": 0.0,
        }
    return {
        "queries": float(len(decisions)),
        "predicted_positive_rate": sum(1 for item in decisions if item.predicted_positive) / len(decisions),
        "mean_confidence": sum(item.confidence for item in decisions) / len(decisions),
        "mean_analyst_confidence": sum(item.analyst_report.analyst_confidence for item in decisions) / len(decisions),
        "mean_accepted_mechanisms": sum(len(item.accepted_mechanisms) for item in decisions) / len(decisions),
    }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _normalize_event_ids(values: Any, allowed_ids: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for item in values:
        event_id = str(item)
        if event_id in allowed_ids and event_id not in normalized:
            normalized.append(event_id)
    return normalized


def _normalize_mechanisms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        lower_text = text.lower()
        mechanism = _MECHANISM_ALIASES.get(lower_text, lower_text)
        if mechanism in _MECHANISM_VOCABULARY and mechanism not in normalized:
            normalized.append(mechanism)
    return normalized


def _drop_forward_looking_claims(claims: list[str]) -> list[str]:
    blocked_phrases = ("after the query", "after query", "subsequent", "following the query", "afterward")
    filtered: list[str] = []
    for claim in claims:
        lower_claim = claim.lower()
        if any(phrase in lower_claim for phrase in blocked_phrases):
            continue
        filtered.append(claim)
    return filtered


def _validate_mechanisms(
    query_event: Event,
    retrieved_events: list[Event],
    mechanisms: list[str],
) -> tuple[list[str], list[str]]:
    if not mechanisms:
        return [], []

    prior_events = [
        event
        for event in retrieved_events
        if _time_value(event.time) <= _time_value(query_event.time) and event.event_id != query_event.event_id
    ]
    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    query_amount = _event_amount(query_event)

    inbound_to_source = [
        event for event in prior_events
        if query_src and str(event.attrs.get("dst_account") or "") == query_src
    ]
    inbound_sources = {
        str(event.attrs.get("src_account") or "")
        for event in inbound_to_source
        if str(event.attrs.get("src_account") or "")
    }
    inbound_total = sum(_event_amount(event) for event in inbound_to_source)

    same_pair_events = [
        event for event in prior_events
        if query_src
        and query_dst
        and str(event.attrs.get("src_account") or "") == query_src
        and str(event.attrs.get("dst_account") or "") == query_dst
    ]
    same_pair_amounts = [_event_amount(event) for event in same_pair_events if _event_amount(event) > 0.0]
    same_pair_count = len(same_pair_amounts)
    same_pair_ratio = query_amount / max(same_pair_amounts) if query_amount > 0.0 and same_pair_amounts else 0.0
    outbound_from_source = [
        event for event in prior_events
        if query_src and str(event.attrs.get("src_account") or "") == query_src
    ]
    outbound_destinations = {
        str(event.attrs.get("dst_account") or "")
        for event in outbound_from_source
        if str(event.attrs.get("dst_account") or "")
    }
    outbound_amounts = [_event_amount(event) for event in outbound_from_source if _event_amount(event) > 0.0]
    inbound_to_destination = [
        event for event in prior_events
        if query_dst and str(event.attrs.get("dst_account") or "") == query_dst
    ]
    destination_sources = {
        str(event.attrs.get("src_account") or "")
        for event in inbound_to_destination
        if str(event.attrs.get("src_account") or "")
    }
    destination_inbound_total = sum(_event_amount(event) for event in inbound_to_destination)

    circular_flow_valid = any(
        query_src
        and query_dst
        and str(event.attrs.get("src_account") or "") == query_dst
        and str(event.attrs.get("dst_account") or "") == query_src
        for event in prior_events
    )
    bridge_relay_valid = any(
        query_dst
        and str(event.attrs.get("dst_account") or "") == query_dst
        and any(
            str(other.attrs.get("src_account") or "") == query_dst
            and str(other.attrs.get("dst_account") or "") not in ("", query_src, query_dst)
            for other in prior_events
        )
        for event in prior_events
    )
    destination_concentration_valid = (
        len(inbound_to_destination) >= 2
        and len(destination_sources) >= 2
        and destination_inbound_total >= max(query_amount * 0.25, 1.0)
    )
    source_dispersion_valid = (
        len(outbound_from_source) >= 2
        and len(outbound_destinations) >= 2
        and query_amount > 0.0
        and (
            not outbound_amounts
            or query_amount >= max(max(outbound_amounts) * 0.75, 1.0)
        )
    )
    multi_source_accumulation_valid = (
        len(inbound_sources) >= 2
        and len(inbound_to_source) >= 2
        and inbound_total >= max(query_amount * 0.25, 1.0)
    )
    accumulation_valid = multi_source_accumulation_valid or (
        len(inbound_to_source) >= 2
        and inbound_total >= max(query_amount * 0.25, 1.0)
        and (bridge_relay_valid or circular_flow_valid)
    )
    bridge_relay_valid = bridge_relay_valid or destination_concentration_valid
    novelty_escalation_valid = (
        same_pair_count == 0
        and query_amount > 0.0
        and (
            len(inbound_sources) >= 2
            or source_dispersion_valid
            or destination_concentration_valid
        )
    )

    validated: list[str] = []
    dropped: list[str] = []
    for mechanism in mechanisms:
        is_valid = False
        if mechanism == "accumulation_before_outflow":
            is_valid = accumulation_valid
        elif mechanism == "bridge_relay":
            is_valid = bridge_relay_valid
        elif mechanism == "circular_flow":
            is_valid = circular_flow_valid
        elif mechanism == "repeated_pair_escalation":
            if same_pair_count >= 1 and same_pair_ratio >= 1.5:
                if same_pair_count <= 2:
                    is_valid = True
                else:
                    is_valid = bridge_relay_valid or circular_flow_valid
        elif mechanism == "novelty_plus_escalation":
            is_valid = novelty_escalation_valid

        if is_valid:
            validated.append(mechanism)
        else:
            dropped.append(mechanism)
    return validated, dropped
