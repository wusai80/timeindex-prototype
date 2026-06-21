"""DeepSeek-backed temporal evidence review agent for IBM AML experiments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from timeindex.event import Event


DEFAULT_KEY_PATH = Path("key(not for uploaded to github).md")
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
_KEY_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{24,})")


@dataclass(slots=True)
class DeepSeekAgentDecision:
    """One LLM-backed judgment for a suspicious query event."""

    query_event_id: str
    query_label: str | None
    model: str
    predicted_positive: bool
    confidence: float
    rationale: str
    risk_factors: list[str]
    supporting_event_ids: list[str]
    retrieved_event_ids: list[str]
    retrieved_aspects: list[str]
    prompt_event_count: int
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_api_key(
    provider: str = "deepseek",
    *,
    env_var: str | None = None,
    key_path: str | Path = DEFAULT_KEY_PATH,
) -> str:
    """Load an API key from the environment or the local key file."""

    resolved_env_var = env_var or f"{provider.upper()}_API_KEY"
    env_value = os.getenv(resolved_env_var, "").strip()
    if env_value:
        return env_value

    key_file = Path(key_path)
    if not key_file.exists():
        raise FileNotFoundError(
            f"No {provider} API key found in ${resolved_env_var} and key file is missing: {key_file}"
        )

    key_value = parse_key_file_for_provider(key_file.read_text(encoding="utf-8"), provider)
    if key_value:
        return key_value
    raise ValueError(f"Could not find a {provider} API key in {key_file}")


def parse_key_file_for_provider(text: str, provider: str) -> str | None:
    """Extract a provider key from a simple markdown or notes file."""

    aliases = {
        "deepseek": ("deepseek", "deep-seek"),
        "gpt": ("gpt", "openai"),
        "gemini": ("gemini", "google"),
    }.get(provider.lower(), (provider.lower(),))

    matching_lines: list[str] = []
    fallback_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(alias in line.lower() for alias in aliases):
            matching_lines.append(line)
        fallback_lines.append(line)

    for candidate in matching_lines + fallback_lines:
        match = _KEY_PATTERN.search(candidate)
        if match:
            return match.group(1)
    return None


def build_chat_messages(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
) -> list[dict[str, str]]:
    """Build a sanitized analyst-oriented prompt for the chat model."""

    aspects = sorted(str(aspect) for aspect in retrieved_aspects if str(aspect))
    sorted_events = sorted(retrieved_events, key=lambda event: (_time_value(event.time), event.event_id))
    allowed_ids = [event.event_id for event in sorted_events]
    query_summary = _format_event_summary(query_event, query_event, include_relations=False)
    evidence_overview = _build_evidence_overview(query_event, sorted_events)
    structure_summary = _build_structure_summary(query_event, sorted_events)
    interpretation_hints = _build_interpretation_hints(query_event, sorted_events)
    event_lines = [
        _format_timeline_line(position + 1, event, query_event)
        for position, event in enumerate(sorted_events)
    ]

    user_sections = [
        "Task",
        "Decide whether the query event appears suspicious based only on the prior evidence below.",
        "",
        "Query Event",
        query_summary,
        "",
        "Evidence Overview",
        evidence_overview,
        "",
        "Structural Story Summary",
        structure_summary,
        "",
        "Evidence Interpretation Hints",
        interpretation_hints,
        "",
        "Retrieved Aspect Hints",
        ", ".join(aspects) if aspects else "none",
        "",
        "Prior Evidence Timeline",
        *(event_lines if event_lines else ["No prior retrieved evidence."]),
        "",
        "Reasoning Guidance",
        "- Focus on concrete temporal-causal stories such as buildup into a source-like entity followed by outward movement, many-to-one concentration into a bridge-like entity, sudden novelty, state shift, or precursor patterns that plausibly explain the query.",
        "- Events that touch the same destination-like entity are weak by themselves if the query is simply another inbound event and there is no evidence of onward movement, escalation, or transformation.",
        "- Repeated same-pair or same-type activity is often routine unless it is combined with buildup, novelty, bridging, or an unusual transition in behavior.",
        "- Small or moderate magnitudes can still be suspicious if the temporal structure, repeated pattern, or buildup evidence is unusual.",
        "- If several retrieved events point to one coherent structural story, you may classify the query as suspicious even when no single event is decisive.",
        "- Generic similarity or long-range unrelated history is weak support unless it forms a direct bridge or precursor path to the query.",
        "- destination_only_buildup=true is evidence against suspicion for the current query unless the destination-like entity also shows onward movement or another concrete transition.",
        "- same_pair_repeat=true by itself is weak and often routine unless combined with accumulation, a bridge pattern, novelty, or a state shift.",
        "- If strong_non_ach_structure=true, do not dismiss the case just because the format or surface type looks routine.",
        "- Use false when the retrieved evidence is mostly routine, disconnected, inbound-only, or lacks a concrete structural explanation for the query.",
        "- Do not assume any labels or outcomes beyond what is explicitly shown here.",
        "",
        "Output Requirements",
        f"- Return valid JSON only.",
        f"- supporting_event_ids must be a subset of: {allowed_ids}.",
        "- confidence must be between 0 and 1.",
        "- predicted_positive should be false when the evidence is insufficient or routine.",
    ]
    return [
        {
            "role": "system",
        "content": (
                "You are a careful temporal evidence analyst reviewing one query event and a compact set of prior events. "
                "Ground-truth labels are intentionally unavailable. "
                "Rely only on temporal structure, repeated interaction patterns, buildup, novelty, bridge continuity, and other concrete precursor evidence that appears in the retrieved context. "
                "Do not overcall routine repeated behavior, but do recognize suspicious multi-event buildup even when individual events look ordinary in isolation. "
                "Surface format or event type can look routine while the surrounding structure is still suspicious, so prioritize causal evidence over labels or stereotypes. "
                "Use the retrieved evidence to judge whether a suspicious structural explanation is supported better than a routine explanation. "
                "Return valid JSON only with keys: predicted_positive, confidence, risk_factors, supporting_event_ids, rationale."
            ),
        },
        {"role": "user", "content": "\n".join(user_sections)},
    ]


def classify_query_with_deepseek(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
    transport: Callable[[str, str, list[dict[str, str]], str, float], dict[str, Any]] | None = None,
) -> DeepSeekAgentDecision:
    """Call the DeepSeek-compatible chat endpoint and normalize the result."""

    resolved_key = api_key or load_api_key("deepseek", key_path=key_path)
    resolved_transport = transport or _post_chat_completion
    messages = build_chat_messages(query_event, retrieved_events, retrieved_aspects)
    response = resolved_transport(base_url, resolved_key, messages, model, timeout_s)
    content = _extract_message_content(response)
    parsed = _parse_json_response(content)

    retrieved_ids = [event.event_id for event in retrieved_events]
    normalized_support_ids: list[str] = []
    for event_id in parsed.get("supporting_event_ids", []):
        event_id_text = str(event_id)
        if event_id_text in retrieved_ids and event_id_text not in normalized_support_ids:
            normalized_support_ids.append(event_id_text)

    return DeepSeekAgentDecision(
        query_event_id=query_event.event_id,
        query_label=query_event.label,
        model=model,
        predicted_positive=bool(parsed.get("predicted_positive", False)),
        confidence=_clamp_zero_one(parsed.get("confidence", 0.0)),
        rationale=str(parsed.get("rationale", "")).strip(),
        risk_factors=[str(item).strip() for item in parsed.get("risk_factors", []) if str(item).strip()],
        supporting_event_ids=normalized_support_ids,
        retrieved_event_ids=retrieved_ids,
        retrieved_aspects=sorted(str(aspect) for aspect in retrieved_aspects if str(aspect)),
        prompt_event_count=len(retrieved_events),
        raw_response=response,
    )


def summarize_decisions(decisions: list[DeepSeekAgentDecision]) -> dict[str, float]:
    """Aggregate a DeepSeek agent run."""

    if not decisions:
        return {
            "queries": 0.0,
            "predicted_positive_rate": 0.0,
            "mean_confidence": 0.0,
            "mean_supporting_events": 0.0,
        }
    return {
        "queries": float(len(decisions)),
        "predicted_positive_rate": sum(1 for item in decisions if item.predicted_positive) / len(decisions),
        "mean_confidence": sum(item.confidence for item in decisions) / len(decisions),
        "mean_supporting_events": sum(len(item.supporting_event_ids) for item in decisions) / len(decisions),
    }


def _post_chat_completion(
    base_url: str,
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = request.Request(
        base_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("DeepSeek response did not include choices")
    first = choices[0]
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek response did not include message content")
    return content


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response was not valid JSON") from None
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed


def _event_amount(event: Event) -> float:
    value = event.attrs.get("amount")
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _clamp_zero_one(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _format_event_summary(event: Event, query_event: Event, include_relations: bool = True) -> str:
    amount = _event_amount(event)
    currency = event.attrs.get("currency") or "unknown currency"
    src_account = event.attrs.get("src_account") or "unknown-src"
    dst_account = event.attrs.get("dst_account") or "unknown-dst"
    src_bank = event.attrs.get("src_bank") or "unknown-src-bank"
    dst_bank = event.attrs.get("dst_bank") or "unknown-dst-bank"
    payment_format = event.attrs.get("payment_format") or event.event_type or "transaction"
    relation_tags = _relation_tags(event, query_event) if include_relations else []
    relation_text = f" relation={','.join(relation_tags)}" if relation_tags else ""
    return (
        f"{event.event_id}: t={event.time}, type={payment_format}, "
        f"src={src_account}@{src_bank}, dst={dst_account}@{dst_bank}, "
        f"amount={amount:.2f} {currency}.{relation_text}"
    )


def _format_timeline_line(position: int, event: Event, query_event: Event) -> str:
    return f"{position}. {_format_event_summary(event, query_event)}"


def _build_evidence_overview(query_event: Event, retrieved_events: list[Event]) -> str:
    if not retrieved_events:
        return "No prior events retrieved."
    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    same_src = 0
    same_dst = 0
    inbound_to_query_src = 0
    outbound_from_query_src = 0
    inbound_to_query_dst = 0
    outbound_from_query_dst = 0
    same_flow_pair = 0
    total_amount = 0.0
    inbound_to_query_src_amount = 0.0
    outbound_from_query_src_amount = 0.0
    inbound_to_query_dst_amount = 0.0
    outbound_from_query_dst_amount = 0.0
    distinct_sources_to_query_src: set[str] = set()
    distinct_sources_to_query_dst: set[str] = set()
    distinct_destinations_from_query_src: set[str] = set()
    format_counts: dict[str, int] = {}
    for event in retrieved_events:
        amount = _event_amount(event)
        total_amount += amount
        src = str(event.attrs.get("src_account") or "")
        dst = str(event.attrs.get("dst_account") or "")
        payment_format = str(event.attrs.get("payment_format") or event.event_type or "transaction")
        format_counts[payment_format] = format_counts.get(payment_format, 0) + 1
        if src == query_src:
            same_src += 1
            outbound_from_query_src += 1
            outbound_from_query_src_amount += amount
            if dst:
                distinct_destinations_from_query_src.add(dst)
        if dst == query_dst:
            same_dst += 1
            inbound_to_query_dst += 1
            inbound_to_query_dst_amount += amount
            if src:
                distinct_sources_to_query_dst.add(src)
        if query_src and dst == query_src:
            inbound_to_query_src += 1
            inbound_to_query_src_amount += amount
            if src:
                distinct_sources_to_query_src.add(src)
        if query_dst and src == query_dst:
            outbound_from_query_dst += 1
            outbound_from_query_dst_amount += amount
        if query_src and query_dst and src == query_src and dst == query_dst:
            same_flow_pair += 1
    format_summary = ",".join(
        f"{name}:{count}"
        for name, count in sorted(format_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return (
        f"retrieved_events={len(retrieved_events)}, total_amount={total_amount:.2f}, "
        f"same_src_count={same_src}, same_dst_count={same_dst}, same_flow_pair_count={same_flow_pair}, "
        f"inbound_to_query_src={inbound_to_query_src} ({inbound_to_query_src_amount:.2f}), "
        f"outbound_from_query_src={outbound_from_query_src} ({outbound_from_query_src_amount:.2f}), "
        f"inbound_to_query_dst={inbound_to_query_dst} ({inbound_to_query_dst_amount:.2f}), "
        f"outbound_from_query_dst={outbound_from_query_dst} ({outbound_from_query_dst_amount:.2f}), "
        f"distinct_sources_to_query_src={len(distinct_sources_to_query_src)}, "
        f"distinct_sources_to_query_dst={len(distinct_sources_to_query_dst)}, "
        f"distinct_destinations_from_query_src={len(distinct_destinations_from_query_src)}, "
        f"prior_formats={format_summary or 'none'}"
    )


def _build_structure_summary(query_event: Event, retrieved_events: list[Event]) -> str:
    if not retrieved_events:
        return "No structural precursor evidence was retrieved."

    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    prior_to_origin = 0
    prior_from_origin = 0
    prior_to_target = 0
    prior_from_target = 0
    repeated_pair = 0
    origin_precursor_sources: set[str] = set()
    target_precursor_sources: set[str] = set()
    origin_follow_on_targets: set[str] = set()

    for event in retrieved_events:
        src = str(event.attrs.get("src_account") or "")
        dst = str(event.attrs.get("dst_account") or "")
        if query_src and dst == query_src:
            prior_to_origin += 1
            if src:
                origin_precursor_sources.add(src)
        if query_src and src == query_src:
            prior_from_origin += 1
            if dst:
                origin_follow_on_targets.add(dst)
        if query_dst and dst == query_dst:
            prior_to_target += 1
            if src:
                target_precursor_sources.add(src)
        if query_dst and src == query_dst:
            prior_from_target += 1
        if query_src and query_dst and src == query_src and dst == query_dst:
            repeated_pair += 1

    if prior_to_origin >= 2 and prior_from_origin >= 1:
        dominant_story = "origin_buildup_then_forwarding"
    elif prior_to_target >= 2 and prior_from_target >= 1:
        dominant_story = "target_bridge_or_relay"
    elif repeated_pair >= 2:
        dominant_story = "repeated_pair_interaction"
    elif prior_to_target >= 2:
        dominant_story = "target_only_buildup"
    elif prior_to_origin >= 1:
        dominant_story = "origin_has_precursors"
    else:
        dominant_story = "weak_or_fragmented_history"

    return (
        f"dominant_story={dominant_story}; "
        f"origin_precursor_count={prior_to_origin}; "
        f"origin_precursor_sources={len(origin_precursor_sources)}; "
        f"origin_follow_on_count={prior_from_origin}; "
        f"origin_follow_on_targets={len(origin_follow_on_targets)}; "
        f"target_precursor_count={prior_to_target}; "
        f"target_precursor_sources={len(target_precursor_sources)}; "
        f"target_follow_on_count={prior_from_target}; "
        f"repeated_pair_count={repeated_pair}"
    )


def _build_interpretation_hints(query_event: Event, retrieved_events: list[Event]) -> str:
    if not retrieved_events:
        return "weak_evidence=true; no prior history was retrieved."

    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    query_amount = _event_amount(query_event)
    query_format = str(query_event.attrs.get("payment_format") or query_event.event_type or "transaction")

    inbound_to_query_src = 0
    outbound_from_query_src = 0
    inbound_to_query_dst = 0
    outbound_from_query_dst = 0
    same_flow_pair = 0
    inbound_to_query_src_amount = 0.0
    distinct_sources_to_query_src: set[str] = set()
    distinct_sources_to_query_dst: set[str] = set()

    for event in retrieved_events:
        amount = _event_amount(event)
        src = str(event.attrs.get("src_account") or "")
        dst = str(event.attrs.get("dst_account") or "")
        if query_src and dst == query_src:
            inbound_to_query_src += 1
            inbound_to_query_src_amount += amount
            if src:
                distinct_sources_to_query_src.add(src)
        if src == query_src:
            outbound_from_query_src += 1
        if dst == query_dst:
            inbound_to_query_dst += 1
            if src:
                distinct_sources_to_query_dst.add(src)
        if query_dst and src == query_dst:
            outbound_from_query_dst += 1
        if query_src and query_dst and src == query_src and dst == query_dst:
            same_flow_pair += 1

    source_accumulation = (
        inbound_to_query_src >= 2
        and len(distinct_sources_to_query_src) >= 2
        and inbound_to_query_src_amount >= max(query_amount * 0.5, query_amount)
    )
    destination_bridge = inbound_to_query_dst >= 2 and outbound_from_query_dst >= 1
    destination_only_buildup = inbound_to_query_dst >= 2 and inbound_to_query_src == 0 and outbound_from_query_dst == 0
    same_pair_repeat = same_flow_pair >= 2 and inbound_to_query_src == 0
    likely_routine_card_payment = query_format in {"Credit Card", "Cheque"} and not source_accumulation and not destination_bridge
    strong_non_ach_structure = query_format in {"Credit Card", "Cheque"} and (
        source_accumulation
        or (destination_bridge and len(distinct_sources_to_query_dst) >= 2)
    )

    return (
        f"source_accumulation_candidate={str(source_accumulation).lower()}; "
        f"destination_bridge_candidate={str(destination_bridge).lower()}; "
        f"destination_only_buildup={str(destination_only_buildup).lower()}; "
        f"same_pair_repeat={str(same_pair_repeat).lower()}; "
        f"likely_routine_card_or_cheque={str(likely_routine_card_payment).lower()}; "
        f"strong_non_ach_structure={str(strong_non_ach_structure).lower()}; "
        f"query_payment_format={query_format}; "
        f"query_amount={query_amount:.2f}; "
        f"query_src_prior_inbound_total={inbound_to_query_src_amount:.2f}; "
        f"query_src_prior_inbound_sources={len(distinct_sources_to_query_src)}; "
        f"query_dst_prior_inbound_sources={len(distinct_sources_to_query_dst)}"
    )


def _relation_tags(event: Event, query_event: Event) -> list[str]:
    tags: list[str] = []
    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    src = str(event.attrs.get("src_account") or "")
    dst = str(event.attrs.get("dst_account") or "")
    if query_src and src == query_src:
        tags.append("same_src")
        tags.append("outbound_from_query_src")
    if query_dst and dst == query_dst:
        tags.append("same_dst")
        tags.append("inbound_to_query_dst")
    if query_src and dst == query_src:
        tags.append("inbound_to_query_src")
    if query_dst and src == query_dst:
        tags.append("outbound_from_query_dst")
    if query_src and query_dst and src == query_src and dst == query_dst:
        tags.append("same_flow_pair")
    return tags


def _time_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        digits = "".join(character for character in str(value) if character.isdigit())
        return float(digits) if digits else 0.0
