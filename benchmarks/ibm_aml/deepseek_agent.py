"""DeepSeek-backed temporal evidence review agent for IBM AML experiments."""

from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

import certifi

from timeindex.event import Event, EvidenceObject


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
    retrieved_objects: list[EvidenceObject] | None = None,
    *,
    domain: str | None = None,
) -> list[dict[str, str]]:
    """Build a sanitized analyst-oriented prompt for the chat model."""

    resolved_domain = (domain or _infer_prompt_domain(query_event)).strip().lower()
    if resolved_domain == "lanl":
        return _build_lanl_chat_messages(query_event, retrieved_events, retrieved_aspects, retrieved_objects)
    return _build_financial_chat_messages(query_event, retrieved_events, retrieved_aspects, retrieved_objects)


def _build_financial_chat_messages(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
    retrieved_objects: list[EvidenceObject] | None = None,
) -> list[dict[str, str]]:
    """Build the transaction-oriented prompt used for IBM AML style data."""

    aspects = sorted(str(aspect) for aspect in retrieved_aspects if str(aspect))
    sorted_events = sorted(retrieved_events, key=lambda event: (_time_value(event.time), event.event_id))
    allowed_ids = [event.event_id for event in sorted_events]
    allowed_id_set = set(allowed_ids)
    query_summary = _format_event_summary(query_event, query_event, include_relations=False)
    evidence_overview = _build_evidence_overview(query_event, sorted_events)
    structure_summary = _build_structure_summary(query_event, sorted_events)
    interpretation_hints = _build_interpretation_hints(query_event, sorted_events)
    query_card = _build_query_card(query_event)
    evidence_cards = _build_evidence_cards(query_event, sorted_events, retrieved_objects or [], allowed_id_set)
    event_lines = [
        _format_timeline_line(position + 1, event, query_event)
        for position, event in enumerate(sorted_events)
    ]

    user_sections: list[str] = [
        "Task",
        "Decide whether the query event appears suspicious based only on the prior evidence below.",
        "",
        "Query Card (JSON)",
        "```json",
        json.dumps(query_card, indent=2, sort_keys=True),
        "```",
        "",
        "Evidence Cards (JSON)",
        "```json",
        json.dumps(evidence_cards, indent=2, sort_keys=True),
        "```",
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
        "- Events that touch the same destination-like entity are weak by themselves only when they look isolated and routine; they become meaningful when they show many-to-one concentration, short-window escalation, unusually large continuation into the same destination-like entity, or another coherent bridge-arrival pattern.",
        "- Repeated same-pair or same-type activity is often routine unless it is combined with buildup, novelty, bridging, or an unusual transition in behavior.",
        "- Small or moderate magnitudes can still be suspicious if the temporal structure, repeated pattern, or buildup evidence is unusual.",
        "- A source-like entity that sends repeated or escalating outward transfers can be suspicious even when the visible history shows only partial inbound buildup, as long as the outward pattern itself is concentrated, unusual, or temporally structured.",
        "- If several retrieved events point to one coherent structural story, you may classify the query as suspicious even when no single event is decisive.",
        "- Generic similarity or long-range unrelated history is weak support unless it forms a direct bridge or precursor path to the query.",
        "- destination_only_buildup=true is not enough by itself, but it can still support suspicion when the destination-like entity shows concentrated inbound buildup, short-window escalation, bridge-like receiving behavior, or an unusually large continuation at the query.",
        "- same_pair_repeat=true by itself is weak and often routine unless combined with accumulation, a bridge pattern, novelty, or a state shift.",
        "- If strong_non_ach_structure=true, do not dismiss the case just because the format or surface type looks routine.",
        "- Use false when the retrieved evidence is mostly routine, disconnected, or lacks a concrete structural explanation for the query; do not require a perfect full laundering chain if the visible history already supports a coherent suspicious structure.",
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
                "You are a careful temporal evidence analyst reviewing one query event, a set of JSON evidence cards, and a compact set of prior events. "
                "Ground-truth labels are intentionally unavailable. "
                "Rely only on temporal structure, repeated interaction patterns, buildup, novelty, bridge continuity, and other concrete precursor evidence that appears in the retrieved context. "
                "Do not overcall routine repeated behavior, but do recognize suspicious multi-event buildup even when individual events look ordinary in isolation. "
                "Visible history may be incomplete, so a partial but coherent buildup, concentration, relay, or outward-dispersion pattern can still justify suspicion. "
                "Use the JSON cards as the primary semantic representation: each card separates positive_evidence from limitations, and continuity_to_query plus card_confidence indicate how strongly the card connects to the query. "
                "Treat limitations as warnings that qualify an evidence card, not as automatic vetoes. If several cards contribute aligned positive_evidence with moderate continuity, that can still support a suspicious explanation. "
                "Surface format or event type can look routine while the surrounding structure is still suspicious, so prioritize causal evidence over labels or stereotypes. "
                "Use the retrieved evidence to judge whether a suspicious structural explanation is supported better than a routine explanation. "
                "Return valid JSON only with keys: predicted_positive, confidence, risk_factors, supporting_event_ids, rationale."
            ),
        },
        {"role": "user", "content": "\n".join(user_sections)},
    ]


def _build_lanl_chat_messages(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
    retrieved_objects: list[EvidenceObject] | None = None,
) -> list[dict[str, str]]:
    """Build an authentication-oriented prompt for LANL style host activity."""

    aspects = sorted(str(aspect) for aspect in retrieved_aspects if str(aspect))
    sorted_events = sorted(retrieved_events, key=lambda event: (_time_value(event.time), event.event_id))
    allowed_ids = [event.event_id for event in sorted_events]
    allowed_id_set = set(allowed_ids)
    query_summary = _format_lanl_event_summary(query_event, query_event, include_relations=False)
    evidence_overview = _build_lanl_evidence_overview(query_event, sorted_events)
    structure_summary = _build_lanl_structure_summary(query_event, sorted_events)
    interpretation_hints = _build_lanl_interpretation_hints(query_event, sorted_events)
    query_card = _build_lanl_query_card(query_event)
    evidence_cards = _build_lanl_evidence_cards(query_event, sorted_events, retrieved_objects or [], allowed_id_set)
    event_lines = [
        _format_lanl_timeline_line(position + 1, event, query_event)
        for position, event in enumerate(sorted_events)
    ]

    user_sections: list[str] = [
        "Task",
        "Decide whether the query authentication event appears suspicious based only on the prior evidence below.",
        "",
        "Query Card (JSON)",
        "```json",
        json.dumps(query_card, indent=2, sort_keys=True),
        "```",
        "",
        "Evidence Cards (JSON)",
        "```json",
        json.dumps(evidence_cards, indent=2, sort_keys=True),
        "```",
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
        "- Focus on temporal host-and-credential stories with a concrete path into the query. Strong signals are: a prior local or network event on the query destination host, the same user moving from one host into several distinct real hosts in a short window, or a bridge host that clearly connects earlier activity to the query.",
        "- Default to false for routine authentication shapes unless there is a concrete bridge. In particular, novelty alone, same-user activity alone, same-source-host fanout alone, or a skip card with weak overlap are not enough.",
        "- Treat machine or service-account activity as routine by default. Treat `tgt` destinations, self-logons, and common Kerberos/NTLM maintenance patterns as weak unless paired with a clear host-to-host bridge.",
        "- A real-host fanout pattern can still be suspicious even when the user has substantial prior history, as long as the retrieved evidence shows the same user from the same source host reaching multiple distinct real hosts in a short window and then continuing into the query path.",
        "- Do not require a large fanout. Two distinct real hosts from the same user and same source-host corridor can already be enough when the window is tight and the query immediately continues that corridor.",
        "- A prior local or network event on the query destination host is a strong positive signal, especially when it occurs shortly before the query and shares the same user or source-host corridor.",
        "- A skip object is valuable only when it reaches materially earlier history or recovers missing context and still connects to the query user, source host, destination host, or an immediately adjacent bridge host. A skip object that only swaps in another nearby event is weak.",
        "- `detached_from_query=true` on one card is a warning, not an automatic veto. It becomes acceptable when another card supplies the actual bridge into the query corridor or when the detached card extends a multi-step same-user or same-source-host buildup that later reconnects to the query.",
        "- If the user already has many prior events or hosts, `is_new_dst_for_user=true` becomes weak evidence unless the retrieved history shows short-window escalation or a direct bridge into the query destination host.",
        "- Repeated access to the exact same path is weak by itself unless it is part of a larger host-spread or bridge pattern.",
        "- Host continuity matters more than generic similarity. Prefer evidence that shares the query user and at least one concrete host-level connection.",
        "- `query_host_touch=true` by itself is only a moderate cue. It should not drive a positive decision unless it is paired with a destination-host precursor, two-or-more distinct real targets from the same source corridor, or a recovered earlier bridge.",
        "- Be especially cautious when the strongest suspicious story comes from exactly one skip card with `pattern=lanl_source_host_fanout`, while the remaining cards are only single-hop `query_host_touch` or `user_continuity` steps and none show `query_dst_precursor=true`. That shape is often insufficient by itself.",
        "- Sparse or disconnected context should make you conservative. Use false when the retrieved evidence does not explain why this query stands out from routine LANL authentication traffic.",
        "- Do not assume any labels or outcomes beyond what is explicitly shown here.",
        "",
        "Decision Rule",
        "- Predict true only if there is either one strong host-level bridge signal or at least two aligned moderate signals. Strong bridge signals include a prior event on the query destination host, a recovered skip bridge with material temporal gain and participant overlap, or a clear short-window spread from the same user/source host into multiple real hosts including the query path.",
        "- A same-user + same-source-host + multiple-real-host fanout pattern followed by the query can count as a strong signal even if the user has a large prior baseline, provided the visible precursor window is tight and the query target is a real host rather than `tgt`.",
        "- A destination-host precursor or local event on the query destination host can outweigh detached upstream history. If one card bridges directly into the query destination and another card provides earlier buildup on the same corridor, treat them as complementary rather than contradictory.",
        "- Do not predict true from same-user continuity, repeated `query_host_touch`, or one-host repetition alone. Those require either a destination-host precursor, a recovered bridge, or spread into multiple distinct real hosts.",
        "- When a positive story depends on a skip fanout card, prefer false unless at least one non-skip card independently corroborates the spread or the query destination host itself has a precursor/resident event.",
        "- Otherwise predict false.",
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
                "You are a careful temporal security analyst reviewing one authentication query event, a set of JSON evidence cards, and a compact set of earlier host-access events. "
                "Ground-truth labels are intentionally unavailable. "
                "Judge whether the query is better explained by suspicious credential or host-movement structure than by routine authentication activity. "
                "Prioritize temporal continuity, bridge hosts, destination-host precursors, credential reuse across real hosts, new-host access after buildup, and multi-step lateral movement patterns. "
                "Do not overcall routine Kerberos or NTLM activity, `tgt` requests, repeated same-path access, or service-account behavior unless the retrieved evidence provides a concrete structural reason. "
                "Use the JSON cards as the primary semantic representation: each card separates positive_evidence from limitations, and continuity_to_query plus card_confidence indicate how strongly the card connects to the query. "
                "Treat limitations as warnings, not automatic vetoes, but remain conservative: novelty alone, generic fanout alone, repeated query_host_touch alone, or a weak skip card are not enough. Several moderate cards can justify suspicion only when they align into one coherent host-movement story with a concrete bridge into the query. "
                "Do allow a positive decision when the evidence shows a specific short-window fanout from the same user and source host into two or more real hosts, or a prior event on the query destination host immediately preceding the query, even if the user has a nontrivial baseline history. "
                "Do not let detached upstream history cancel a strong destination-host precursor; instead decide whether the detached card extends the same temporal corridor that later reconnects to the query. "
                "Be skeptical of cases where one skip card suggests source-host fanout but the rest of the evidence only shows repeated one-hop query_host_touch continuity without a destination-host precursor or another independent spread card. "
                "Return valid JSON only with keys: predicted_positive, confidence, risk_factors, supporting_event_ids, rationale."
            ),
        },
        {"role": "user", "content": "\n".join(user_sections)},
    ]


def classify_query_with_deepseek(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_aspects: list[str] | set[str],
    retrieved_objects: list[EvidenceObject] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
    transport: Callable[[str, str, list[dict[str, str]], str, float], dict[str, Any]] | None = None,
    domain: str | None = None,
) -> DeepSeekAgentDecision:
    """Call the DeepSeek-compatible chat endpoint and normalize the result."""

    resolved_key = api_key or load_api_key("deepseek", key_path=key_path)
    resolved_transport = transport or _post_chat_completion
    messages = build_chat_messages(
        query_event,
        retrieved_events,
        retrieved_aspects,
        retrieved_objects,
        domain=domain,
    )
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
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with request.urlopen(req, timeout=timeout_s, context=ssl_context) as response:
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


def _infer_prompt_domain(query_event: Event) -> str:
    attrs = query_event.attrs
    if "src_computer" in attrs or "dst_computer" in attrs:
        return "lanl"
    return "financial"


def _format_lanl_event_summary(event: Event, query_event: Event, include_relations: bool = True) -> str:
    src_user = event.attrs.get("src_user") or "unknown-user"
    dst_user = event.attrs.get("dst_user") or "unknown-dst-user"
    src_host = event.attrs.get("src_computer") or "unknown-src-host"
    dst_host = event.attrs.get("dst_computer") or "unknown-dst-host"
    auth_type = event.attrs.get("auth_type") or event.event_type or "authentication"
    logon_type = event.attrs.get("logon_type") or "unknown-logon"
    success = event.attrs.get("success")
    success_text = "success" if bool(success) else "failure"
    relation_tags = _lanl_relation_tags(event, query_event) if include_relations else []
    relation_text = f" relation={','.join(relation_tags)}" if relation_tags else ""
    return (
        f"{event.event_id}: t={event.time}, auth={auth_type}/{logon_type}, "
        f"user={src_user}, src_host={src_host}, dst_host={dst_host}, dst_user={dst_user}, "
        f"result={success_text}.{relation_text}"
    )


def _format_lanl_timeline_line(position: int, event: Event, query_event: Event) -> str:
    return f"{position}. {_format_lanl_event_summary(event, query_event)}"


def _build_lanl_query_card(query_event: Event) -> dict[str, Any]:
    src_user = str(query_event.attrs.get("src_user") or "")
    src_host = str(query_event.attrs.get("src_computer") or "")
    dst_host = str(query_event.attrs.get("dst_computer") or "")
    return {
        "event_id": query_event.event_id,
        "time": query_event.time,
        "event_type": str(query_event.event_type or "authentication"),
        "src_user": src_user,
        "dst_user": str(query_event.attrs.get("dst_user") or ""),
        "src_computer": src_host,
        "dst_computer": dst_host,
        "auth_type": str(query_event.attrs.get("auth_type") or ""),
        "logon_type": str(query_event.attrs.get("logon_type") or ""),
        "auth_orientation": str(query_event.attrs.get("auth_orientation") or ""),
        "success": bool(query_event.attrs.get("success", False)),
        "is_new_dst_for_user": bool(query_event.attrs.get("is_new_dst_for_user", False)),
        "prior_user_event_count": int(query_event.attrs.get("prior_user_event_count", 0) or 0),
        "prior_user_host_count": int(query_event.attrs.get("prior_user_host_count", 0) or 0),
        "prior_pair_seen": bool(query_event.attrs.get("prior_pair_seen", False)),
        "is_machine_account": bool(query_event.attrs.get("is_machine_account", False)),
        "is_anonymous_logon": bool(query_event.attrs.get("is_anonymous_logon", False)),
        "is_tgt_destination": _is_tgt_host(dst_host),
        "is_real_host_target": bool(dst_host) and not _is_tgt_host(dst_host) and dst_host != src_host,
        "is_self_logon": bool(src_host and dst_host and src_host == dst_host),
        "is_human_user": not _is_machine_like_user(src_user),
    }


def _build_lanl_evidence_cards(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_objects: list[EvidenceObject],
    allowed_ids: set[str],
) -> list[dict[str, Any]]:
    event_lookup = {event.event_id: event for event in retrieved_events}
    cards: list[dict[str, Any]] = []
    if retrieved_objects:
        for evidence in retrieved_objects:
            representative_ids = [
                event_id
                for event_id in (str(event_id) for event_id in evidence.event_ids)
                if event_id in allowed_ids
            ]
            representative_events = [event_lookup[event_id] for event_id in representative_ids if event_id in event_lookup]
            cards.append(_build_lanl_object_card(query_event, evidence, representative_events, representative_ids))
        return cards

    for event in retrieved_events:
        evidence = EvidenceObject(
            object_id=f"event:{event.event_id}",
            event_ids=[event.event_id],
            aspects=set(),
            summary="Single prior authentication event relevant to the query.",
            cost=1.0,
        )
        cards.append(_build_lanl_object_card(query_event, evidence, [event], [event.event_id]))
    return cards


def _build_lanl_object_card(
    query_event: Event,
    evidence: EvidenceObject,
    representative_events: list[Event],
    representative_ids: list[str],
) -> dict[str, Any]:
    object_type = _evidence_object_type(evidence)
    time_values = [_time_value(event.time) for event in representative_events]
    pattern = _infer_lanl_event_pattern(query_event, representative_events)
    continuity = _build_lanl_continuity_to_query(query_event, representative_events)
    limitations = _infer_lanl_limitations(query_event, representative_events, pattern, object_type)
    support_strength = {
        "aspect_matches": sorted(str(aspect) for aspect in evidence.aspects if str(aspect)),
        "cost": float(evidence.cost),
        "participant_overlap_with_query": _lanl_participant_overlap_with_query(query_event, representative_events),
        "representative_event_count": len(representative_ids),
    }
    bridge_score = _extract_bridge_score(evidence.summary)
    if bridge_score is not None:
        support_strength["bridge_score"] = bridge_score

    return {
        "object_id": str(evidence.object_id),
        "object_type": object_type,
        "claim": _build_lanl_claim(query_event, pattern, object_type),
        "participants": {
            "src_users": sorted({
                str(event.attrs.get("src_user") or "")
                for event in representative_events
                if str(event.attrs.get("src_user") or "")
            }),
            "src_hosts": sorted({
                str(event.attrs.get("src_computer") or "")
                for event in representative_events
                if str(event.attrs.get("src_computer") or "")
            }),
            "dst_hosts": sorted({
                str(event.attrs.get("dst_computer") or "")
                for event in representative_events
                if str(event.attrs.get("dst_computer") or "")
            }),
            "auth_types": sorted({
                str(event.attrs.get("auth_type") or "")
                for event in representative_events
                if str(event.attrs.get("auth_type") or "")
            }),
        },
        "time_span": {
            "start": min(time_values) if time_values else None,
            "end": max(time_values) if time_values else None,
            "all_before_query": True,
        },
        "support_strength": support_strength,
        "representative_events": representative_ids,
        "event_pattern": pattern,
        "continuity_to_query": continuity,
        "positive_evidence": _build_lanl_positive_evidence(
            query_event,
            representative_events,
            pattern,
            object_type,
            support_strength,
            continuity,
        ),
        "limitations": limitations,
        "card_confidence": _lanl_card_confidence(pattern, object_type, support_strength, continuity, limitations),
        "natural_language_summary": evidence.summary.strip() or _default_lanl_summary(pattern, representative_events),
    }


def _infer_lanl_event_pattern(query_event: Event, representative_events: list[Event]) -> str:
    if not representative_events:
        return "weak_or_fragmented_history"

    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    same_user = 0
    same_src_host = 0
    same_dst_host = 0
    same_path = 0
    prior_destinations_for_user: set[str] = set()
    prior_sources_for_query_dst: set[str] = set()
    prior_targets_from_query_src: set[str] = set()

    for event in representative_events:
        src_user = str(event.attrs.get("src_user") or "")
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        if query_user and src_user == query_user:
            same_user += 1
            if dst_host:
                prior_destinations_for_user.add(dst_host)
        if query_src_host and src_host == query_src_host:
            same_src_host += 1
            if dst_host:
                prior_targets_from_query_src.add(dst_host)
        if query_dst_host and dst_host == query_dst_host:
            same_dst_host += 1
            if src_host:
                prior_sources_for_query_dst.add(src_host)
        if query_user and query_src_host and query_dst_host and src_user == query_user and src_host == query_src_host and dst_host == query_dst_host:
            same_path += 1

    if same_user >= 2 and len(prior_destinations_for_user) >= 2:
        return "credential_reuse_across_hosts"
    if same_src_host >= 2 and len(prior_targets_from_query_src) >= 2:
        return "source_host_fanout"
    if same_dst_host >= 2 and len(prior_sources_for_query_dst) >= 2:
        return "destination_host_convergence"
    if same_path >= 2:
        return "repeated_same_path"
    if same_user >= 1 and same_dst_host >= 1:
        return "bridge_into_query_host"
    if same_user >= 1:
        return "user_continuity"
    return "weak_or_fragmented_history"


def _build_lanl_continuity_to_query(query_event: Event, representative_events: list[Event]) -> dict[str, Any]:
    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    query_auth_type = str(query_event.attrs.get("auth_type") or "")
    same_user = 0
    same_src_host = 0
    same_dst_host = 0
    same_path = 0
    same_auth_type = 0
    touches_query_host = 0

    for event in representative_events:
        src_user = str(event.attrs.get("src_user") or "")
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        auth_type = str(event.attrs.get("auth_type") or "")
        if query_user and src_user == query_user:
            same_user += 1
        if query_src_host and src_host == query_src_host:
            same_src_host += 1
        if query_dst_host and dst_host == query_dst_host:
            same_dst_host += 1
        if query_auth_type and auth_type == query_auth_type:
            same_auth_type += 1
        if query_user and query_src_host and query_dst_host and src_user == query_user and src_host == query_src_host and dst_host == query_dst_host:
            same_path += 1
        if (query_src_host and (src_host == query_src_host or dst_host == query_src_host)) or (query_dst_host and (src_host == query_dst_host or dst_host == query_dst_host)):
            touches_query_host += 1

    connection_kinds = [
        name
        for name, count in (
            ("same_user", same_user),
            ("same_src_host", same_src_host),
            ("same_dst_host", same_dst_host),
            ("same_path", same_path),
            ("same_auth_type", same_auth_type),
            ("touches_query_host", touches_query_host),
        )
        if count > 0
    ]
    strength = round(min(
        1.0,
        0.35 * (1 if same_user > 0 else 0)
        + 0.20 * (1 if same_src_host > 0 else 0)
        + 0.20 * (1 if same_dst_host > 0 else 0)
        + 0.40 * (1 if same_path > 0 else 0)
        + 0.10 * (1 if same_auth_type > 0 else 0)
        + 0.15 * (1 if touches_query_host > 0 else 0),
    ), 4)
    return {
        "connection_kinds": connection_kinds or ["none"],
        "strength": strength,
        "same_user_count": same_user,
        "same_src_host_count": same_src_host,
        "same_dst_host_count": same_dst_host,
        "same_path_count": same_path,
        "same_auth_type_count": same_auth_type,
        "touches_query_host_count": touches_query_host,
    }


def _infer_lanl_limitations(
    query_event: Event,
    representative_events: list[Event],
    pattern: str,
    object_type: str,
) -> list[str]:
    limitations: list[str] = []
    query_is_machine = bool(query_event.attrs.get("is_machine_account", False))
    query_is_anonymous = bool(query_event.attrs.get("is_anonymous_logon", False))
    query_auth_type = str(query_event.attrs.get("auth_type") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    same_path = 0
    diverse_hosts = set()
    touches_query_dst = 0
    real_targets_from_query_src: set[str] = set()
    for event in representative_events:
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        diverse_hosts.update(value for value in (src_host, dst_host) if value)
        if query_dst_host and (src_host == query_dst_host or dst_host == query_dst_host):
            touches_query_dst += 1
        if query_src_host and src_host == query_src_host and dst_host and not _is_tgt_host(dst_host) and dst_host != src_host:
            real_targets_from_query_src.add(dst_host)
        if (
            str(event.attrs.get("src_user") or "") == str(query_event.attrs.get("src_user") or "")
            and src_host == query_src_host
            and dst_host == query_dst_host
        ):
            same_path += 1
    if query_is_machine:
        limitations.append("The query uses a machine or service-style account, which can be routine.")
    if query_is_anonymous:
        limitations.append("Anonymous logon behavior needs stronger structural support than one isolated event.")
    if _is_tgt_host(query_dst_host):
        limitations.append("The query destination is a TGT-style service target, which is often routine without a host-level bridge.")
    if query_src_host and query_dst_host and query_src_host == query_dst_host:
        limitations.append("The query is a self-logon path, which is often routine unless embedded in a broader bridge pattern.")
    if pattern == "repeated_same_path":
        limitations.append("Repeated access on the same path can still be routine.")
    if query_auth_type in {"Kerberos", "NTLM"} and len(diverse_hosts) <= 2 and pattern in {"user_continuity", "weak_or_fragmented_history"}:
        limitations.append("Common authentication type with limited host diversity can be routine.")
    if object_type != "skip" and len(representative_events) <= 1:
        limitations.append("This card is supported by only one representative event.")
    if object_type == "skip":
        limitations.append("Skip evidence is compressed and should be checked against the representative events.")
    if same_path >= 1 and len(diverse_hosts) <= 2:
        limitations.append("The visible evidence may only show a local repeat rather than broader movement.")
    if touches_query_dst == 0 and pattern in {"user_continuity", "credential_reuse_across_hosts", "source_host_fanout"}:
        limitations.append("This card does not touch the query destination host directly, so it should usually be paired with another card or a tight continuation into the same query corridor.")
    if query_src_host and len(real_targets_from_query_src) <= 1 and touches_query_dst == 0 and pattern in {"source_host_fanout", "user_continuity", "bridge_into_query_host"}:
        limitations.append("This card mainly shows one corridor touching the query source host rather than an independently corroborated spread into multiple distinct real hosts.")
    return limitations


def _lanl_participant_overlap_with_query(query_event: Event, representative_events: list[Event]) -> float:
    query_entities = {
        str(query_event.attrs.get("src_user") or ""),
        str(query_event.attrs.get("src_computer") or ""),
        str(query_event.attrs.get("dst_computer") or ""),
    } - {""}
    if not query_entities or not representative_events:
        return 0.0
    candidate_entities = {
        str(value)
        for event in representative_events
        for value in (
            event.attrs.get("src_user") or "",
            event.attrs.get("src_computer") or "",
            event.attrs.get("dst_computer") or "",
        )
        if str(value)
    }
    if not candidate_entities:
        return 0.0
    return round(len(query_entities & candidate_entities) / len(query_entities), 4)


def _build_lanl_claim(query_event: Event, pattern: str, object_type: str) -> str:
    query_user = str(query_event.attrs.get("src_user") or "the user")
    query_src_host = str(query_event.attrs.get("src_computer") or "the source host")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "the destination host")
    if pattern == "credential_reuse_across_hosts":
        return f"{query_user} appears across multiple hosts before the query, suggesting credential reuse or lateral movement."
    if pattern == "source_host_fanout":
        return f"{query_src_host} reaches multiple destinations before the query, suggesting outward spread from the source host."
    if pattern == "destination_host_convergence":
        return f"{query_dst_host} receives prior activity from multiple sources before the query."
    if pattern == "repeated_same_path":
        return f"The exact path {query_user}@{query_src_host}->{query_dst_host} repeats before the query."
    if pattern == "bridge_into_query_host":
        return f"Earlier activity bridges into the query user or query host before the query fires."
    if pattern == "user_continuity":
        return f"The same user {query_user} appears in prior activity linked to the query."
    if object_type == "skip":
        return "This skip object compresses a longer precursor host-access path that may bridge distant context to the query."
    return "This object provides local precursor authentication evidence that may contribute one step of the query's temporal story."


def _build_lanl_positive_evidence(
    query_event: Event,
    representative_events: list[Event],
    pattern: str,
    object_type: str,
    support_strength: dict[str, Any],
    continuity: dict[str, Any],
) -> list[str]:
    evidence: list[str] = []
    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    query_time = _time_value(query_event.time)
    local_on_query_dst = 0
    real_targets_from_query_src: set[str] = set()
    same_user_count = 0
    min_gap_to_query = None
    for event in representative_events:
        event_time = _time_value(event.time)
        gap = max(0.0, query_time - event_time)
        min_gap_to_query = gap if min_gap_to_query is None else min(min_gap_to_query, gap)
        src_user = str(event.attrs.get("src_user") or "")
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        if src_user == query_user:
            same_user_count += 1
        if query_dst_host and src_host == query_dst_host and dst_host == query_dst_host:
            local_on_query_dst += 1
        if query_src_host and src_host == query_src_host and dst_host and not _is_tgt_host(dst_host) and dst_host != src_host:
            real_targets_from_query_src.add(dst_host)
    if continuity["same_user_count"] > 0:
        evidence.append("Contains prior activity by the same user as the query.")
    if continuity["same_src_host_count"] > 0:
        evidence.append("Touches the same source host as the query.")
    if continuity["same_dst_host_count"] > 0:
        evidence.append("Touches the same destination host as the query.")
    if continuity["same_path_count"] > 0:
        evidence.append("Contains the same user-to-host path as the query.")
    if pattern == "credential_reuse_across_hosts":
        evidence.append("Supports credential reuse across multiple hosts.")
    elif pattern == "source_host_fanout":
        evidence.append("Supports outward spread from one source host into multiple destinations.")
    elif pattern == "destination_host_convergence":
        evidence.append("Supports convergence into the query destination host.")
    elif pattern == "bridge_into_query_host":
        evidence.append("Supports a bridge into the query host or query user context.")
    if local_on_query_dst > 0:
        evidence.append("Includes a prior local or resident event on the query destination host.")
    if query_user and query_src_host and len(real_targets_from_query_src) >= 2 and same_user_count >= 1:
        evidence.append("Shows the same user from the same source host reaching multiple real hosts before the query.")
        evidence.append("Even two distinct real hosts from the same source corridor can be meaningful when the continuation into the query is immediate.")
    if min_gap_to_query is not None and min_gap_to_query <= 50:
        evidence.append("Includes very recent precursor activity immediately before the query.")
    if support_strength.get("bridge_score", 0.0) >= 0.3:
        evidence.append("Includes a non-trivial bridge score from a skip summary.")
    if support_strength.get("participant_overlap_with_query", 0.0) >= 0.34:
        evidence.append("Shares at least one key user or host participant with the query.")
    if bool(query_event.attrs.get("is_new_dst_for_user", False)) and continuity["same_user_count"] > 0:
        evidence.append("The query is a new destination for the user, and this card provides prior user context.")
    if object_type in {"chain", "skip"} and representative_events:
        evidence.append("Summarizes a multi-step path rather than a single isolated authentication.")
    if not evidence:
        evidence.append("Provides at least one temporally prior authentication event linked to the query context.")
    return evidence


def _lanl_card_confidence(
    pattern: str,
    object_type: str,
    support_strength: dict[str, Any],
    continuity: dict[str, Any],
    limitations: list[str],
) -> float:
    pattern_strength = {
        "credential_reuse_across_hosts": 0.90,
        "source_host_fanout": 0.80,
        "destination_host_convergence": 0.75,
        "bridge_into_query_host": 0.78,
        "repeated_same_path": 0.45,
        "user_continuity": 0.55,
        "weak_or_fragmented_history": 0.30,
    }.get(pattern, 0.30)
    bridge_score = float(support_strength.get("bridge_score", 0.0))
    overlap = float(support_strength.get("participant_overlap_with_query", 0.0))
    rep_count = int(support_strength.get("representative_event_count", 0))
    aspect_bonus = min(1.0, len(support_strength.get("aspect_matches", [])) / 2.0)
    continuity_strength = float(continuity.get("strength", 0.0))
    structure_bonus = 0.12 if object_type in {"chain", "skip"} else 0.0
    limitation_penalty = min(0.35, 0.05 * len(limitations))
    score = (
        0.30 * pattern_strength
        + 0.20 * overlap
        + 0.20 * continuity_strength
        + 0.10 * min(1.0, rep_count / 3.0)
        + 0.10 * bridge_score
        + 0.10 * aspect_bonus
        + structure_bonus
        - limitation_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)


def _default_lanl_summary(pattern: str, representative_events: list[Event]) -> str:
    if representative_events:
        return f"{pattern} based on {len(representative_events)} representative authentication events."
    return pattern


def _build_lanl_evidence_overview(query_event: Event, retrieved_events: list[Event]) -> str:
    if not retrieved_events:
        return "No prior authentication events retrieved."
    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    same_user = 0
    same_src_host = 0
    same_dst_host = 0
    same_path = 0
    unique_user_destinations: set[str] = set()
    unique_user_real_destinations: set[str] = set()
    unique_query_dst_sources: set[str] = set()
    unique_query_src_targets: set[str] = set()
    auth_type_counts: dict[str, int] = {}
    cross_host = 0
    query_dst_precursor_count = 0
    local_on_query_dst = 0
    time_values: list[float] = []
    for event in retrieved_events:
        src_user = str(event.attrs.get("src_user") or "")
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        auth_type = str(event.attrs.get("auth_type") or event.event_type or "authentication")
        time_values.append(_time_value(event.time))
        auth_type_counts[auth_type] = auth_type_counts.get(auth_type, 0) + 1
        if bool(event.attrs.get("is_cross_host_auth", False)):
            cross_host += 1
        if query_user and src_user == query_user:
            same_user += 1
            if dst_host:
                unique_user_destinations.add(dst_host)
                if not _is_tgt_host(dst_host) and dst_host != src_host:
                    unique_user_real_destinations.add(dst_host)
        if query_src_host and src_host == query_src_host:
            same_src_host += 1
            if dst_host:
                unique_query_src_targets.add(dst_host)
        if query_dst_host and dst_host == query_dst_host:
            same_dst_host += 1
            query_dst_precursor_count += 1
            if src_host:
                unique_query_dst_sources.add(src_host)
        if query_dst_host and src_host == query_dst_host and dst_host == query_dst_host:
            local_on_query_dst += 1
        if query_user and query_src_host and query_dst_host and src_user == query_user and src_host == query_src_host and dst_host == query_dst_host:
            same_path += 1
    auth_summary = ",".join(
        f"{name}:{count}"
        for name, count in sorted(auth_type_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return (
        f"retrieved_events={len(retrieved_events)}, "
        f"same_user_count={same_user}, same_src_host_count={same_src_host}, same_dst_host_count={same_dst_host}, "
        f"same_path_count={same_path}, cross_host_count={cross_host}, "
        f"user_distinct_destinations={len(unique_user_destinations)}, "
        f"user_distinct_real_destinations={len(unique_user_real_destinations)}, "
        f"query_src_distinct_targets={len(unique_query_src_targets)}, "
        f"query_dst_distinct_sources={len(unique_query_dst_sources)}, "
        f"query_dst_precursor_count={query_dst_precursor_count}, "
        f"local_on_query_dst_count={local_on_query_dst}, "
        f"retrieved_time_span={(max(time_values) - min(time_values)) if time_values else 0:.0f}, "
        f"prior_auth_types={auth_summary or 'none'}"
    )


def _build_lanl_structure_summary(query_event: Event, retrieved_events: list[Event]) -> str:
    if not retrieved_events:
        return "No structural precursor authentication evidence was retrieved."

    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    user_destinations: set[str] = set()
    src_host_targets: set[str] = set()
    src_host_real_targets: set[str] = set()
    dst_host_sources: set[str] = set()
    same_path = 0
    local_on_query_dst = 0

    for event in retrieved_events:
        src_user = str(event.attrs.get("src_user") or "")
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        if query_user and src_user == query_user and dst_host:
            user_destinations.add(dst_host)
        if query_src_host and src_host == query_src_host and dst_host:
            src_host_targets.add(dst_host)
            if not _is_tgt_host(dst_host) and dst_host != src_host:
                src_host_real_targets.add(dst_host)
        if query_dst_host and dst_host == query_dst_host and src_host:
            dst_host_sources.add(src_host)
        if query_dst_host and src_host == query_dst_host and dst_host == query_dst_host:
            local_on_query_dst += 1
        if query_user and query_src_host and query_dst_host and src_user == query_user and src_host == query_src_host and dst_host == query_dst_host:
            same_path += 1

    if local_on_query_dst >= 1:
        dominant_story = "target_host_precursor"
    elif len(src_host_real_targets) >= 2 and not _is_tgt_host(query_dst_host):
        dominant_story = "source_host_real_fanout"
    elif len(user_destinations) >= 2:
        dominant_story = "credential_reuse_or_user_spread"
    elif len(dst_host_sources) >= 2:
        dominant_story = "destination_host_convergence"
    elif same_path >= 2:
        dominant_story = "repeated_same_path"
    elif query_event.attrs.get("is_new_dst_for_user", False):
        dominant_story = "new_host_access_with_limited_context"
    else:
        dominant_story = "weak_or_fragmented_history"

    return (
        f"dominant_story={dominant_story}; "
        f"user_distinct_destinations={len(user_destinations)}; "
        f"source_host_distinct_targets={len(src_host_targets)}; "
        f"source_host_real_targets={len(src_host_real_targets)}; "
        f"destination_host_distinct_sources={len(dst_host_sources)}; "
        f"local_on_query_dst_count={local_on_query_dst}; "
        f"repeated_same_path_count={same_path}; "
        f"query_is_new_destination={str(bool(query_event.attrs.get('is_new_dst_for_user', False))).lower()}"
    )


def _build_lanl_interpretation_hints(query_event: Event, retrieved_events: list[Event]) -> str:
    if not retrieved_events:
        return "weak_evidence=true; no prior authentication history was retrieved."

    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    same_user = 0
    same_path = 0
    same_src_host = 0
    same_dst_host = 0
    user_destinations: set[str] = set()
    user_real_destinations: set[str] = set()
    src_host_real_destinations: set[str] = set()
    dst_host_sources: set[str] = set()
    local_on_query_dst = 0
    time_values: list[float] = []
    for event in retrieved_events:
        src_user = str(event.attrs.get("src_user") or "")
        src_host = str(event.attrs.get("src_computer") or "")
        dst_host = str(event.attrs.get("dst_computer") or "")
        time_values.append(_time_value(event.time))
        if query_user and src_user == query_user:
            same_user += 1
            if dst_host:
                user_destinations.add(dst_host)
                if not _is_tgt_host(dst_host) and dst_host != src_host:
                    user_real_destinations.add(dst_host)
        if query_src_host and src_host == query_src_host:
            same_src_host += 1
            if dst_host and not _is_tgt_host(dst_host) and dst_host != src_host:
                src_host_real_destinations.add(dst_host)
        if query_dst_host and dst_host == query_dst_host:
            same_dst_host += 1
            if src_host:
                dst_host_sources.add(src_host)
        if query_dst_host and src_host == query_dst_host and dst_host == query_dst_host:
            local_on_query_dst += 1
        if query_user and query_src_host and query_dst_host and src_user == query_user and src_host == query_src_host and dst_host == query_dst_host:
            same_path += 1

    credential_reuse = same_user >= 2 and len(user_real_destinations) >= 2
    destination_bridge = same_dst_host >= 2 and len(dst_host_sources) >= 2
    source_fanout = same_src_host >= 2 and len(user_real_destinations) >= 2
    repeated_same_path = same_path >= 2
    machine_account = bool(query_event.attrs.get("is_machine_account", False))
    anonymous_logon = bool(query_event.attrs.get("is_anonymous_logon", False))
    new_dst_for_user = bool(query_event.attrs.get("is_new_dst_for_user", False))
    tgt_destination = _is_tgt_host(query_dst_host)
    self_logon = bool(query_src_host and query_dst_host and query_src_host == query_dst_host)
    high_baseline_user_activity = int(query_event.attrs.get("prior_user_event_count", 0) or 0) >= 20 or int(query_event.attrs.get("prior_user_host_count", 0) or 0) >= 8
    strong_bridge_signal = local_on_query_dst >= 1 or (destination_bridge and not tgt_destination)
    retrieved_time_span = (max(time_values) - min(time_values)) if time_values else 0.0
    short_window_structured_spread = (
        same_user >= 2
        and same_src_host >= 2
        and len(user_real_destinations) >= 2
        and retrieved_time_span <= 400.0
        and not tgt_destination
    )
    strong_positive_candidate = strong_bridge_signal or short_window_structured_spread

    return (
        f"credential_reuse_candidate={str(credential_reuse).lower()}; "
        f"destination_bridge_candidate={str(destination_bridge).lower()}; "
        f"source_fanout_candidate={str(source_fanout).lower()}; "
        f"repeated_same_path={str(repeated_same_path).lower()}; "
        f"new_destination_for_user={str(new_dst_for_user).lower()}; "
        f"machine_account={str(machine_account).lower()}; "
        f"anonymous_logon={str(anonymous_logon).lower()}; "
        f"tgt_destination={str(tgt_destination).lower()}; "
        f"self_logon={str(self_logon).lower()}; "
        f"high_baseline_user_activity={str(high_baseline_user_activity).lower()}; "
        f"local_on_query_dst={str(local_on_query_dst > 0).lower()}; "
        f"strong_bridge_signal={str(strong_bridge_signal).lower()}; "
        f"short_window_structured_spread={str(short_window_structured_spread).lower()}; "
        f"same_source_real_host_count={len(src_host_real_destinations)}; "
        f"two_target_same_source_spread={str(len(src_host_real_destinations) >= 2).lower()}; "
        f"strong_positive_candidate={str(strong_positive_candidate).lower()}; "
        f"query_user_prior_event_count={int(query_event.attrs.get('prior_user_event_count', 0) or 0)}; "
        f"query_user_prior_host_count={int(query_event.attrs.get('prior_user_host_count', 0) or 0)}; "
        f"same_user_count={same_user}; "
        f"user_distinct_destinations={len(user_destinations)}; "
        f"user_distinct_real_destinations={len(user_real_destinations)}; "
        f"retrieved_time_span={retrieved_time_span:.0f}"
    )


def _lanl_relation_tags(event: Event, query_event: Event) -> list[str]:
    tags: list[str] = []
    query_user = str(query_event.attrs.get("src_user") or "")
    query_src_host = str(query_event.attrs.get("src_computer") or "")
    query_dst_host = str(query_event.attrs.get("dst_computer") or "")
    src_user = str(event.attrs.get("src_user") or "")
    src_host = str(event.attrs.get("src_computer") or "")
    dst_host = str(event.attrs.get("dst_computer") or "")
    if query_user and src_user == query_user:
        tags.append("same_user")
    if query_src_host and src_host == query_src_host:
        tags.append("same_src_host")
    if query_dst_host and dst_host == query_dst_host:
        tags.append("same_dst_host")
    if query_user and query_src_host and query_dst_host and src_user == query_user and src_host == query_src_host and dst_host == query_dst_host:
        tags.append("same_path")
    if (query_src_host and dst_host == query_src_host) or (query_dst_host and src_host == query_dst_host):
        tags.append("bridge_host_touch")
    return tags


def _is_tgt_host(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"tgt", "krbtgt"}


def _is_machine_like_user(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized.endswith("$")


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


def _build_query_card(query_event: Event) -> dict[str, Any]:
    return {
        "event_id": query_event.event_id,
        "time": query_event.time,
        "event_type": str(query_event.attrs.get("payment_format") or query_event.event_type or "transaction"),
        "src_account": str(query_event.attrs.get("src_account") or ""),
        "dst_account": str(query_event.attrs.get("dst_account") or ""),
        "src_bank": str(query_event.attrs.get("src_bank") or ""),
        "dst_bank": str(query_event.attrs.get("dst_bank") or ""),
        "amount": _event_amount(query_event),
        "currency": str(query_event.attrs.get("currency") or ""),
    }


def _build_evidence_cards(
    query_event: Event,
    retrieved_events: list[Event],
    retrieved_objects: list[EvidenceObject],
    allowed_ids: set[str],
) -> list[dict[str, Any]]:
    event_lookup = {event.event_id: event for event in retrieved_events}
    cards: list[dict[str, Any]] = []
    if retrieved_objects:
        for evidence in retrieved_objects:
            representative_ids = [
                event_id
                for event_id in (str(event_id) for event_id in evidence.event_ids)
                if event_id in allowed_ids
            ]
            representative_events = [event_lookup[event_id] for event_id in representative_ids if event_id in event_lookup]
            cards.append(_build_object_card(query_event, evidence, representative_events, representative_ids))
        return cards

    for event in retrieved_events:
        evidence = EvidenceObject(
            object_id=f"event:{event.event_id}",
            event_ids=[event.event_id],
            aspects=set(),
            summary="Single prior event relevant to the query.",
            cost=1.0,
        )
        cards.append(_build_object_card(query_event, evidence, [event], [event.event_id]))
    return cards


def _build_object_card(
    query_event: Event,
    evidence: EvidenceObject,
    representative_events: list[Event],
    representative_ids: list[str],
) -> dict[str, Any]:
    object_type = _evidence_object_type(evidence)
    source_entities = sorted({
        str(event.attrs.get("src_account") or "")
        for event in representative_events
        if str(event.attrs.get("src_account") or "")
    })
    destination_entities = sorted({
        str(event.attrs.get("dst_account") or "")
        for event in representative_events
        if str(event.attrs.get("dst_account") or "")
    })
    time_values = [_time_value(event.time) for event in representative_events]
    pattern = _infer_event_pattern(query_event, representative_events)
    continuity = _build_continuity_to_query(query_event, representative_events)
    limitations = _infer_limitations(query_event, representative_events, pattern, object_type)
    participant_overlap = _participant_overlap_with_query(query_event, representative_events)
    support_strength = {
        "aspect_matches": sorted(str(aspect) for aspect in evidence.aspects if str(aspect)),
        "cost": float(evidence.cost),
        "participant_overlap_with_query": participant_overlap,
        "representative_event_count": len(representative_ids),
    }
    bridge_score = _extract_bridge_score(evidence.summary)
    if bridge_score is not None:
        support_strength["bridge_score"] = bridge_score
    positive_evidence = _build_positive_evidence(
        query_event,
        representative_events,
        pattern,
        object_type,
        support_strength,
        continuity,
    )
    card_confidence = _card_confidence(
        pattern,
        object_type,
        support_strength,
        continuity,
        limitations,
    )

    return {
        "object_id": str(evidence.object_id),
        "object_type": object_type,
        "claim": _build_claim(query_event, pattern, object_type, representative_events),
        "participants": {
            "source_entities": source_entities,
            "destination_entities": destination_entities,
        },
        "time_span": {
            "start": min(time_values) if time_values else None,
            "end": max(time_values) if time_values else None,
            "all_before_query": True,
        },
        "support_strength": support_strength,
        "representative_events": representative_ids,
        "event_pattern": pattern,
        "continuity_to_query": continuity,
        "positive_evidence": positive_evidence,
        "limitations": limitations,
        "card_confidence": card_confidence,
        "natural_language_summary": evidence.summary.strip() or _default_summary(pattern, representative_events),
    }


def _evidence_object_type(evidence: EvidenceObject) -> str:
    object_id = str(evidence.object_id)
    summary = evidence.summary.lower()
    if object_id.startswith("skip:"):
        return "skip"
    if object_id.startswith("ordinary:"):
        return "ordinary"
    if object_id.startswith("chain:") or " chain " in f" {summary} ":
        return "chain"
    return "evidence"


def _infer_event_pattern(query_event: Event, representative_events: list[Event]) -> str:
    if not representative_events:
        return "weak_or_fragmented_history"

    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    inbound_to_src = 0
    inbound_to_dst = 0
    outbound_from_src = 0
    repeated_pair = 0
    for event in representative_events:
        src = str(event.attrs.get("src_account") or "")
        dst = str(event.attrs.get("dst_account") or "")
        if query_src and dst == query_src:
            inbound_to_src += 1
        if query_dst and dst == query_dst:
            inbound_to_dst += 1
        if query_src and src == query_src:
            outbound_from_src += 1
        if query_src and query_dst and src == query_src and dst == query_dst:
            repeated_pair += 1

    if inbound_to_src >= 2 and outbound_from_src >= 1:
        return "source_accumulation_then_outflow"
    if inbound_to_dst >= 2 and repeated_pair >= 1:
        return "destination_concentration_with_pair_continuation"
    if inbound_to_dst >= 2:
        return "many_to_one_concentration"
    if repeated_pair >= 1:
        return "same_pair_repeat"
    if outbound_from_src >= 2:
        return "source_dispersion"
    return "weak_or_fragmented_history"


def _infer_limitations(
    query_event: Event,
    representative_events: list[Event],
    pattern: str,
    object_type: str,
) -> list[str]:
    limitations: list[str] = []
    query_format = str(query_event.attrs.get("payment_format") or query_event.event_type or "transaction")
    query_amount = _event_amount(query_event)
    same_pair_count = 0
    inbound_to_dst_amount = 0.0
    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    outbound_from_dst = 0
    for event in representative_events:
        src = str(event.attrs.get("src_account") or "")
        dst = str(event.attrs.get("dst_account") or "")
        amount = _event_amount(event)
        if query_dst and dst == query_dst:
            inbound_to_dst_amount += amount
        if query_dst and src == query_dst:
            outbound_from_dst += 1
        if query_src and query_dst and src == query_src and dst == query_dst:
            same_pair_count += 1

    if pattern == "many_to_one_concentration":
        limitations.append("Destination buildup appears without visible onward movement inside this card.")
    if same_pair_count >= 1:
        limitations.append("Same-pair repetition alone can still be routine.")
    if query_format in {"Credit Card", "Cheque"} and pattern in {"many_to_one_concentration", "same_pair_repeat", "weak_or_fragmented_history"}:
        limitations.append("Surface format looks routine (card/cheque), so structural evidence must carry the case.")
    if inbound_to_dst_amount > 0.0 and query_amount <= inbound_to_dst_amount * 0.10:
        limitations.append("Query amount is small relative to prior destination buildup in this card.")
    if outbound_from_dst == 0 and pattern in {"many_to_one_concentration", "destination_concentration_with_pair_continuation"}:
        limitations.append("No onward movement from the destination is visible within this card.")
    if object_type != "skip" and len(representative_events) <= 1:
        limitations.append("This card is supported by only one representative event.")
    if object_type == "skip":
        limitations.append("Skip evidence is compressed and should be checked against the representative events.")
    return limitations


def _build_continuity_to_query(query_event: Event, representative_events: list[Event]) -> dict[str, Any]:
    query_src = str(query_event.attrs.get("src_account") or "")
    query_dst = str(query_event.attrs.get("dst_account") or "")
    same_source = 0
    same_destination = 0
    same_flow_pair = 0
    inbound_to_query_source = 0
    outbound_from_query_source = 0
    inbound_to_query_destination = 0
    outbound_from_query_destination = 0
    for event in representative_events:
        src = str(event.attrs.get("src_account") or "")
        dst = str(event.attrs.get("dst_account") or "")
        if query_src and src == query_src:
            same_source += 1
            outbound_from_query_source += 1
        if query_dst and dst == query_dst:
            same_destination += 1
            inbound_to_query_destination += 1
        if query_src and dst == query_src:
            inbound_to_query_source += 1
        if query_dst and src == query_dst:
            outbound_from_query_destination += 1
        if query_src and query_dst and src == query_src and dst == query_dst:
            same_flow_pair += 1

    connection_kinds = [
        name
        for name, count in (
            ("same_source", same_source),
            ("same_destination", same_destination),
            ("same_flow_pair", same_flow_pair),
            ("inbound_to_query_source", inbound_to_query_source),
            ("outbound_from_query_source", outbound_from_query_source),
            ("inbound_to_query_destination", inbound_to_query_destination),
            ("outbound_from_query_destination", outbound_from_query_destination),
        )
        if count > 0
    ]
    strength = round(min(
        1.0,
        0.30 * (1 if same_source > 0 else 0)
        + 0.30 * (1 if same_destination > 0 else 0)
        + 0.40 * (1 if same_flow_pair > 0 else 0)
        + 0.20 * (1 if inbound_to_query_source > 0 else 0)
        + 0.20 * (1 if inbound_to_query_destination > 0 else 0),
    ), 4)
    return {
        "connection_kinds": connection_kinds or ["none"],
        "strength": strength,
        "same_source_count": same_source,
        "same_destination_count": same_destination,
        "same_flow_pair_count": same_flow_pair,
        "inbound_to_query_source_count": inbound_to_query_source,
        "outbound_from_query_source_count": outbound_from_query_source,
        "inbound_to_query_destination_count": inbound_to_query_destination,
        "outbound_from_query_destination_count": outbound_from_query_destination,
    }


def _participant_overlap_with_query(query_event: Event, representative_events: list[Event]) -> float:
    query_entities = {
        str(query_event.attrs.get("src_account") or ""),
        str(query_event.attrs.get("dst_account") or ""),
    } - {""}
    if not query_entities or not representative_events:
        return 0.0
    candidate_entities = {
        str(value)
        for event in representative_events
        for value in (event.attrs.get("src_account") or "", event.attrs.get("dst_account") or "")
        if str(value)
    }
    if not candidate_entities:
        return 0.0
    return round(len(query_entities & candidate_entities) / len(query_entities), 4)


def _build_claim(
    query_event: Event,
    pattern: str,
    object_type: str,
    representative_events: list[Event],
) -> str:
    query_src = str(query_event.attrs.get("src_account") or "the source")
    query_dst = str(query_event.attrs.get("dst_account") or "the destination")
    if pattern == "source_accumulation_then_outflow":
        return f"Funds or activity accumulate into {query_src} before outward movement connected to the query."
    if pattern == "destination_concentration_with_pair_continuation":
        return f"{query_dst} already receives concentrated inbound activity and the query continues a known path into that destination."
    if pattern == "many_to_one_concentration":
        return f"{query_dst} acts as a concentration point before the query."
    if pattern == "same_pair_repeat":
        return f"The query repeats an earlier path between {query_src} and {query_dst}."
    if pattern == "source_dispersion":
        return f"{query_src} shows repeated outward movement before the query."
    if object_type == "skip":
        return "This skip object compresses a longer precursor path that may bridge distant context to the query."
    return "This object provides local precursor evidence that may contribute one step of the query's temporal story."


def _build_positive_evidence(
    query_event: Event,
    representative_events: list[Event],
    pattern: str,
    object_type: str,
    support_strength: dict[str, Any],
    continuity: dict[str, Any],
) -> list[str]:
    evidence: list[str] = []
    if continuity["same_flow_pair_count"] > 0:
        evidence.append("Contains a prior event on the same source-to-destination path as the query.")
    if continuity["inbound_to_query_source_count"] > 0:
        evidence.append("Shows prior inbound activity into the query source.")
    if continuity["outbound_from_query_source_count"] > 0:
        evidence.append("Shows prior outbound activity from the query source.")
    if continuity["inbound_to_query_destination_count"] > 0:
        evidence.append("Shows prior inbound activity into the query destination.")
    if pattern == "source_accumulation_then_outflow":
        evidence.append("Supports a source-accumulation-then-outflow story.")
    elif pattern == "destination_concentration_with_pair_continuation":
        evidence.append("Supports destination concentration plus continuation of a known path.")
    elif pattern == "many_to_one_concentration":
        evidence.append("Supports many-to-one concentration into the query destination.")
    elif pattern == "source_dispersion":
        evidence.append("Supports repeated outward movement from the query source.")
    if support_strength.get("bridge_score", 0.0) >= 0.3:
        evidence.append("Includes a non-trivial bridge score from a skip summary.")
    if "large_transfer" in support_strength.get("aspect_matches", []):
        evidence.append("Carries a large-transfer signal.")
    if support_strength.get("participant_overlap_with_query", 0.0) >= 0.5:
        evidence.append("Shares at least one key participant with the query.")
    if object_type in {"chain", "skip"} and representative_events:
        evidence.append("Summarizes a multi-step path rather than a single isolated event.")
    if not evidence:
        evidence.append("Provides at least one temporally prior event linked to the retrieved query context.")
    return evidence


def _card_confidence(
    pattern: str,
    object_type: str,
    support_strength: dict[str, Any],
    continuity: dict[str, Any],
    limitations: list[str],
) -> float:
    pattern_strength = {
        "source_accumulation_then_outflow": 0.90,
        "destination_concentration_with_pair_continuation": 0.80,
        "many_to_one_concentration": 0.65,
        "same_pair_repeat": 0.55,
        "source_dispersion": 0.60,
        "weak_or_fragmented_history": 0.35,
    }.get(pattern, 0.35)
    bridge_score = float(support_strength.get("bridge_score", 0.0))
    overlap = float(support_strength.get("participant_overlap_with_query", 0.0))
    rep_count = int(support_strength.get("representative_event_count", 0))
    aspect_bonus = min(1.0, len(support_strength.get("aspect_matches", [])) / 2.0)
    continuity_strength = float(continuity.get("strength", 0.0))
    structure_bonus = 0.15 if object_type in {"chain", "skip"} else 0.0
    limitation_penalty = min(0.30, 0.05 * len(limitations))
    score = (
        0.30 * pattern_strength
        + 0.20 * overlap
        + 0.15 * continuity_strength
        + 0.15 * min(1.0, rep_count / 3.0)
        + 0.10 * bridge_score
        + 0.10 * aspect_bonus
        + structure_bonus
        - limitation_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)


def _default_summary(pattern: str, representative_events: list[Event]) -> str:
    if representative_events:
        return f"{pattern} based on {len(representative_events)} representative events."
    return pattern


def _extract_bridge_score(summary: str) -> float | None:
    match = re.search(r"bridge=([0-9]+(?:\.[0-9]+)?)", summary or "")
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return None


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
