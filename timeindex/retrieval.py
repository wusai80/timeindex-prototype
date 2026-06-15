"""Dual-frontier retrieval for the TimeIndex prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import RetrievalConfig
from .event import ChainSummary, DecisionIntent, Event, EventQuery, EventRecord, EvidenceObject, OrdinaryLink, SkipLink
from .stores import ChainStore, EdgeStore, EventStore, SkipLinkStore


@dataclass(slots=True)
class _FrontierCandidate:
    """Internal retrieval candidate with bookkeeping for expansion."""

    evidence: EvidenceObject
    predecessor_id: str | None = None
    representative_event_ids: list[str] = field(default_factory=list)
    depth: int = 0
    kind: str = "ordinary"
    marginal_utility: float = 0.0
    priority: float = 0.0


class DualFrontierRetriever:
    """Retriever that traverses ordinary and skip frontiers together."""

    def __init__(
        self,
        event_store: EventStore,
        edge_store: EdgeStore,
        chain_store: ChainStore,
        skip_link_store: SkipLinkStore,
        config: RetrievalConfig,
    ) -> None:
        self.event_store = event_store
        self.edge_store = edge_store
        self.chain_store = chain_store
        self.skip_link_store = skip_link_store
        self.config = config

    def retrieve(self, query: EventQuery) -> Sequence[EvidenceObject]:
        return retrieve(self, query.event.event_id, query.intent, query.budget)


def retrieve(
    index: Any,
    query_event_id: str,
    intent: DecisionIntent,
    budget: int | float,
) -> list[EvidenceObject]:
    """Retrieve a budgeted set of evidence using ordinary and skip frontiers."""

    query_record = _get_event_record(index, query_event_id)
    if query_record is None:
        return []

    config = _get_retrieval_config(index)
    stop_threshold = _get_stop_threshold(index)
    selected: list[EvidenceObject] = []
    selected_object_ids: set[str] = set()
    ordinary_seen: set[tuple[str, str]] = set()
    skip_seen: set[tuple[str, str]] = set()
    spent_budget = 0.0

    ordinary_frontier = _initialize_local_frontier(index, query_record, intent, ordinary_seen)
    skip_frontier = _initialize_skip_frontier(index, query_record, intent, skip_seen)

    while spent_budget < float(budget):
        ordinary_candidate = _best_candidate(
            ordinary_frontier,
            selected,
            query_record,
            intent,
            selected_object_ids,
            config.priority_epsilon,
        )
        skip_candidate = _best_candidate(
            skip_frontier,
            selected,
            query_record,
            intent,
            selected_object_ids,
            config.priority_epsilon,
        )
        next_candidate = _pick_best_candidate(ordinary_candidate, skip_candidate)
        if next_candidate is None or next_candidate.marginal_utility < stop_threshold:
            break

        evidence = _materialize_evidence(next_candidate, budget - spent_budget, config)
        if evidence is None:
            _remove_candidate(ordinary_frontier, next_candidate)
            _remove_candidate(skip_frontier, next_candidate)
            continue

        selected.append(evidence)
        selected_object_ids.add(evidence.object_id)
        spent_budget += evidence.cost
        _remove_candidate(ordinary_frontier, next_candidate)
        _remove_candidate(skip_frontier, next_candidate)

        if next_candidate.kind == "ordinary" and next_candidate.predecessor_id is not None:
            ordinary_frontier.extend(
                _expand_ordinary_frontier(
                    index,
                    next_candidate.predecessor_id,
                    intent,
                    ordinary_seen,
                    depth=next_candidate.depth + 1,
                )
            )

    return selected


def _get_retrieval_config(index: Any) -> RetrievalConfig:
    config = getattr(index, "config", None)
    if isinstance(config, RetrievalConfig):
        return config
    retrieval_config = getattr(config, "retrieval", None)
    if isinstance(retrieval_config, RetrievalConfig):
        return retrieval_config
    return RetrievalConfig()


def _get_stop_threshold(index: Any) -> float:
    config = getattr(index, "config", None)
    scoring = getattr(config, "scoring", None)
    threshold = getattr(scoring, "retrieval_stop_threshold", None)
    if isinstance(threshold, (int, float)):
        return float(threshold)
    options = getattr(_get_retrieval_config(index), "options", {})
    option_threshold = options.get("retrieval_stop_threshold")
    if isinstance(option_threshold, (int, float)):
        return float(option_threshold)
    return 0.05


def _get_event_record(index: Any, event_id: str) -> EventRecord | None:
    for owner in (index, getattr(index, "event_store", None)):
        if owner is None:
            continue
        getter = getattr(owner, "get_event", None) or getattr(owner, "get", None)
        if getter is None:
            continue
        record = getter(event_id)
        if record is None:
            return None
        if isinstance(record, EventRecord):
            return record
        if isinstance(record, Event):
            return EventRecord(event=record)
    return None


def _get_incoming_links(store_owner: Any, event_id: str, skip: bool = False) -> Sequence[OrdinaryLink] | Sequence[SkipLink]:
    names = ("SkipIn", "skip_in", "incoming") if skip else ("In", "incoming")
    store = store_owner
    for name in names:
        method = getattr(store, name, None)
        if callable(method):
            return method(event_id)
    return []


def _get_chain_summaries(index: Any, tail_id: str) -> Sequence[ChainSummary]:
    chain_store = getattr(index, "chain_store", None)
    if chain_store is None:
        return []
    for name in ("get_for_tail", "get", "chains"):
        method = getattr(chain_store, name, None)
        if callable(method):
            summaries = method(tail_id)
            return list(summaries)
    return []


def _initialize_local_frontier(
    index: Any,
    query_record: EventRecord,
    intent: DecisionIntent,
    ordinary_seen: set[tuple[str, str]],
) -> list[_FrontierCandidate]:
    edge_store = getattr(index, "edge_store", index)
    incoming_links = _get_incoming_links(edge_store, query_record.event.event_id, skip=False)
    candidates: list[_FrontierCandidate] = []
    for link in incoming_links:
        key = (link.predecessor_id, link.successor_id)
        if key in ordinary_seen or link.predecessor_id == link.successor_id:
            continue
        ordinary_seen.add(key)
        candidates.append(_ordinary_candidate(index, link, intent, depth=0))
    return candidates


def _initialize_skip_frontier(
    index: Any,
    query_record: EventRecord,
    intent: DecisionIntent,
    skip_seen: set[tuple[str, str]],
) -> list[_FrontierCandidate]:
    skip_link_store = getattr(index, "skip_link_store", index)
    incoming_links = _get_incoming_links(skip_link_store, query_record.event.event_id, skip=True)
    candidates: list[_FrontierCandidate] = []
    for link in incoming_links:
        key = (link.from_id, link.to_id)
        if key in skip_seen or link.from_id == link.to_id:
            continue
        skip_seen.add(key)
        candidates.append(_skip_candidate(link, intent))
    return candidates


def _ordinary_candidate(
    index: Any,
    link: OrdinaryLink,
    intent: DecisionIntent,
    depth: int,
) -> _FrontierCandidate:
    summaries = _get_chain_summaries(index, link.successor_id)
    matching_summary = next(
        (
            summary
            for summary in summaries
            if summary.head_id == link.predecessor_id
            or (
                summary.representative_event_ids
                and summary.representative_event_ids[0] == link.predecessor_id
            )
        ),
        None,
    )
    if matching_summary is not None:
        event_ids = list(matching_summary.representative_event_ids) or [
            matching_summary.head_id,
            matching_summary.tail_id,
        ]
        summary_text = matching_summary.summary or f"Ordinary chain {matching_summary.chain_id}"
        cost = float(matching_summary.cost or max(1, len(event_ids)))
        aspects = set(matching_summary.aspects)
        object_id = matching_summary.chain_id
        predecessor_id = matching_summary.head_id
    else:
        event_ids = [link.predecessor_id, link.successor_id]
        summary_text = f"Ordinary evidence from {link.predecessor_id} to {link.successor_id}"
        cost = 1.0
        aspects = _collect_event_aspects(index, [link.predecessor_id, link.successor_id])
        object_id = f"ordinary:{link.predecessor_id}->{link.successor_id}"
        predecessor_id = link.predecessor_id

    if not aspects:
        aspects = set(intent.aspects) if intent.aspects else {"ordinary_evidence"}

    evidence = EvidenceObject(
        object_id=object_id,
        event_ids=event_ids,
        aspects=aspects,
        summary=summary_text,
        cost=cost,
    )
    return _FrontierCandidate(
        evidence=evidence,
        predecessor_id=predecessor_id,
        representative_event_ids=list(event_ids),
        depth=depth,
        kind="ordinary",
    )


def _skip_candidate(link: SkipLink, intent: DecisionIntent) -> _FrontierCandidate:
    aspects = set(link.aspects) or set(intent.aspects) or {"skip_evidence"}
    evidence = EvidenceObject(
        object_id=f"skip:{link.from_id}->{link.to_id}",
        event_ids=[],
        aspects=aspects,
        summary=link.summary or f"Skip summary from {link.from_id} to {link.to_id}",
        cost=float(link.cost or 1.0),
    )
    representative_ids = list(link.representative_event_ids)
    if not representative_ids:
        representative_ids = [link.from_id]
    return _FrontierCandidate(
        evidence=evidence,
        predecessor_id=link.from_id,
        representative_event_ids=representative_ids,
        depth=0,
        kind="skip",
    )


def _best_candidate(
    frontier: list[_FrontierCandidate],
    selected: Sequence[EvidenceObject],
    query_record: EventRecord,
    intent: DecisionIntent,
    selected_object_ids: set[str],
    epsilon: float,
) -> _FrontierCandidate | None:
    best: _FrontierCandidate | None = None
    best_key: tuple[float, float, str] | None = None
    for candidate in frontier:
        if candidate.evidence.object_id in selected_object_ids:
            continue
        utility = _marginal_utility(candidate, selected, query_record, intent)
        priority = utility / (candidate.evidence.cost + epsilon)
        candidate.marginal_utility = utility
        candidate.priority = priority
        ranking_key = (priority, utility, candidate.evidence.object_id)
        if best is None or ranking_key > best_key:
            best = candidate
            best_key = ranking_key
    return best


def _pick_best_candidate(
    ordinary_candidate: _FrontierCandidate | None,
    skip_candidate: _FrontierCandidate | None,
) -> _FrontierCandidate | None:
    if ordinary_candidate is None:
        return skip_candidate
    if skip_candidate is None:
        return ordinary_candidate
    ordinary_key = (
        ordinary_candidate.priority,
        ordinary_candidate.marginal_utility,
        ordinary_candidate.evidence.object_id,
    )
    skip_key = (
        skip_candidate.priority,
        skip_candidate.marginal_utility,
        skip_candidate.evidence.object_id,
    )
    if skip_key > ordinary_key:
        return skip_candidate
    return ordinary_candidate


def _materialize_evidence(
    candidate: _FrontierCandidate,
    remaining_budget: int | float,
    config: RetrievalConfig,
) -> EvidenceObject | None:
    if candidate.kind != "skip":
        if candidate.evidence.cost > float(remaining_budget):
            return None
        return candidate.evidence

    base_cost = candidate.evidence.cost
    if base_cost > float(remaining_budget):
        return None

    event_ids: list[str] = []
    total_cost = base_cost
    if config.return_summaries:
        event_ids = []
    if config.allow_skip_expansion and candidate.representative_event_ids:
        expansion_cost = max(0.0, float(len(candidate.representative_event_ids) - 1))
        if total_cost + expansion_cost <= float(remaining_budget):
            event_ids = list(candidate.representative_event_ids)
            total_cost += expansion_cost

    return EvidenceObject(
        object_id=candidate.evidence.object_id,
        event_ids=event_ids,
        aspects=set(candidate.evidence.aspects),
        summary=candidate.evidence.summary,
        cost=total_cost,
    )


def _expand_ordinary_frontier(
    index: Any,
    event_id: str,
    intent: DecisionIntent,
    ordinary_seen: set[tuple[str, str]],
    depth: int,
) -> list[_FrontierCandidate]:
    edge_store = getattr(index, "edge_store", index)
    incoming_links = _get_incoming_links(edge_store, event_id, skip=False)
    candidates: list[_FrontierCandidate] = []
    for link in incoming_links:
        key = (link.predecessor_id, link.successor_id)
        if key in ordinary_seen or link.predecessor_id == link.successor_id:
            continue
        ordinary_seen.add(key)
        candidates.append(_ordinary_candidate(index, link, intent, depth=depth))
    return candidates


def _marginal_utility(
    candidate: _FrontierCandidate,
    selected: Sequence[EvidenceObject],
    query_record: EventRecord,
    intent: DecisionIntent,
) -> float:
    selected_event_ids = {event_id for evidence in selected for event_id in evidence.event_ids}
    selected_aspects = {aspect for evidence in selected for aspect in evidence.aspects}

    aspect_overlap = _weighted_aspect_overlap(candidate.evidence.aspects, intent)
    aspect_novelty = _novelty(candidate.evidence.aspects, selected_aspects)
    event_novelty = _novelty(set(candidate.evidence.event_ids), selected_event_ids)
    query_relevance = _query_overlap(candidate.evidence.event_ids, query_record.event.event_id)
    depth_bonus = 1.0 / float(candidate.depth + 1)

    utility = (
        0.45 * aspect_overlap
        + 0.25 * aspect_novelty
        + 0.20 * event_novelty
        + 0.05 * query_relevance
        + 0.05 * depth_bonus
    )
    return max(0.0, min(1.0, utility))


def _weighted_aspect_overlap(candidate_aspects: set[str], intent: DecisionIntent) -> float:
    if not candidate_aspects:
        return 0.0
    if not intent.aspects:
        return 0.5
    total_weight = 0.0
    matched_weight = 0.0
    for aspect in intent.aspects:
        weight = float(intent.aspect_weights.get(aspect, 1.0))
        total_weight += weight
        if aspect in candidate_aspects:
            matched_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return matched_weight / total_weight


def _novelty(values: set[str], already_selected: set[str]) -> float:
    if not values:
        return 0.0
    unseen = values - already_selected
    return len(unseen) / len(values)


def _query_overlap(candidate_event_ids: Sequence[str], query_event_id: str) -> float:
    if not candidate_event_ids:
        return 0.5
    return 0.0 if query_event_id in candidate_event_ids else 1.0


def _collect_event_aspects(index: Any, event_ids: Sequence[str]) -> set[str]:
    aspects: set[str] = set()
    for event_id in event_ids:
        record = _get_event_record(index, event_id)
        if record is not None:
            aspects.update(record.aspects)
    return aspects


def _remove_candidate(frontier: list[_FrontierCandidate], candidate: _FrontierCandidate) -> None:
    if candidate in frontier:
        frontier.remove(candidate)
