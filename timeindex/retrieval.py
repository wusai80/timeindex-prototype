"""Dual-frontier retrieval for the TimeIndex prototype."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import RetrievalConfig, ScoringConfig
from .event import ChainSummary, DecisionIntent, Event, EventQuery, EventRecord, EvidenceObject, OrdinaryLink, SkipLink
from .scoring import retrieval_marginal_utility
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
    bridge_score: float = 0.0
    density_score: float = 0.0
    generic_penalty: float = 0.0


@dataclass(slots=True)
class _LookupCache:
    """Per-retrieval cache to avoid repeated backend lookups."""

    event_records: dict[str, EventRecord | None] = field(default_factory=dict)
    chain_summaries: dict[str, Sequence[ChainSummary]] = field(default_factory=dict)


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
    lookup_cache = _LookupCache(event_records={query_event_id: query_record})
    query_time_key = _time_sort_key(query_record.event.time)
    selected: list[EvidenceObject] = []
    selected_object_ids: set[str] = set()
    ordinary_seen: set[tuple[str, str]] = set()
    skip_seen: set[tuple[str, str]] = set()
    spent_budget = 0.0
    expansions = 0
    scoring_config = _get_scoring_config(index)

    frontier: list[_FrontierCandidate] = []
    _enqueue_candidates(
        frontier,
        _initialize_local_frontier(index, query_record, intent, ordinary_seen, lookup_cache, query_time_key),
        selected,
        query_record,
        intent,
        selected_object_ids,
        config,
        scoring_config,
    )
    _enqueue_candidates(
        frontier,
        _initialize_skip_frontier(index, query_record, intent, skip_seen, lookup_cache, query_time_key),
        selected,
        query_record,
        intent,
        selected_object_ids,
        config,
        scoring_config,
    )

    while spent_budget < float(budget) and frontier and expansions < max(1, config.max_search_expansions):
        ordinary_candidate = _best_candidate(frontier, "ordinary")
        skip_candidate = _best_candidate(frontier, "skip")
        next_candidate = _pick_best_candidate(ordinary_candidate, skip_candidate, config)
        if next_candidate is None or next_candidate.marginal_utility < stop_threshold:
            break
        if next_candidate.kind == "skip" and not _should_select_skip_candidate(next_candidate, selected, intent):
            _remove_candidate(frontier, next_candidate)
            continue

        evidence = _materialize_evidence(next_candidate, budget - spent_budget, config)
        if evidence is None:
            _remove_candidate(frontier, next_candidate)
            continue

        selected.append(evidence)
        selected_object_ids.add(evidence.object_id)
        spent_budget += evidence.cost
        _remove_candidate(frontier, next_candidate)
        expansions += 1

        if (
            next_candidate.kind == "ordinary"
            and next_candidate.predecessor_id is not None
            and next_candidate.depth < max(0, config.max_depth)
        ):
            _enqueue_candidates(
                frontier,
                _expand_ordinary_frontier(
                    index,
                    next_candidate.predecessor_id,
                    query_record,
                    intent,
                    ordinary_seen,
                    lookup_cache,
                    query_time_key,
                    depth=next_candidate.depth + 1,
                    branch_factor=config.max_branch_factor,
                )
                ,
                selected,
                query_record,
                intent,
                selected_object_ids,
                config,
                scoring_config,
            )
            _enqueue_candidates(
                frontier,
                _expand_skip_frontier(
                    index,
                    next_candidate.predecessor_id,
                    query_record,
                    intent,
                    skip_seen,
                    lookup_cache,
                    query_time_key,
                    depth=next_candidate.depth + 1,
                ),
                selected,
                query_record,
                intent,
                selected_object_ids,
                config,
                scoring_config,
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


def _get_scoring_config(index: Any) -> ScoringConfig:
    config = getattr(index, "config", None)
    scoring = getattr(config, "scoring", None)
    if isinstance(scoring, ScoringConfig):
        return scoring
    return ScoringConfig()


def _get_event_record(
    index: Any,
    event_id: str,
    lookup_cache: _LookupCache | None = None,
) -> EventRecord | None:
    if lookup_cache is not None and event_id in lookup_cache.event_records:
        return lookup_cache.event_records[event_id]

    for owner in (index, getattr(index, "event_store", None)):
        if owner is None:
            continue
        getter = getattr(owner, "get_event", None) or getattr(owner, "get", None)
        if getter is None:
            continue
        record = getter(event_id)
        if record is None:
            if lookup_cache is not None:
                lookup_cache.event_records[event_id] = None
            return None
        if isinstance(record, EventRecord):
            if lookup_cache is not None:
                lookup_cache.event_records[event_id] = record
            return record
        if isinstance(record, Event):
            wrapped_record = EventRecord(event=record)
            if lookup_cache is not None:
                lookup_cache.event_records[event_id] = wrapped_record
            return wrapped_record
    return None


def _get_incoming_links(store_owner: Any, event_id: str, skip: bool = False) -> Sequence[OrdinaryLink] | Sequence[SkipLink]:
    names = ("SkipIn", "skip_in", "incoming") if skip else ("In", "incoming")
    store = store_owner
    for name in names:
        method = getattr(store, name, None)
        if callable(method):
            return method(event_id)
    return []


def _get_chain_summaries(
    index: Any,
    tail_id: str,
    lookup_cache: _LookupCache | None = None,
) -> Sequence[ChainSummary]:
    if lookup_cache is not None and tail_id in lookup_cache.chain_summaries:
        return lookup_cache.chain_summaries[tail_id]

    chain_store = getattr(index, "chain_store", None)
    if chain_store is None:
        return []
    for name in ("get_for_tail", "get", "chains"):
        method = getattr(chain_store, name, None)
        if callable(method):
            summaries = method(tail_id)
            summary_list = list(summaries)
            if lookup_cache is not None:
                lookup_cache.chain_summaries[tail_id] = summary_list
            return summary_list
    return []


def _initialize_local_frontier(
    index: Any,
    query_record: EventRecord,
    intent: DecisionIntent,
    ordinary_seen: set[tuple[str, str]],
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
) -> list[_FrontierCandidate]:
    edge_store = getattr(index, "edge_store", index)
    incoming_links = _get_incoming_links(edge_store, query_record.event.event_id, skip=False)
    candidates: list[_FrontierCandidate] = []
    for link in incoming_links:
        key = (link.predecessor_id, link.successor_id)
        if key in ordinary_seen or link.predecessor_id == link.successor_id:
            continue
        ordinary_seen.add(key)
        candidate = _ordinary_candidate(index, link, query_record, intent, lookup_cache, query_time_key, depth=0)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _initialize_skip_frontier(
    index: Any,
    query_record: EventRecord,
    intent: DecisionIntent,
    skip_seen: set[tuple[str, str]],
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
) -> list[_FrontierCandidate]:
    return _skip_frontier_for_target(
        index,
        query_record.event.event_id,
        query_record,
        intent,
        skip_seen,
        lookup_cache,
        query_time_key,
        depth=0,
    )


def _skip_frontier_for_target(
    index: Any,
    target_event_id: str,
    query_record: EventRecord,
    intent: DecisionIntent,
    skip_seen: set[tuple[str, str]],
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
    depth: int,
) -> list[_FrontierCandidate]:
    skip_link_store = getattr(index, "skip_link_store", index)
    incoming_links = _get_incoming_links(skip_link_store, target_event_id, skip=True)
    candidates: list[_FrontierCandidate] = []
    for link in incoming_links:
        key = (link.from_id, link.to_id)
        if key in skip_seen or link.from_id == link.to_id:
            continue
        skip_seen.add(key)
        candidate = _skip_candidate(index, link, query_record, intent, lookup_cache, query_time_key, depth=depth)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _ordinary_candidate(
    index: Any,
    link: OrdinaryLink,
    query_record: EventRecord,
    intent: DecisionIntent,
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
    depth: int,
) -> _FrontierCandidate | None:
    summaries = _get_chain_summaries(index, link.successor_id, lookup_cache)
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
        event_ids = _causal_event_ids(index, event_ids, query_record, lookup_cache, query_time_key)
        if not event_ids:
            event_ids = _causal_event_ids(index, [matching_summary.head_id], query_record, lookup_cache, query_time_key)
        if not event_ids:
            return None
        cost = float(max(1, len(event_ids)))
        aspects = set(matching_summary.aspects)
        object_id = matching_summary.chain_id
        predecessor_id = matching_summary.head_id
        base_summary = matching_summary.summary or f"Ordinary chain {matching_summary.chain_id}"
    else:
        event_ids = _causal_event_ids(index, [link.predecessor_id], query_record, lookup_cache, query_time_key)
        if not event_ids:
            return None
        cost = 1.0
        aspects = _collect_event_aspects(index, [link.predecessor_id, link.successor_id], lookup_cache)
        object_id = f"ordinary:{link.predecessor_id}->{link.successor_id}"
        predecessor_id = link.predecessor_id
        base_summary = f"Ordinary evidence from {link.predecessor_id} to {link.successor_id}"

    if not aspects:
        aspects = set(intent.aspects) if intent.aspects else {"ordinary_evidence"}

    representative_records = _records_for_ids(index, event_ids, lookup_cache)
    bridge_score = _bridge_score(index, event_ids, query_record, lookup_cache)
    lanl_aspects = _derive_lanl_aspects(query_record, representative_records)
    if lanl_aspects:
        aspects.update(lanl_aspects)
    summary_text = _candidate_summary(
        kind="ordinary",
        query_record=query_record,
        representative_records=representative_records,
        base_summary=base_summary,
        bridge_score=bridge_score,
    )

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
        bridge_score=bridge_score,
        density_score=_density_score(index, event_ids, lookup_cache),
        generic_penalty=_generic_penalty(aspects),
    )


def _skip_candidate(
    index: Any,
    link: SkipLink,
    query_record: EventRecord,
    intent: DecisionIntent,
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
    depth: int,
) -> _FrontierCandidate | None:
    aspects = set(link.aspects) or set(intent.aspects) or {"skip_evidence"}
    representative_ids = _causal_event_ids(index, list(link.representative_event_ids), query_record, lookup_cache, query_time_key)
    if not representative_ids:
        representative_ids = _causal_event_ids(index, [link.from_id], query_record, lookup_cache, query_time_key)
    if not representative_ids:
        return None
    representative_records = _records_for_ids(index, representative_ids, lookup_cache)
    bridge_score = max(
        _bridge_score(index, representative_ids, query_record, lookup_cache),
        _entity_bridge_score(link, query_record),
    )
    lanl_aspects = _derive_lanl_aspects(query_record, representative_records)
    if lanl_aspects:
        aspects.update(lanl_aspects)
    evidence = EvidenceObject(
        object_id=f"skip:{link.from_id}->{link.to_id}",
        event_ids=[],
        aspects=aspects,
        summary=_candidate_summary(
            kind="skip",
            query_record=query_record,
            representative_records=representative_records,
            base_summary=link.summary or f"Skip summary from {link.from_id} to {link.to_id}",
            bridge_score=bridge_score,
        ),
        cost=float(link.cost or 1.0),
    )
    navigation_id = representative_ids[-1] if representative_ids else link.from_id
    return _FrontierCandidate(
        evidence=evidence,
        predecessor_id=navigation_id,
        representative_event_ids=representative_ids,
        depth=depth,
        kind="skip",
        bridge_score=bridge_score,
        density_score=_density_score(index, representative_ids, lookup_cache),
        generic_penalty=_generic_penalty(aspects),
    )


def _best_candidate(frontier: list[_FrontierCandidate], kind: str) -> _FrontierCandidate | None:
    best: _FrontierCandidate | None = None
    best_key: tuple[float, float, str] | None = None
    for candidate in frontier:
        if candidate.kind != kind:
            continue
        ranking_key = (candidate.priority, candidate.marginal_utility, candidate.evidence.object_id)
        if best is None or ranking_key > best_key:
            best = candidate
            best_key = ranking_key
    return best


def _pick_best_candidate(
    ordinary_candidate: _FrontierCandidate | None,
    skip_candidate: _FrontierCandidate | None,
    config: RetrievalConfig,
) -> _FrontierCandidate | None:
    if ordinary_candidate is None:
        return skip_candidate
    if skip_candidate is None:
        return ordinary_candidate
    if ordinary_candidate.marginal_utility <= 0.0:
        return skip_candidate
    if skip_candidate.priority >= ordinary_candidate.priority * max(config.skip_competitive_ratio, 0.0):
        return skip_candidate
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
        event_ids = list(candidate.representative_event_ids[:1])
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
    query_record: EventRecord,
    intent: DecisionIntent,
    ordinary_seen: set[tuple[str, str]],
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
    depth: int,
    branch_factor: int,
) -> list[_FrontierCandidate]:
    edge_store = getattr(index, "edge_store", index)
    incoming_links = _get_incoming_links(edge_store, event_id, skip=False)
    candidates: list[_FrontierCandidate] = []
    ranked_links = sorted(
        incoming_links,
        key=lambda item: (-item.score, item.predecessor_id, item.successor_id),
    )[: max(1, branch_factor)]
    for link in ranked_links:
        key = (link.predecessor_id, link.successor_id)
        if key in ordinary_seen or link.predecessor_id == link.successor_id:
            continue
        ordinary_seen.add(key)
        candidate = _ordinary_candidate(index, link, query_record, intent, lookup_cache, query_time_key, depth=depth)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _expand_skip_frontier(
    index: Any,
    event_id: str,
    query_record: EventRecord,
    intent: DecisionIntent,
    skip_seen: set[tuple[str, str]],
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
    depth: int,
) -> list[_FrontierCandidate]:
    target_record = _get_event_record(index, event_id, lookup_cache)
    if target_record is None:
        return []
    return _skip_frontier_for_target(
        index,
        target_record.event.event_id,
        query_record,
        intent,
        skip_seen,
        lookup_cache,
        query_time_key,
        depth=depth,
    )


def _marginal_utility(
    candidate: _FrontierCandidate,
    selected: Sequence[EvidenceObject],
    query_record: EventRecord,
    intent: DecisionIntent,
    config: ScoringConfig,
) -> float:
    utility = retrieval_marginal_utility(
        candidate.evidence,
        selected,
        EventQuery(event=query_record.event, intent=intent, budget=0),
        intent,
        config,
    )
    depth_bonus = 1.0 / float(candidate.depth + 1)
    query_relevance = _query_overlap(candidate.evidence.event_ids, query_record.event.event_id)
    if candidate.kind != "skip":
        utility = (
            0.55 * utility
            + 0.15 * query_relevance
            + 0.10 * depth_bonus
            + 0.15 * candidate.bridge_score
            + 0.10 * candidate.density_score
            - 0.15 * candidate.generic_penalty
        )
        return max(0.0, min(1.0, utility))

    support_score = _skip_support_score(candidate)
    specificity_score = _skip_specificity_score(candidate.evidence.aspects)
    novel_specific_gain = _skip_novel_specific_gain(candidate.evidence.aspects, selected, intent)
    skip_bonus = 0.05 if candidate.bridge_score >= 0.85 and support_score >= 0.50 else 0.0
    utility = (
        0.40 * utility
        + 0.10 * query_relevance
        + 0.05 * depth_bonus
        + 0.20 * candidate.bridge_score
        + 0.10 * candidate.density_score
        + 0.10 * support_score
        + 0.10 * specificity_score
        + 0.15 * novel_specific_gain
        + skip_bonus
        - 0.25 * candidate.generic_penalty
    )
    return max(0.0, min(1.0, utility))


def _enqueue_candidates(
    frontier: list[_FrontierCandidate],
    candidates: Sequence[_FrontierCandidate],
    selected: Sequence[EvidenceObject],
    query_record: EventRecord,
    intent: DecisionIntent,
    selected_object_ids: set[str],
    config: RetrievalConfig,
    scoring_config: ScoringConfig,
) -> None:
    best_by_signature: dict[tuple[str, str | None], _FrontierCandidate] = {
        _candidate_signature(candidate): candidate
        for candidate in frontier
    }
    for candidate in candidates:
        if candidate.evidence.object_id in selected_object_ids:
            continue
        candidate.marginal_utility = _marginal_utility(candidate, selected, query_record, intent, scoring_config)
        candidate.priority = candidate.marginal_utility / (candidate.evidence.cost + config.priority_epsilon)
        if candidate.kind == "skip":
            candidate.priority *= _skip_priority_multiplier(candidate, selected, intent)
        signature = _candidate_signature(candidate)
        existing = best_by_signature.get(signature)
        if existing is not None and (
            existing.priority > candidate.priority
            or (
                existing.priority == candidate.priority
                and existing.marginal_utility >= candidate.marginal_utility
            )
        ):
            continue
        if existing is not None:
            _remove_candidate(frontier, existing)
        frontier.append(candidate)
        best_by_signature[signature] = candidate

    frontier.sort(key=lambda item: (-item.priority, -item.marginal_utility, item.evidence.object_id))
    del frontier[max(1, config.max_frontier_size) :]


def _candidate_signature(candidate: _FrontierCandidate) -> tuple[str, str | None]:
    if candidate.kind == "skip":
        return (candidate.kind, candidate.evidence.object_id)
    return (candidate.kind, candidate.predecessor_id or candidate.evidence.object_id)


def _bridge_score(
    index: Any,
    event_ids: Sequence[str],
    query_record: EventRecord,
    lookup_cache: _LookupCache,
) -> float:
    best = 0.0
    for event_id in event_ids:
        record = _get_event_record(index, event_id, lookup_cache)
        if record is None:
            continue
        best = max(best, _record_bridge_score(record, query_record))
    return best


def _record_bridge_score(record: EventRecord, query_record: EventRecord) -> float:
    source = set(record.source_entities)
    destination = set(record.destination_entities)
    query_source = set(query_record.source_entities)
    query_destination = set(query_record.destination_entities)
    if destination & query_source:
        return 1.0
    if source & query_source:
        return 0.8
    if destination & query_destination:
        return 0.65
    if source & query_destination:
        return 0.45
    return 0.0


def _entity_bridge_score(link: SkipLink, query_record: EventRecord) -> float:
    source = set(getattr(link, "source_entities", ()))
    destination = set(getattr(link, "destination_entities", ()))
    query_source = set(query_record.source_entities)
    query_destination = set(query_record.destination_entities)
    if destination & query_source:
        return 1.0
    if source & query_source:
        return 0.8
    if destination & query_destination:
        return 0.65
    if source & query_destination:
        return 0.45
    return 0.0


def _density_score(index: Any, event_ids: Sequence[str], lookup_cache: _LookupCache) -> float:
    if not event_ids:
        return 0.0
    source_values: set[str] = set()
    destination_values: set[str] = set()
    for event_id in event_ids:
        record = _get_event_record(index, event_id, lookup_cache)
        if record is None:
            continue
        source_values.update(str(value) for value in record.source_entities)
        destination_values.update(str(value) for value in record.destination_entities)
    support = max(len(event_ids), len(source_values), len(destination_values))
    return max(0.0, min(1.0, support / 4.0))


def _generic_penalty(aspects: set[str]) -> float:
    if not aspects:
        return 1.0
    if aspects == {"generic_evidence"}:
        return 1.0
    if "generic_evidence" in aspects and len(aspects) == 1:
        return 1.0
    if "generic_evidence" in aspects:
        return 0.5
    return 0.0


def _specific_aspects(aspects: set[str]) -> set[str]:
    return {
        aspect
        for aspect in aspects
        if aspect not in {"generic_evidence", "ordinary_evidence", "skip_evidence"}
    }


def _skip_support_score(candidate: _FrontierCandidate) -> float:
    return max(0.0, min(1.0, float(len(candidate.representative_event_ids)) / 4.0))


def _skip_specificity_score(aspects: set[str]) -> float:
    if not aspects:
        return 0.0
    specific = _specific_aspects(aspects)
    return len(specific) / len(aspects)


def _skip_novel_specific_gain(
    aspects: set[str],
    selected: Sequence[EvidenceObject],
    intent: DecisionIntent,
) -> float:
    specific = _specific_aspects(aspects)
    if not specific:
        return 0.0
    selected_specific = set().union(*(_specific_aspects(item.aspects) for item in selected)) if selected else set()
    novel = specific - selected_specific
    if not novel:
        return 0.0
    if not intent.aspects:
        return 1.0
    return 1.0 if (novel & set(intent.aspects)) else 0.0


def _skip_priority_multiplier(
    candidate: _FrontierCandidate,
    selected: Sequence[EvidenceObject],
    intent: DecisionIntent,
) -> float:
    support_score = _skip_support_score(candidate)
    specificity_score = _skip_specificity_score(candidate.evidence.aspects)
    novel_specific_gain = _skip_novel_specific_gain(candidate.evidence.aspects, selected, intent)
    multiplier = (
        0.10
        + 0.40 * candidate.bridge_score
        + 0.15 * candidate.density_score
        + 0.15 * support_score
        + 0.20 * novel_specific_gain
        + 0.10 * specificity_score
        - 0.30 * candidate.generic_penalty
    )
    return max(0.05, min(1.0, multiplier))


def _should_select_skip_candidate(
    candidate: _FrontierCandidate,
    selected: Sequence[EvidenceObject],
    intent: DecisionIntent,
) -> bool:
    candidate_specific = _specific_aspects(candidate.evidence.aspects)
    selected_aspects = set().union(*(item.aspects for item in selected)) if selected else set()
    selected_specific = _specific_aspects(selected_aspects)

    if not intent.aspects:
        return candidate.bridge_score >= 0.9 and _skip_support_score(candidate) >= 0.5

    intent_aspects = set(intent.aspects)
    uncovered_intent_aspects = intent_aspects - selected_aspects
    new_intent_aspects = (candidate_specific & intent_aspects) - selected_specific
    if new_intent_aspects:
        return True

    if not uncovered_intent_aspects:
        return False

    return (
        candidate.bridge_score >= 0.95
        and _skip_support_score(candidate) >= 0.75
        and candidate.generic_penalty <= 0.0
    )


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


def _collect_event_aspects(
    index: Any,
    event_ids: Sequence[str],
    lookup_cache: _LookupCache,
) -> set[str]:
    aspects: set[str] = set()
    for event_id in event_ids:
        record = _get_event_record(index, event_id, lookup_cache)
        if record is not None:
            aspects.update(record.aspects)
    return aspects


def _records_for_ids(
    index: Any,
    event_ids: Sequence[str],
    lookup_cache: _LookupCache,
) -> list[EventRecord]:
    records: list[EventRecord] = []
    for event_id in event_ids:
        record = _get_event_record(index, event_id, lookup_cache)
        if record is not None:
            records.append(record)
    return records


def _candidate_summary(
    *,
    kind: str,
    query_record: EventRecord,
    representative_records: Sequence[EventRecord],
    base_summary: str,
    bridge_score: float,
) -> str:
    if not representative_records or not _is_lanl_record(query_record):
        return base_summary

    stats = _lanl_candidate_stats(query_record, representative_records)
    sanitized_base = _strip_bridge_annotation(base_summary)
    pattern = str(stats["pattern"])
    kind_label = "skip" if kind == "skip" else ("chain" if len(representative_records) > 1 else "event")
    detached = "true" if bool(stats["detached_from_query"]) else "false"
    dst_precursor = "true" if bool(stats["query_dst_precursor"]) else "false"
    query_host_touch = "true" if bool(stats["query_host_touch"]) else "false"
    same_user = int(stats["same_user_count"])
    same_src = int(stats["same_src_count"])
    same_dst = int(stats["same_dst_count"])
    real_targets = int(stats["distinct_real_targets_from_query_src"])
    span_seconds = int(round(float(stats["span_seconds"])))
    return (
        f"LANL {kind_label} pattern={pattern}; bridge={bridge_score:.2f}; "
        f"same_user={same_user}; same_src_host={same_src}; same_dst_host={same_dst}; "
        f"real_targets_from_query_src={real_targets}; query_dst_precursor={dst_precursor}; "
        f"query_host_touch={query_host_touch}; detached_from_query={detached}; "
        f"span_seconds={span_seconds}; base={sanitized_base}"
    )


def _derive_lanl_aspects(
    query_record: EventRecord,
    representative_records: Sequence[EventRecord],
) -> set[str]:
    if not representative_records or not _is_lanl_record(query_record):
        return set()

    stats = _lanl_candidate_stats(query_record, representative_records)
    aspects: set[str] = {str(stats["pattern"])}
    if bool(stats["query_dst_precursor"]):
        aspects.add("lanl_query_dst_precursor")
    if bool(stats["query_host_touch"]):
        aspects.add("lanl_query_host_touch")
    if int(stats["distinct_real_targets_from_query_src"]) >= 2:
        aspects.add("lanl_source_host_fanout")
    if int(stats["same_user_real_targets"]) >= 2:
        aspects.add("lanl_credential_reuse")
    if bool(stats["detached_from_query"]):
        aspects.add("lanl_detached_history")
    if float(stats["span_seconds"]) >= 60.0:
        aspects.add("lanl_temporal_bridge")
    if len(representative_records) >= 2:
        aspects.add("lanl_multi_step_context")
    return {aspect for aspect in aspects if aspect}


def _lanl_candidate_stats(
    query_record: EventRecord,
    representative_records: Sequence[EventRecord],
) -> dict[str, int | float | bool | str]:
    query_user = str(query_record.event.attrs.get("src_user") or "")
    query_src = str(query_record.event.attrs.get("src_computer") or "")
    query_dst = str(query_record.event.attrs.get("dst_computer") or "")
    query_auth = str(query_record.event.attrs.get("auth_type") or "")
    same_user_count = 0
    same_src_count = 0
    same_dst_count = 0
    same_path_count = 0
    same_auth_count = 0
    query_host_touch_count = 0
    real_targets_from_query_src: set[str] = set()
    same_user_real_targets: set[str] = set()
    time_values: list[float] = []

    for record in representative_records:
        attrs = record.event.attrs
        src_user = str(attrs.get("src_user") or "")
        src_host = str(attrs.get("src_computer") or "")
        dst_host = str(attrs.get("dst_computer") or "")
        auth_type = str(attrs.get("auth_type") or "")
        parsed_time = _parse_time_value(record.event.time)
        if parsed_time is not None:
            time_values.append(parsed_time)
        if query_user and src_user == query_user:
            same_user_count += 1
            if _is_real_host(dst_host, src_host):
                same_user_real_targets.add(dst_host)
        if query_src and src_host == query_src:
            same_src_count += 1
            if _is_real_host(dst_host, src_host):
                real_targets_from_query_src.add(dst_host)
        if query_dst and dst_host == query_dst:
            same_dst_count += 1
        if query_auth and auth_type == query_auth:
            same_auth_count += 1
        if query_user and query_src and query_dst and src_user == query_user and src_host == query_src and dst_host == query_dst:
            same_path_count += 1
        if (query_src and (src_host == query_src or dst_host == query_src)) or (
            query_dst and (src_host == query_dst or dst_host == query_dst)
        ):
            query_host_touch_count += 1

    query_dst_precursor = same_dst_count > 0
    query_host_touch = query_host_touch_count > 0
    detached_from_query = not query_dst_precursor and not query_host_touch
    span_seconds = (max(time_values) - min(time_values)) if len(time_values) >= 2 else 0.0

    pattern = "lanl_weak_or_fragmented_history"
    if same_src_count >= 2 and len(real_targets_from_query_src) >= 2:
        pattern = "lanl_source_host_fanout"
    elif same_user_count >= 2 and len(same_user_real_targets) >= 2:
        pattern = "lanl_credential_reuse_across_hosts"
    elif query_dst_precursor and query_host_touch:
        pattern = "lanl_bridge_into_query_host"
    elif same_path_count >= 1:
        pattern = "lanl_same_path_repeat"
    elif query_host_touch:
        pattern = "lanl_query_host_touch"
    elif same_user_count >= 1:
        pattern = "lanl_user_continuity"
    if detached_from_query and pattern in {
        "lanl_source_host_fanout",
        "lanl_credential_reuse_across_hosts",
        "lanl_user_continuity",
    }:
        pattern = f"{pattern}_detached"

    return {
        "pattern": pattern,
        "same_user_count": same_user_count,
        "same_src_count": same_src_count,
        "same_dst_count": same_dst_count,
        "same_path_count": same_path_count,
        "same_auth_count": same_auth_count,
        "query_host_touch_count": query_host_touch_count,
        "query_dst_precursor": query_dst_precursor,
        "query_host_touch": query_host_touch,
        "detached_from_query": detached_from_query,
        "distinct_real_targets_from_query_src": len(real_targets_from_query_src),
        "same_user_real_targets": len(same_user_real_targets),
        "span_seconds": span_seconds,
    }


def _is_lanl_record(record: EventRecord) -> bool:
    attrs = record.event.attrs
    return any(
        key in attrs
        for key in ("src_computer", "dst_computer", "src_user", "auth_type", "logon_type")
    )


def _is_tgt_host(value: str) -> bool:
    return value.strip().lower() == "tgt"


def _is_real_host(dst_host: str, src_host: str) -> bool:
    if not dst_host:
        return False
    if _is_tgt_host(dst_host):
        return False
    return dst_host != src_host


def _strip_bridge_annotation(summary: str) -> str:
    return re.sub(r"\s*\[bridge=[0-9]+(?:\.[0-9]+)?\]\s*", " ", summary or "").strip()


def _remove_candidate(frontier: list[_FrontierCandidate], candidate: _FrontierCandidate) -> None:
    if candidate in frontier:
        frontier.remove(candidate)


def _causal_event_ids(
    index: Any,
    event_ids: Sequence[str],
    query_record: EventRecord,
    lookup_cache: _LookupCache,
    query_time_key: tuple[int, float | str],
) -> list[str]:
    query_event_id = query_record.event.event_id
    causal_ids: list[str] = []
    seen: set[str] = set()
    for event_id in event_ids:
        event_id_text = str(event_id)
        if event_id_text == query_event_id or event_id_text in seen:
            continue
        record = _get_event_record(index, event_id_text, lookup_cache)
        if record is None:
            continue
        if _time_sort_key(record.event.time) >= query_time_key:
            continue
        seen.add(event_id_text)
        causal_ids.append(event_id_text)
    return causal_ids


def _time_sort_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    parsed = _parse_time_value(value)
    if parsed is not None:
        return (0, parsed)
    return (1, str(value))


def _parse_time_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    for candidate in (text, text.replace("Z", "+00:00"), text.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate).timestamp()
        except ValueError:
            continue
    return None
