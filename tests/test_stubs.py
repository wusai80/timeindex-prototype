import pytest

from timeindex.candidate_index import SkipCandidateIndex
from timeindex.config import (
    ConstructionConfig,
    ExtractorConfig,
    RetrievalConfig,
    ScoringConfig,
    SyntheticConfig,
)
from timeindex.construction import IndexConstructor
from timeindex.event import ChainSummary, DecisionIntent, Event, EventQuery, EventRecord
from timeindex.extractors import EventRepresentationExtractor
from timeindex.retrieval import DualFrontierRetriever
from timeindex.scoring import PrototypeScorer
from timeindex.stores import ChainStore, EdgeStore, EventStore, KeyDirectory, SkipLinkStore
from timeindex.synthetic import SyntheticStreamGenerator


def test_extractor_smoke() -> None:
    extractor = EventRepresentationExtractor(ExtractorConfig())
    record = extractor.extract(Event(event_id="e1", time=1, event_type="transfer"))

    assert record.event.event_id == "e1"
    assert "type:transfer" in record.lookup_keys


def test_scorer_smoke() -> None:
    scorer = PrototypeScorer(ScoringConfig())
    record = EventRecord(event=Event(event_id="e1", time=1, event_type="transfer"))
    score = scorer.score_local_dependency(record, record)

    assert 0.0 <= score <= 1.0


def test_skip_candidate_index_smoke() -> None:
    index = SkipCandidateIndex()
    record = EventRecord(event=Event(event_id="e1", time=1, event_type="transfer"))
    assert index.retrieve(record, DecisionIntent()) == []


def test_constructor_stub_raises() -> None:
    constructor = IndexConstructor(
        ConstructionConfig(),
        EventStore(),
        KeyDirectory(),
        EdgeStore(),
        ChainStore(),
        SkipCandidateIndex(),
        SkipLinkStore(),
    )
    with pytest.raises(NotImplementedError):
        constructor.add_event(Event(event_id="e1", time=1, event_type="transfer"))


def test_retriever_smoke() -> None:
    retriever = DualFrontierRetriever(
        EventStore(),
        EdgeStore(),
        ChainStore(),
        SkipLinkStore(),
        RetrievalConfig(),
    )
    result = retriever.retrieve(EventQuery(event=Event(event_id="e1", time=1, event_type="transfer")))

    assert result == []


def test_synthetic_stub_raises() -> None:
    generator = SyntheticStreamGenerator(SyntheticConfig())
    with pytest.raises(NotImplementedError):
        generator.generate_events()


def test_chain_anchor_smoke() -> None:
    index = SkipCandidateIndex()
    summary = ChainSummary(chain_id="c1", family="generic", head_id="e1", tail_id="e2")
    index.add_chain_anchor(summary)

    assert index.get_object("c1") is summary
