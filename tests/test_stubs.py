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


def test_extractor_stub_raises() -> None:
    extractor = EventRepresentationExtractor(ExtractorConfig())
    with pytest.raises(NotImplementedError):
        extractor.extract(Event(event_id="e1", time=1, event_type="transfer"))


def test_scorer_stub_raises() -> None:
    scorer = PrototypeScorer(ScoringConfig())
    record = EventRecord(event=Event(event_id="e1", time=1, event_type="transfer"))
    with pytest.raises(NotImplementedError):
        scorer.score_local_dependency(record, record)


def test_skip_candidate_index_stub_raises() -> None:
    index = SkipCandidateIndex()
    record = EventRecord(event=Event(event_id="e1", time=1, event_type="transfer"))
    with pytest.raises(NotImplementedError):
        index.retrieve(record, DecisionIntent())


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


def test_retriever_stub_raises() -> None:
    retriever = DualFrontierRetriever(
        EventStore(),
        EdgeStore(),
        ChainStore(),
        SkipLinkStore(),
        RetrievalConfig(),
    )
    with pytest.raises(NotImplementedError):
        retriever.retrieve(EventQuery(event=Event(event_id="e1", time=1, event_type="transfer")))


def test_synthetic_stub_raises() -> None:
    generator = SyntheticStreamGenerator(SyntheticConfig())
    with pytest.raises(NotImplementedError):
        generator.generate_events()


def test_chain_anchor_stub_raises() -> None:
    index = SkipCandidateIndex()
    summary = ChainSummary(chain_id="c1", family="generic", head_id="e1", tail_id="e2")
    with pytest.raises(NotImplementedError):
        index.add_chain_anchor(summary)
