from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ibm_aml.deepseek_agent import DeepSeekAgentDecision
from benchmarks.lanl.run_deepseek_sample import run_lanl_deepseek_sample
from benchmarks.lanl.verifier import verify_lanl_decision
from timeindex.event import Event, EventMetadata, EventRecord


def test_verifier_vetoes_skip_dominated_fanout_story() -> None:
    query = _lanl_query_event("auth-q1")
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "event",
            "ordinary:e1->q",
            ["credential_reuse", "lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true",
        ),
        _payload(
            "event",
            "ordinary:e2->e1",
            ["credential_reuse", "lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true",
        ),
        _payload(
            "skip",
            "skip:s1->e2",
            ["lanl_source_host_fanout", "lanl_query_host_touch"],
            "LANL skip pattern=lanl_source_host_fanout; query_dst_precursor=false; query_host_touch=true",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is False
    assert verified.verifier_overrode is True
    assert "skip-fanout" in verified.verifier_reason


def test_verifier_keeps_query_destination_precursor_story() -> None:
    query = _lanl_query_event("auth-q2")
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "event",
            "ordinary:e1->q",
            ["lanl_query_dst_precursor", "lanl_query_host_touch"],
            "LANL event pattern=lanl_bridge_into_query_host; query_dst_precursor=true; query_host_touch=true",
        ),
        _payload(
            "event",
            "ordinary:e2->e1",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is True
    assert verified.verifier_overrode is False


def test_verifier_keeps_chain_backed_story() -> None:
    query = _lanl_query_event("auth-q3")
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "chain",
            "chain:e0->e2:0",
            ["lanl_multi_step_context", "lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false",
        ),
        _payload(
            "event",
            "ordinary:e2->q",
            ["lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is True
    assert verified.verifier_overrode is False


def test_verifier_keeps_temporal_bridge_with_detached_continuity() -> None:
    query = _lanl_query_event("auth-q-bridge")
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "event",
            "ordinary:e1->q",
            ["lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true; span_seconds=0",
        ),
        _payload(
            "skip",
            "skip:s1->e2",
            ["lanl_source_host_fanout", "lanl_temporal_bridge", "lanl_multi_step_context"],
            "LANL skip pattern=lanl_source_host_fanout; query_dst_precursor=false; query_host_touch=true; span_seconds=219",
        ),
        _payload(
            "event",
            "ordinary:e2->e1",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
        _payload(
            "event",
            "ordinary:e3->e2",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is True
    assert verified.verifier_overrode is False
    assert verified.verifier_features["has_temporal_bridge"] is True
    assert verified.verifier_features["has_multi_step_context"] is True


def test_verifier_vetoes_detached_skip_without_direct_bridge() -> None:
    query = _lanl_query_event("auth-q-detached")
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "event",
            "ordinary:e1->q",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
        _payload(
            "skip",
            "skip:s1->e2",
            ["lanl_source_host_fanout"],
            "LANL skip pattern=lanl_source_host_fanout; query_dst_precursor=false; query_host_touch=true; span_seconds=20",
        ),
        _payload(
            "event",
            "ordinary:e2->e1",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
        _payload(
            "event",
            "ordinary:e3->e2",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is False
    assert verified.verifier_overrode is True
    assert "direct bridged path" in verified.verifier_reason


def test_verifier_vetoes_high_baseline_routine_touch_corridor() -> None:
    query = _lanl_query_event(
        "auth-q-baseline",
        prior_user_event_count=301,
        prior_user_host_count=17,
    )
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "skip",
            "skip:s0->q",
            ["lanl_query_host_touch"],
            "LANL skip pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true; span_seconds=0",
        ),
        _payload(
            "event",
            "ordinary:e1->q",
            ["lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true; span_seconds=0",
        ),
        _payload(
            "skip",
            "skip:s1->e1",
            ["lanl_credential_reuse_across_hosts", "lanl_temporal_bridge", "lanl_multi_step_context", "lanl_query_host_touch"],
            "LANL skip pattern=lanl_credential_reuse_across_hosts; query_dst_precursor=false; query_host_touch=true; span_seconds=152",
        ),
        _payload(
            "event",
            "ordinary:e2->e1",
            ["lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true; span_seconds=0",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is False
    assert verified.verifier_overrode is True
    assert "high-baseline touch corridor" in verified.verifier_reason


def test_verifier_keeps_high_baseline_chain_backed_positive() -> None:
    query = _lanl_query_event(
        "auth-q-high-chain",
        prior_user_event_count=500,
        prior_user_host_count=60,
    )
    decision = _positive_decision(query.event_id)
    objects = [
        _payload(
            "chain",
            "chain:e0->e2:0",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
        _payload(
            "event",
            "ordinary:e2->q",
            ["lanl_query_host_touch"],
            "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true; span_seconds=0",
        ),
        _payload(
            "event",
            "ordinary:e1->e2",
            ["lanl_user_continuity_detached"],
            "LANL event pattern=lanl_user_continuity_detached; query_dst_precursor=false; query_host_touch=false; span_seconds=0",
        ),
    ]

    verified = verify_lanl_decision(query, decision, objects, mode="lanl_rule")

    assert verified.predicted_positive is True
    assert verified.verifier_overrode is False


def test_run_lanl_deepseek_sample_records_verifier_fields(tmp_path: Path, monkeypatch) -> None:
    index_path = tmp_path / "index.sqlite"
    sample_path = tmp_path / "sample.json"
    output_dir = tmp_path / "outputs"
    sample_path.write_text(json.dumps({"positives": [{"event_id": "auth-q4"}], "negatives": []}), encoding="utf-8")

    query_record = _lanl_query_record("auth-q4", label="1")

    class _FakeIndex:
        edge_store = type("EdgeStore", (), {"incoming": staticmethod(lambda _event_id: ["e1", "e2"])})()
        skip_link_store = type("SkipStore", (), {"incoming": staticmethod(lambda _event_id: ["s1"])})()
        chain_store = type("ChainStore", (), {"get_for_tail": staticmethod(lambda _event_id: [])})()

        def get_event(self, event_id: str) -> EventRecord | None:
            if event_id == "auth-q4":
                return query_record
            return None

        def cache_stats(self) -> dict[str, int]:
            return {"hits": 0, "misses": 0}

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "benchmarks.lanl.run_deepseek_sample.SqliteTimeIndexBackend.open",
        lambda _path: _FakeIndex(),
    )
    monkeypatch.setattr(
        "benchmarks.lanl.run_deepseek_sample.classify_query_with_deepseek",
        lambda *args, **kwargs: _positive_decision("auth-q4"),
    )
    monkeypatch.setattr(
        "benchmarks.lanl.run_deepseek_sample._collect_retrieval",
        lambda *args, **kwargs: {
            "event_ids": ["e1", "e2"],
            "aspects": ["lanl_query_host_touch", "lanl_source_host_fanout"],
            "events": [],
            "objects": [],
            "object_types": ["event", "skip"],
            "object_ids": ["ordinary:e1->q", "skip:s1->e1"],
            "object_payloads": [
                _payload(
                    "event",
                    "ordinary:e1->q",
                    ["lanl_query_host_touch"],
                    "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true",
                ),
                _payload(
                    "event",
                    "ordinary:e2->e1",
                    ["lanl_query_host_touch"],
                    "LANL event pattern=lanl_query_host_touch; query_dst_precursor=false; query_host_touch=true",
                ),
                _payload(
                    "skip",
                    "skip:s1->e2",
                    ["lanl_source_host_fanout", "lanl_query_host_touch"],
                    "LANL skip pattern=lanl_source_host_fanout; query_dst_precursor=false; query_host_touch=true",
                ),
            ],
            "frontier_stats": {"ordinary_incoming_links": 2, "skip_incoming_links": 1, "chain_summaries": 0},
        },
    )

    summary = run_lanl_deepseek_sample(
        index_path,
        sample_path,
        output_dir=output_dir,
        verifier_mode="lanl_rule",
        max_retries=1,
    )

    assert summary["summary"]["false_negative"] == 1.0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert row["llm_predicted_positive"] is True
    assert row["predicted_positive"] is False
    assert row["verifier_applied"] is True
    assert row["verifier_overrode"] is True
    assert row["verifier_mode"] == "lanl_rule"


def _lanl_query_event(
    event_id: str,
    *,
    prior_user_event_count: int = 5,
    prior_user_host_count: int = 3,
) -> Event:
    return Event(
        event_id=event_id,
        time=10,
        event_type="auth",
        attrs={
            "src_user": "alice@LANL",
            "src_host": "c1",
            "dst_host": "c2",
            "prior_user_event_count": prior_user_event_count,
            "prior_user_host_count": prior_user_host_count,
        },
        ctx={"dataset": "lanl_auth"},
        label="1",
    )


def _lanl_query_record(event_id: str, *, label: str) -> EventRecord:
    return EventRecord(
        event=_lanl_query_event(event_id),
        lookup_keys={"entity:alice@lanl", "type:auth"},
        aspects={"credential_reuse"},
        metadata=EventMetadata(),
    )


def _positive_decision(event_id: str) -> DeepSeekAgentDecision:
    return DeepSeekAgentDecision(
        query_event_id=event_id,
        query_label="1",
        model="deepseek-chat",
        predicted_positive=True,
        confidence=0.81,
        rationale="The history suggests suspicious lateral movement.",
        risk_factors=["credential reuse"],
        supporting_event_ids=["e1"],
        retrieved_event_ids=["e1", "e2"],
        retrieved_aspects=["lanl_query_host_touch"],
        prompt_event_count=2,
        raw_response={},
    )


def _payload(object_type: str, object_id: str, aspects: list[str], summary: str) -> dict[str, object]:
    return {
        "object_id": object_id,
        "type": object_type,
        "event_ids": ["e1"],
        "aspects": aspects,
        "cost": 1.0,
        "summary": summary,
    }
