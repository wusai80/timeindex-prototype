from __future__ import annotations

from benchmarks.ibm_aml.two_agent import (
    AnalystCaseReport,
    build_analysis_messages,
    build_judge_messages,
    classify_query_with_two_agents,
    analyze_case_with_llm,
    judge_case_with_llm,
)
from timeindex.event import Event


def test_build_analysis_messages_contains_case_sections() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )

    messages = build_analysis_messages(query, [support], {"source_accumulation"})

    assert messages[0]["role"] == "system"
    assert "Evidence Analyst" in messages[0]["content"]
    assert "Structural Story Summary" in messages[1]["content"]
    assert "supporting_event_ids" in messages[1]["content"]
    assert "do not infer or mention any event after the query" in messages[1]["content"]
    assert "e1" in messages[1]["content"]


def test_analyze_case_with_llm_normalizes_supporting_ids() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support_a = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )
    support_b = Event(
        event_id="e2",
        time=8,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Y", "amount": 700.0},
        label="1",
    )

    def stub_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"timeline_summary":"incoming funds then outward movement",'
                            '"suspicious_patterns":["accumulation before outflow"],'
                            '"routine_explanations":["could still be a routine transfer"],'
                            '"key_entities":["A","B"],'
                            '"supporting_event_ids":["e2","missing","e1"],'
                            '"confidence_signals":["multiple precursors"],'
                            '"missing_evidence":["no beneficiary history"],'
                            '"analyst_confidence":0.73}'
                        )
                    }
                }
            ]
        }

    report = analyze_case_with_llm(
        query,
        [support_a, support_b],
        {"large_transfer"},
        api_key="test-key",
        transport=stub_transport,
    )

    assert report.supporting_event_ids == ["e2", "e1"]
    assert report.suspicious_patterns == ["accumulation before outflow"]
    assert report.analyst_confidence == 0.73


def test_build_judge_messages_contains_analyst_case_file() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )
    report = AnalystCaseReport(
        query_event_id="q1",
        query_label="1",
        model="stub-model",
        timeline_summary="incoming funds then outward movement",
        suspicious_patterns=["accumulation before outflow"],
        routine_explanations=["routine payments possible"],
        key_entities=["A", "B"],
        supporting_event_ids=["e1"],
        confidence_signals=["multiple precursor accounts"],
        missing_evidence=["no downstream bridge"],
        analyst_confidence=0.7,
        retrieved_event_ids=["e1"],
        retrieved_aspects=["large_transfer"],
        raw_response={},
    )

    messages = build_judge_messages(query, report, [support])

    assert "Analyst Case File" in messages[1]["content"]
    assert "accepted_mechanisms" in messages[1]["content"]
    assert "accumulation_before_outflow" in messages[1]["content"]
    assert "Never use or mention any event after the query" in messages[1]["content"]
    assert "q1" in messages[1]["content"]
    assert "e1" in messages[1]["content"]


def test_judge_case_with_llm_forces_false_without_mechanism() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )
    report = AnalystCaseReport(
        query_event_id="q1",
        query_label="1",
        model="stub-model",
        timeline_summary="weak case",
        suspicious_patterns=["possible build-up"],
        routine_explanations=["might be routine"],
        key_entities=["A"],
        supporting_event_ids=["e1"],
        confidence_signals=[],
        missing_evidence=["no strong path"],
        analyst_confidence=0.3,
        retrieved_event_ids=["e1"],
        retrieved_aspects=[],
        raw_response={},
    )

    def stub_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"predicted_positive": true, "confidence": 0.88, '
                            '"accepted_supporting_event_ids": ["e1"], '
                            '"accepted_mechanisms": [], '
                            '"rejected_claims": ["weak chain"], '
                            '"rationale": "No durable mechanism survives review."}'
                        )
                    }
                }
            ]
        }

    decision = judge_case_with_llm(
        query,
        report,
        [support],
        api_key="test-key",
        transport=stub_transport,
    )

    assert decision.predicted_positive is False
    assert decision.confidence == 0.49
    assert "no_concrete_temporal_mechanism" in decision.rejected_claims


def test_judge_case_with_llm_normalizes_mechanism_aliases_and_drops_future_claims() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support_a = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )
    support_b = Event(
        event_id="e2",
        time=8,
        event_type="deposit",
        attrs={"src_account": "Z", "dst_account": "A", "amount": 450.0},
        label="1",
    )
    support_c = Event(
        event_id="e3",
        time=9,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Y", "amount": 700.0},
        label="1",
    )
    report = AnalystCaseReport(
        query_event_id="q1",
        query_label="1",
        model="stub-model",
        timeline_summary="funds collected and moved onward",
        suspicious_patterns=["accumulation before outflow"],
        routine_explanations=["limited history"],
        key_entities=["A", "B"],
        supporting_event_ids=["e1", "e2", "e3"],
        confidence_signals=["two linked precursors"],
        missing_evidence=["no beneficiary history"],
        analyst_confidence=0.81,
        retrieved_event_ids=["e1", "e2", "e3"],
        retrieved_aspects=["large_transfer"],
        raw_response={},
    )

    def stub_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"predicted_positive": true, "confidence": 0.77, '
                            '"accepted_supporting_event_ids": ["e3","e2","e1"], '
                            '"accepted_mechanisms": ["accumulation before outflow"], '
                            '"rejected_claims": ["subsequent outbound from destination", "weak path"], '
                            '"rationale": "The source accumulates funds and then forwards them."}'
                        )
                    }
                }
            ]
        }

    decision = judge_case_with_llm(
        query,
        report,
        [support_a, support_b, support_c],
        api_key="test-key",
        transport=stub_transport,
    )

    assert decision.predicted_positive is True
    assert decision.accepted_mechanisms == ["accumulation_before_outflow"]
    assert decision.rejected_claims == ["weak path"]


def test_judge_case_with_llm_drops_weak_accumulation_and_repeated_pair() -> None:
    query = Event(
        event_id="q1",
        time=20,
        event_type="cheque",
        attrs={"src_account": "A", "dst_account": "B", "amount": 400.0},
        label="0",
    )
    prior_inbound = Event(
        event_id="e1",
        time=10,
        event_type="card",
        attrs={"src_account": "X", "dst_account": "A", "amount": 150.0},
        label="0",
    )
    prior_pair_a = Event(
        event_id="e2",
        time=12,
        event_type="cash",
        attrs={"src_account": "A", "dst_account": "B", "amount": 50.0},
        label="0",
    )
    prior_pair_b = Event(
        event_id="e3",
        time=13,
        event_type="ach",
        attrs={"src_account": "A", "dst_account": "B", "amount": 75.0},
        label="0",
    )
    prior_pair_c = Event(
        event_id="e4",
        time=14,
        event_type="cash",
        attrs={"src_account": "A", "dst_account": "B", "amount": 90.0},
        label="0",
    )
    report = AnalystCaseReport(
        query_event_id="q1",
        query_label="0",
        model="stub-model",
        timeline_summary="single-source accumulation and repeated transfers",
        suspicious_patterns=["accumulation before outflow", "repeated pair escalation"],
        routine_explanations=["vendor payments possible"],
        key_entities=["A", "B"],
        supporting_event_ids=["e1", "e2", "e3", "e4"],
        confidence_signals=["rapid same-pair sequence"],
        missing_evidence=["no destination outflow"],
        analyst_confidence=0.8,
        retrieved_event_ids=["e1", "e2", "e3", "e4"],
        retrieved_aspects=[],
        raw_response={},
    )

    def stub_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"predicted_positive": true, "confidence": 0.83, '
                            '"accepted_supporting_event_ids": ["e1", "e2", "e3", "e4"], '
                            '"accepted_mechanisms": ["accumulation_before_outflow", "repeated_pair_escalation"], '
                            '"rejected_claims": [], '
                            '"rationale": "Single-source buildup and repeated transfers appear suspicious."}'
                        )
                    }
                }
            ]
        }

    decision = judge_case_with_llm(
        query,
        report,
        [prior_inbound, prior_pair_a, prior_pair_b, prior_pair_c],
        api_key="test-key",
        transport=stub_transport,
    )

    assert decision.predicted_positive is False
    assert decision.accepted_mechanisms == []
    assert decision.accepted_supporting_event_ids == []
    assert "mechanism_not_validated:accumulation_before_outflow" in decision.rejected_claims
    assert "mechanism_not_validated:repeated_pair_escalation" in decision.rejected_claims


def test_judge_case_with_llm_keeps_repeated_pair_for_strong_two_event_escalation() -> None:
    query = Event(
        event_id="q1",
        time=20,
        event_type="ach",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    prior_pair = Event(
        event_id="e1",
        time=18,
        event_type="ach",
        attrs={"src_account": "A", "dst_account": "B", "amount": 200.0},
        label="1",
    )
    report = AnalystCaseReport(
        query_event_id="q1",
        query_label="1",
        model="stub-model",
        timeline_summary="same pair with a larger follow-up transfer",
        suspicious_patterns=["repeated pair escalation"],
        routine_explanations=["could be regular payments"],
        key_entities=["A", "B"],
        supporting_event_ids=["e1"],
        confidence_signals=["query is much larger than prior transfer"],
        missing_evidence=["limited history"],
        analyst_confidence=0.7,
        retrieved_event_ids=["e1"],
        retrieved_aspects=[],
        raw_response={},
    )

    def stub_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"predicted_positive": true, "confidence": 0.74, '
                            '"accepted_supporting_event_ids": ["e1"], '
                            '"accepted_mechanisms": ["repeated_pair_escalation"], '
                            '"rejected_claims": [], '
                            '"rationale": "The same pair repeats and the query is much larger."}'
                        )
                    }
                }
            ]
        }

    decision = judge_case_with_llm(
        query,
        report,
        [prior_pair],
        api_key="test-key",
        transport=stub_transport,
    )

    assert decision.predicted_positive is True
    assert decision.accepted_mechanisms == ["repeated_pair_escalation"]


def test_judge_case_with_llm_accepts_destination_concentration_as_bridge_relay() -> None:
    query = Event(
        event_id="q1",
        time=20,
        event_type="ach",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    prior_inbound_a = Event(
        event_id="e1",
        time=15,
        event_type="ach",
        attrs={"src_account": "X", "dst_account": "B", "amount": 250.0},
        label="1",
    )
    prior_inbound_b = Event(
        event_id="e2",
        time=17,
        event_type="wire",
        attrs={"src_account": "Y", "dst_account": "B", "amount": 300.0},
        label="1",
    )
    report = AnalystCaseReport(
        query_event_id="q1",
        query_label="1",
        model="stub-model",
        timeline_summary="multiple sources feed the destination and the query continues the buildup",
        suspicious_patterns=["destination concentration"],
        routine_explanations=["could be a collection account"],
        key_entities=["A", "B", "X", "Y"],
        supporting_event_ids=["e1", "e2"],
        confidence_signals=["many-to-one concentration before the query"],
        missing_evidence=["no visible outbound from destination in horizon"],
        analyst_confidence=0.72,
        retrieved_event_ids=["e1", "e2"],
        retrieved_aspects=[],
        raw_response={},
    )

    def stub_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"predicted_positive": true, "confidence": 0.69, '
                            '"accepted_supporting_event_ids": ["e2", "e1"], '
                            '"accepted_mechanisms": ["destination concentration"], '
                            '"rejected_claims": [], '
                            '"rationale": "The destination receives concentrated inbound flow and the query continues that pattern."}'
                        )
                    }
                }
            ]
        }

    decision = judge_case_with_llm(
        query,
        report,
        [prior_inbound_a, prior_inbound_b],
        api_key="test-key",
        transport=stub_transport,
    )

    assert decision.predicted_positive is True
    assert decision.accepted_mechanisms == ["bridge_relay"]


def test_classify_query_with_two_agents_runs_both_stages() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0},
        label="1",
    )
    support_a = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0},
        label="1",
    )
    support_b = Event(
        event_id="e2",
        time=8,
        event_type="deposit",
        attrs={"src_account": "Z", "dst_account": "A", "amount": 450.0},
        label="1",
    )
    support_c = Event(
        event_id="e3",
        time=9,
        event_type="transfer",
        attrs={"src_account": "A", "dst_account": "Y", "amount": 700.0},
        label="1",
    )

    def analyzer_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"timeline_summary":"funds collected and moved onward",'
                            '"suspicious_patterns":["accumulation before outflow"],'
                            '"routine_explanations":["limited history"],'
                            '"key_entities":["A","B","X","Z"],'
                            '"supporting_event_ids":["e1","e2","e3"],'
                            '"confidence_signals":["two linked precursors"],'
                            '"missing_evidence":["no beneficiary history"],'
                            '"analyst_confidence":0.81}'
                        )
                    }
                }
            ]
        }

    def judge_transport(
        _base_url: str,
        _api_key: str,
        _messages: list[dict[str, str]],
        _model: str,
        _timeout_s: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"predicted_positive": true, "confidence": 0.76, '
                            '"accepted_supporting_event_ids": ["e3","e2","e1"], '
                            '"accepted_mechanisms": ["accumulation before outflow"], '
                            '"rejected_claims": ["bridge story unsupported"], '
                            '"rationale": "The source receives funds and then sends a larger outward transfer."}'
                        )
                    }
                }
            ]
        }

    decision = classify_query_with_two_agents(
        query,
        [support_a, support_b, support_c],
        {"large_transfer", "source_accumulation"},
        analyzer_api_key="test-key",
        judge_api_key="test-key",
        analyzer_transport=analyzer_transport,
        judge_transport=judge_transport,
    )

    assert decision.predicted_positive is True
    assert decision.accepted_supporting_event_ids == ["e3", "e2", "e1"]
    assert decision.accepted_mechanisms == ["accumulation_before_outflow"]
    assert decision.analyst_report.supporting_event_ids == ["e1", "e2", "e3"]
