from __future__ import annotations

from benchmarks.ibm_aml.deepseek_agent import (
    build_chat_messages,
    classify_query_with_deepseek,
    parse_key_file_for_provider,
)
from timeindex.event import Event


def test_parse_key_file_for_provider_prefers_matching_line() -> None:
    text = """
    # local keys
    deepseek: sk-deepseektestkey123456
    gpt: sk-openaitestkey987654
    """

    assert parse_key_file_for_provider(text, "deepseek") == "sk-deepseektestkey123456"


def test_build_chat_messages_contains_query_and_retrieved_ids() -> None:
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

    messages = build_chat_messages(query, [support], {"large_transfer"})

    assert messages[0]["role"] == "system"
    assert "Query Event" in messages[1]["content"]
    assert "Structural Story Summary" in messages[1]["content"]
    assert "Prior Evidence Timeline" in messages[1]["content"]
    assert "q1" in messages[1]["content"]
    assert "e1" in messages[1]["content"]
    assert "large_transfer" in messages[1]["content"]


def test_build_chat_messages_omits_labels_and_uses_relation_tags() -> None:
    query = Event(
        event_id="q1",
        time=10,
        event_type="wire",
        attrs={"src_account": "A", "dst_account": "B", "amount": 900.0, "currency": "USD"},
        text="wire with label 1 embedded",
        label="1",
    )
    support = Event(
        event_id="e1",
        time=7,
        event_type="deposit",
        attrs={"src_account": "X", "dst_account": "A", "amount": 500.0, "currency": "USD"},
        text="deposit label 0",
        label="0",
    )

    messages = build_chat_messages(query, [support], {"source_accumulation"})
    content = messages[1]["content"]

    assert '"label"' not in content
    assert "label 1" not in content
    assert "label 0" not in content
    assert "inbound_to_query_src" in content
    assert "same_src_count=" in content
    assert "dominant_story=" in content
    assert "Evidence Interpretation Hints" in content
    assert "likely_routine_card_or_cheque=" in content
    assert "strong_non_ach_structure=" in content
    assert "temporal evidence analyst" in messages[0]["content"]


def test_classify_query_with_deepseek_normalizes_stub_response() -> None:
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
        attrs={"src_account": "A", "dst_account": "Y", "amount": 600.0},
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
                            '{"predicted_positive": true, "confidence": 0.82, '
                            '"risk_factors": ["account accumulation", "large transfer"], '
                            '"supporting_event_ids": ["e2", "missing", "e1"], '
                            '"rationale": "Prior inflows support the suspicious transfer."}'
                        )
                    }
                }
            ]
        }

    decision = classify_query_with_deepseek(
        query,
        [support_a, support_b],
        {"large_transfer", "source_accumulation"},
        api_key="test-key",
        transport=stub_transport,
    )

    assert decision.predicted_positive is True
    assert decision.confidence == 0.82
    assert decision.supporting_event_ids == ["e2", "e1"]
    assert "large transfer" in decision.risk_factors
