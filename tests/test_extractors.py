import math
from types import MappingProxyType

import numpy as np

from timeindex.config import ExtractorConfig
from timeindex.event import Event
from timeindex.extractors import (
    EventRepresentationExtractor,
    compute_vector,
    extract_aspects,
    extract_keys,
    featurize_event,
)


def test_extract_keys_covers_entity_attr_context_type_and_time_block() -> None:
    event = Event(
        event_id="e1",
        time=245,
        event_type="transfer",
        attrs={
            "src_account": "acct_a",
            "dst_user": "user_b",
            "amount": 1250,
            "status": "approved",
            "region": "us-east",
        },
        ctx={"channel": "mobile"},
    )

    keys = extract_keys(event, ExtractorConfig(time_bucket_width=100))

    assert "type:transfer" in keys
    assert "time_block:2" in keys
    assert "entity:src_account=acct_a" in keys
    assert "entity:dst_user=user_b" in keys
    assert "participant:acct_a" in keys
    assert "flow_src:acct_a" in keys
    assert "flow_dst:user_b" in keys
    assert "flow_pair:acct_a->user_b" in keys
    assert "attr:status=approved" in keys
    assert "attr_bin:amount=10^3" in keys
    assert "ctx:region=us-east" in keys
    assert "ctx:channel=mobile" in keys


def test_compute_vector_has_expected_shape_and_unit_norm() -> None:
    event = Event(
        event_id="e2",
        time=10,
        event_type="login",
        attrs={"user_id": "u1"},
        text="Login from a trusted device",
    )
    keys = {"type:login", "entity:user_id=u1"}

    vector = compute_vector(event, keys, dim=32)

    assert vector.shape == (32,)
    assert np.isfinite(vector).all()
    assert math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=1e-9)


def test_extract_aspects_for_transaction_like_event() -> None:
    event = Event(
        event_id="e3",
        time=300,
        event_type="transfer",
        attrs={
            "amount": 9500,
            "prior_balance": 10000,
            "balance": 9800,
            "is_new_beneficiary": True,
            "device_changed": True,
            "burst_count": 4,
        },
        text="Customer used a new device for a new beneficiary transfer",
    )

    aspects = extract_aspects(event)

    assert "large_transfer" in aspects
    assert "source_accumulation" in aspects
    assert "beneficiary_novelty" in aspects
    assert "device_shift" in aspects
    assert "temporal_burst" in aspects


def test_extract_aspects_falls_back_to_generic_evidence() -> None:
    event = Event(
        event_id="e4",
        time=5,
        event_type="heartbeat",
        attrs={"status": "ok"},
        text="steady state",
    )

    aspects = extract_aspects(event)

    assert aspects == {"generic_evidence"}


def test_extract_aspects_for_lanl_auth_event() -> None:
    event = Event(
        event_id="lanl-1",
        time=151036,
        event_type="authentication",
        attrs={
            "src_user": "u748",
            "dst_user": "u748",
            "src_computer": "c17693",
            "dst_computer": "c305",
            "auth_type": "NTLM",
            "logon_type": "Network",
            "auth_orientation": "LogOn",
            "success": True,
            "is_cross_host_auth": True,
            "is_new_dst_for_user": True,
            "prior_user_event_count": 6,
            "prior_user_host_count": 2,
            "prior_pair_seen": False,
            "is_machine_account": False,
            "is_anonymous_logon": False,
        },
        text="user u748 authenticated as u748 from c17693 to c305 via NTLM success=true",
    )

    aspects = extract_aspects(event)

    assert "credential_reuse" in aspects
    assert "new_host_access" in aspects
    assert "lateral_movement" in aspects
    assert "rare_auth_path" in aspects
    assert "privilege_spread" in aspects


def test_featurize_event_builds_complete_event_record() -> None:
    config = ExtractorConfig(sketch_dim=16, time_bucket_width=50)
    event = Event(
        event_id="e5",
        time=125,
        event_type="deployment",
        attrs={"service": "payments-api", "metric_delta": 0.4},
        ctx={"env": "prod"},
        text="Deployment caused a metric shift",
    )

    record = featurize_event(event, config)
    extractor = EventRepresentationExtractor(config)
    extracted = extractor.extract(event)

    assert record.event is event
    assert record.sketch is not None
    assert record.sketch.shape == (16,)
    assert "type:deployment" in record.lookup_keys
    assert "entity:service=payments-api" in record.lookup_keys
    assert "ctx:env=prod" in record.lookup_keys
    assert "deployment_change" in record.aspects
    assert "metric_shift" in record.aspects
    assert extracted.lookup_keys == record.lookup_keys
    assert np.allclose(extracted.sketch, record.sketch)
    assert extracted.aspects == record.aspects


def test_featurize_event_precomputes_cached_subsets_and_flow_entities() -> None:
    config = ExtractorConfig(sketch_dim=16, time_bucket_width=50)
    event = Event(
        event_id="e6",
        time=125,
        event_type="transfer",
        attrs={
            "src_account": "acct_a",
            "beneficiary_id": "acct_b",
            "amount": 1200,
            "payment_format": "wire",
        },
        ctx={"region": "us"},
        text="wire transfer",
    )

    record = featurize_event(event, config)

    assert "entity:src_account=acct_a" in record.entity_keys
    assert "entity:beneficiary_id=acct_b" in record.entity_keys
    assert "attr_bin:amount=10^3" in record.attribute_keys
    assert "attr:payment_format=wire" in record.attribute_keys
    assert "ctx:region=us" in record.context_keys
    assert "type:transfer" in record.context_keys
    assert "time_block:2" in record.context_keys
    assert record.source_entities == frozenset({"acct_a"})
    assert record.destination_entities == frozenset({"acct_b"})
    assert record.participant_entities == frozenset({"acct_a", "acct_b"})
    assert record.sketch_is_normalized is True
    assert isinstance(record.event.attrs, MappingProxyType)
    assert isinstance(record.event.ctx, MappingProxyType)
    assert isinstance(record.lookup_keys, frozenset)
    assert isinstance(record.aspects, frozenset)
    assert record.sketch.flags.writeable is False


def test_extract_keys_does_not_treat_bank_columns_as_entity_endpoints() -> None:
    event = Event(
        event_id="e7",
        time=10,
        event_type="transfer",
        attrs={
            "src_account": "acct_a",
            "dst_account": "acct_b",
            "src_bank": "bank_1",
            "dst_bank": "bank_1",
        },
    )

    keys = extract_keys(event, ExtractorConfig())

    assert "entity:src_account=acct_a" in keys
    assert "entity:dst_account=acct_b" in keys
    assert "entity:src_bank=bank_1" not in keys
    assert "entity:dst_bank=bank_1" not in keys
    assert "ctx:src_bank=bank_1" not in keys
    assert "ctx:dst_bank=bank_1" not in keys
    assert "attr:src_bank=bank_1" in keys
    assert "attr:dst_bank=bank_1" in keys


def test_extract_keys_treats_computers_as_flow_endpoints() -> None:
    event = Event(
        event_id="lanl-2",
        time=10,
        event_type="authentication",
        attrs={
            "src_user": "alice",
            "src_computer": "c1",
            "dst_computer": "c2",
        },
    )

    keys = extract_keys(event, ExtractorConfig())

    assert "entity:src_computer=c1" in keys
    assert "entity:dst_computer=c2" in keys
    assert "participant:c1" in keys
    assert "participant:c2" in keys
    assert "flow_pair:c1->c2" in keys


def test_featurize_event_populates_computer_flow_entities() -> None:
    event = Event(
        event_id="lanl-3",
        time=10,
        event_type="authentication",
        attrs={
            "src_user": "alice",
            "dst_user": "bob",
            "src_computer": "c1",
            "dst_computer": "c2",
        },
    )

    record = featurize_event(event, ExtractorConfig())

    assert record.source_entities == frozenset({"alice", "c1"})
    assert record.destination_entities == frozenset({"bob", "c2"})
    assert record.participant_entities == frozenset({"alice", "c1", "bob", "c2"})
