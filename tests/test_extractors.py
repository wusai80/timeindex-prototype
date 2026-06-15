import math

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
