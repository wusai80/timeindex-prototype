from timeindex import Event, EventQuery, TimeIndexConfig


def test_package_imports() -> None:
    config = TimeIndexConfig()
    event = Event(event_id="e1", time=1, event_type="transfer")
    query = EventQuery(event=event, budget=5)

    assert config.stores.ordinary_fan_in == 5
    assert query.event.event_id == "e1"
