import dataclasses

from tennis.pipeline.graph import ORDER, SOURCES, SPECS, fingerprint


def test_order_is_a_valid_topological_order_over_all_stages():
    assert set(ORDER) == set(SPECS)
    seen = set(SOURCES)
    for name in ORDER:
        assert set(SPECS[name].deps) <= seen, f"{name} runs before one of its deps"
        seen.add(name)


def test_ball_tracking_does_not_depend_on_players_or_calibration():
    # POSE -> BALL was a false dependency in the old linear runner.
    assert "PLAYER_TRACKING" not in SPECS["BALL_TRACKING"].deps
    assert "CALIBRATION" not in SPECS["BALL_TRACKING"].deps


def test_every_stage_output_has_an_explicit_extension():
    for spec in SPECS.values():
        assert "." in spec.output


def test_fingerprint_is_stable_and_changes_with_inputs_version_and_params():
    spec = SPECS["PLAYER_TRACKING"]
    hashes = {dep: f"hash-{dep}" for dep in spec.deps}

    base = fingerprint(spec, hashes)

    assert fingerprint(spec, dict(hashes)) == base
    assert fingerprint(spec, {**hashes, "CALIBRATION": "other"}) != base
    assert fingerprint(dataclasses.replace(spec, version="999"), hashes) != base
    assert fingerprint(dataclasses.replace(spec, params={"stride": 3}), hashes) != base
