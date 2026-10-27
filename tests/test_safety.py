"""Safety layer rules: these must hold no matter which model produced the request."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from safety import SafetyGuard  # noqa: E402
from vehicle import VehicleState  # noqa: E402

G = SafetyGuard()


def test_no_video_or_trunk_while_moving():
    for name in ("play_video", "open_trunk"):
        assert not G.check(name, {}, VehicleState(speed_kmh=30)).allowed
        assert G.check(name, {}, VehicleState(speed_kmh=0)).allowed


def test_values_are_clamped_to_safe_ranges():
    s = VehicleState(speed_kmh=50)
    assert G.check("set_temperature", {"celsius": 35}, s).args["celsius"] == 28
    assert G.check("set_temperature", {"celsius": 5}, s).args["celsius"] == 16
    assert G.check("set_volume", {"level": 15}, s).args["level"] == 10
    assert G.check("set_seat_heating", {"level": 9, "seat": "driver"}, s).args["level"] == 3


def test_relative_changes_become_absolute_and_are_clamped():
    s = VehicleState(speed_kmh=50)
    s.temperature["driver"] = 27.0
    assert G.check("set_temperature", {"change": 5}, s).args == {"celsius": 28}
    assert G.check("set_volume", {"change": -20}, s).args == {"level": 0}


def test_calls_need_confirmation_and_unknown_functions_are_refused():
    assert G.check("call_contact", {"name": "Mom"}, VehicleState()).needs_confirmation
    assert not G.check("disable_esp", {}, VehicleState()).allowed
