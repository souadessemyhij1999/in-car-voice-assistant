"""
Deterministic safety layer between the assistant and the car.

Every tool call - whether it came from the offline model or from Claude - passes through this guard
before anything happens in the vehicle. An LLM can be persuaded or can make mistakes; these rules
cannot. That separation (AI suggests, deterministic code decides) is the core design principle.

Rules
  * Driver-distraction rules: no video and no opening the trunk unless the car is stationary.
  * Plausible ranges: temperature 16-28 °C, fan 0-5, seat heating 0-3, volume 0-10, window 0-100% (clamped).
  * Phone calls are prepared but need the driver's confirmation.
  * Anything outside the defined tools (e.g. "switch off the airbag") simply cannot be executed.
"""

from dataclasses import dataclass

from vehicle import TOOL_NAMES, VehicleState

LIMITS = {("set_temperature", "celsius"): (16, 28), ("set_fan", "level"): (0, 5), ("set_seat_heating", "level"): (0, 3),
          ("set_volume", "level"): (0, 10), ("set_window", "percent"): (0, 100)}
STATIONARY_ONLY = {"play_video": "Videos can only be played while the car is parked.",
                   "open_trunk": "The trunk can only be opened while the car is stationary."}


@dataclass
class Decision:
    allowed: bool
    args: dict
    note: str = ""            # spoken explanation when something was blocked or adjusted
    needs_confirmation: bool = False


class SafetyGuard:
    def check(self, name: str, args: dict, state: VehicleState) -> Decision:
        if name not in TOOL_NAMES:
            return Decision(False, args, "I can't control that function.")
        if name in STATIONARY_ONLY and state.speed_kmh > 0:
            return Decision(False, args, STATIONARY_ONLY[name])
        args = dict(args)
        notes = []
        # relative changes: turn them into an absolute target so the limits can be applied
        if name == "set_temperature" and args.get("celsius") is None and args.get("change") is not None:
            zone = args.get("zone") or "all"
            base = state.temperature["driver" if zone == "all" else zone]
            args["celsius"], args["change"] = base + args["change"], None
        if name in ("set_fan", "set_volume") and args.get("level") is None and args.get("change") is not None:
            current = state.fan if name == "set_fan" else state.volume
            args["level"], args["change"] = current + args["change"], None
        for (tool, key), (lo, hi) in LIMITS.items():
            if tool == name and args.get(key) is not None:
                v = args[key]
                if v < lo or v > hi:
                    args[key] = max(lo, min(hi, v))
                    notes.append(f"{key.replace('celsius', 'temperature')} limited to {args[key]}")
        args = {k: v for k, v in args.items() if v is not None}
        return Decision(True, args, "; ".join(notes), needs_confirmation=(name == "call_contact"))
