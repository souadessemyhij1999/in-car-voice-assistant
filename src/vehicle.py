"""
A simulated vehicle and the functions the assistant can call.

TOOLS is the single source of truth: the same definitions are used by the offline in-car model and
sent to Claude as tool definitions, so both "brains" control the car through exactly the same interface.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field

SEATS = ["driver", "passenger", "rear"]
WINDOWS = ["driver", "passenger", "rear_left", "rear_right", "all"]
NEARBY = ["charging_station", "parking", "fuel_station", "restaurant", "coffee", "hospital", "supermarket"]
STATUS_ITEMS = ["range", "battery", "tyre_pressure", "service", "doors", "oil", "fuel"]
DRIVE_MODES = ["eco", "comfort", "sport"]

TOOLS = [
    {"name": "set_temperature",
     "description": "Set the cabin temperature. Use 'celsius' for an absolute target (16-28) or 'change' for a "
                    "relative change in degrees (e.g. +2 when the user is cold, -2 when warm). Zone defaults to all.",
     "input_schema": {"type": "object", "properties": {
         "zone": {"type": "string", "enum": ["driver", "passenger", "all"]},
         "celsius": {"type": "number"}, "change": {"type": "number"}}}},
    {"name": "set_fan", "description": "Set the climate fan speed from 0 (off) to 5 (max), or change it relatively.",
     "input_schema": {"type": "object", "properties": {"level": {"type": "integer"}, "change": {"type": "integer"}}}},
    {"name": "set_seat_heating", "description": "Seat heating level 0 (off) to 3 (max) for a seat.",
     "input_schema": {"type": "object", "properties": {
         "seat": {"type": "string", "enum": SEATS}, "level": {"type": "integer"}}, "required": ["level"]}},
    {"name": "set_window", "description": "Open or close a window. 'percent' optionally sets how far open (0-100).",
     "input_schema": {"type": "object", "properties": {
         "window": {"type": "string", "enum": WINDOWS}, "action": {"type": "string", "enum": ["open", "close"]},
         "percent": {"type": "integer"}}, "required": ["action"]}},
    {"name": "navigate_to", "description": "Start route guidance to a destination (address, city, place or saved place "
                                           "such as 'home' or 'work').",
     "input_schema": {"type": "object", "properties": {"destination": {"type": "string"}}, "required": ["destination"]}},
    {"name": "find_nearby", "description": "Search for places near the car's current position.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string", "enum": NEARBY}},
                      "required": ["category"]}},
    {"name": "play_media", "description": "Play music, a radio station, a podcast or an artist.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "source": {"type": "string", "enum": ["radio", "music", "podcast"]}}}},
    {"name": "media_control", "description": "Pause, resume, skip or go back in the current media.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["pause", "resume", "next", "previous"]}}, "required": ["action"]}},
    {"name": "set_volume", "description": "Set the audio volume 0-10 ('level'), or change it relatively ('change').",
     "input_schema": {"type": "object", "properties": {"level": {"type": "integer"}, "change": {"type": "integer"}}}},
    {"name": "call_contact", "description": "Phone a contact by name. The driver must confirm before the call starts.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_vehicle_status", "description": "Read vehicle information such as remaining range, battery level, "
                                                  "tyre pressure, next service, door status.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string", "enum": STATUS_ITEMS}},
                      "required": ["item"]}},
    {"name": "set_drive_mode", "description": "Switch the driving mode.",
     "input_schema": {"type": "object", "properties": {"mode": {"type": "string", "enum": DRIVE_MODES}}, "required": ["mode"]}},
    {"name": "set_ambient_light", "description": "Set the colour of the interior ambient lighting.",
     "input_schema": {"type": "object", "properties": {"colour": {"type": "string"}}, "required": ["colour"]}},
    {"name": "open_trunk", "description": "Open the boot / trunk lid.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "play_video", "description": "Play a video or film on the central display.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "manual_lookup", "description": "Answer a question about how the car works, warning lights or "
                                             "maintenance, using the owner's handbook.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
]
TOOL_NAMES = [t["name"] for t in TOOLS]


@dataclass
class VehicleState:
    speed_kmh: float = 0.0
    temperature: dict = field(default_factory=lambda: {"driver": 21.0, "passenger": 21.0})
    fan: int = 2
    seat_heating: dict = field(default_factory=lambda: {"driver": 0, "passenger": 0, "rear": 0})
    windows: dict = field(default_factory=lambda: {w: 0 for w in WINDOWS if w != "all"})
    destination: str = ""
    media: str = ""
    media_playing: bool = False
    volume: int = 5
    drive_mode: str = "comfort"
    ambient_light: str = "white"
    trunk_open: bool = False
    video: str = ""
    pending_call: str = ""
    # read-only information
    range_km: int = 312
    battery_pct: int = 64
    tyre_pressure_bar: dict = field(default_factory=lambda: {"front_left": 2.5, "front_right": 2.5, "rear_left": 2.4, "rear_right": 2.2})
    service_due_km: int = 8400

    def snapshot(self):
        return deepcopy(asdict(self))


class Vehicle:
    """Executes tool calls on the simulated state and returns a short factual result."""

    def __init__(self, state: VehicleState | None = None):
        self.state = state or VehicleState()

    def execute(self, name: str, args: dict) -> str:
        return getattr(self, name)(**args)

    def set_temperature(self, zone="all", celsius=None, change=None):
        zones = ["driver", "passenger"] if zone in (None, "all") else [zone]
        for z in zones:
            if celsius is not None:
                self.state.temperature[z] = float(celsius)
            elif change is not None:
                self.state.temperature[z] = float(self.state.temperature[z] + change)
        return ", ".join(f"{z} {self.state.temperature[z]:.1f} °C" for z in zones)

    def set_fan(self, level=None, change=None):
        self.state.fan = int(level if level is not None else self.state.fan + (change or 0))
        return f"fan {self.state.fan}"

    def set_seat_heating(self, level, seat="driver"):
        self.state.seat_heating[seat or "driver"] = int(level)
        return f"{seat or 'driver'} seat heating {level}"

    def set_window(self, action, window="all", percent=None):
        target = (percent if percent is not None else 100) if action == "open" else 0
        for w in ([k for k in self.state.windows] if window in (None, "all") else [window]):
            self.state.windows[w] = int(target)
        return f"{window or 'all'} window(s) {action}" + (f" {percent}%" if percent is not None and action == "open" else "")

    def navigate_to(self, destination):
        self.state.destination = destination
        return f"route to {destination} started"

    def find_nearby(self, category):
        return f"3 {category.replace('_', ' ')}s found, nearest 1.2 km"

    def play_media(self, query=None, source=None):
        self.state.media = query or source or "radio"
        self.state.media_playing = True
        return f"playing {self.state.media}"

    def media_control(self, action):
        self.state.media_playing = action != "pause"
        return f"media {action}"

    def set_volume(self, level=None, change=None):
        self.state.volume = int(level if level is not None else self.state.volume + (change or 0))
        return f"volume {self.state.volume}"

    def call_contact(self, name):
        self.state.pending_call = name
        return f"ready to call {name}, waiting for driver confirmation"

    def get_vehicle_status(self, item):
        s = self.state
        return {"range": f"{s.range_km} km remaining", "battery": f"battery {s.battery_pct}%",
                "tyre_pressure": ", ".join(f"{k.replace('_', ' ')} {v} bar" for k, v in s.tyre_pressure_bar.items())
                                 + " (rear right is low, recommended 2.5 bar)",
                "service": f"next service in {s.service_due_km} km", "doors": "all doors closed and locked",
                "oil": "electric drive: no engine oil", "fuel": "electric vehicle, see battery"}[item]

    def set_drive_mode(self, mode):
        self.state.drive_mode = mode
        return f"{mode} mode"

    def set_ambient_light(self, colour):
        self.state.ambient_light = colour
        return f"ambient light {colour}"

    def open_trunk(self):
        self.state.trunk_open = True
        return "trunk opened"

    def play_video(self, query=None):
        self.state.video = query or "video"
        return f"playing video {self.state.video}"

    def manual_lookup(self, question):
        from manual import Handbook
        return Handbook().answer(question)
