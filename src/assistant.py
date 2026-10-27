"""
The assistant: routes each request to the offline in-car model or to Claude, executes the result
through the safety layer, and produces a short spoken reply.

Routing (hybrid):
  offline model confident  -> handled in the car: ~1 ms, no network, predictable
  otherwise                -> Claude (cloud) with the same vehicle functions
  no connection / no key   -> safe fallback reply, nothing is executed

Usage:
    python src/assistant.py                         # interactive text chat
    python src/assistant.py --text "Mir ist kalt"   # one command
    python src/assistant.py --speed 100 --backend local
    python src/assistant.py --voice                 # microphone + speech (see README: needs speech models)
"""

import argparse
import re
import time

from llm import ClaudeBackend, available
from nlu_local import LocalNLU
from safety import SafetyGuard
from vehicle import Vehicle, VehicleState

GERMAN = re.compile(r"[äöüß]|\b(bitte|mach|mir|ist|und|nach|auf|der|die|das|ich|stell|spiel|ruf|wie|wo|zu|an|aus)\b", re.I)


def is_german(text):
    return bool(GERMAN.search(text))


def say(name, args, result, de):
    """Short spoken confirmation for an action done by the offline model."""
    a = args
    en = {
        "set_temperature": lambda: f"Setting {a.get('zone', 'all') if a.get('zone') not in (None, 'all') else 'the cabin'} to {a.get('celsius', 0):.0f} degrees.",
        "set_fan": lambda: f"Fan set to {a.get('level')}.",
        "set_seat_heating": lambda: f"{a.get('seat', 'driver').capitalize()} seat heating {'off' if a.get('level') == 0 else 'on level ' + str(a.get('level'))}.",
        "set_window": lambda: f"{'Opening' if a['action'] == 'open' else 'Closing'} {'all windows' if a.get('window', 'all') == 'all' else 'the ' + a['window'].replace('_', ' ') + ' window'}.",
        "navigate_to": lambda: f"Starting route guidance to {a['destination']}.",
        "find_nearby": lambda: f"Searching for {a['category'].replace('_', ' ')}s nearby. {result.split(',')[1].strip().capitalize()}.",
        "play_media": lambda: f"Playing {a.get('query') or a.get('source') or 'music'}.",
        "media_control": lambda: {"pause": "Paused.", "resume": "Resuming.", "next": "Next track.", "previous": "Previous track."}[a["action"]],
        "set_volume": lambda: f"Volume {a.get('level')}.",
        "call_contact": lambda: f"Calling {a['name']}. Please confirm on the steering wheel.",
        "get_vehicle_status": lambda: result[0].upper() + result[1:] + ".",
        "set_drive_mode": lambda: f"{a['mode'].capitalize()} mode.",
        "set_ambient_light": lambda: f"Ambient lighting {a['colour']}.",
        "open_trunk": lambda: "Opening the trunk.",
        "play_video": lambda: "Playing the video.",
        "manual_lookup": lambda: result,
    }
    de_ = {
        "set_temperature": lambda: f"Temperatur auf {a.get('celsius', 0):.0f} Grad.",
        "set_fan": lambda: f"Gebläse auf Stufe {a.get('level')}.",
        "set_seat_heating": lambda: "Sitzheizung aus." if a.get("level") == 0 else f"Sitzheizung Stufe {a.get('level')}.",
        "set_window": lambda: "Fenster werden geöffnet." if a["action"] == "open" else "Fenster werden geschlossen.",
        "navigate_to": lambda: f"Route nach {a['destination']} wird gestartet.",
        "find_nearby": lambda: "Ich suche in der Nähe.",
        "play_media": lambda: f"Ich spiele {a.get('query') or 'Radio'}.",
        "media_control": lambda: {"pause": "Pausiert.", "resume": "Weiter geht's.", "next": "Nächster Titel.", "previous": "Vorheriger Titel."}[a["action"]],
        "set_volume": lambda: f"Lautstärke {a.get('level')}.",
        "call_contact": lambda: f"Ich rufe {a['name']} an. Bitte am Lenkrad bestätigen.",
        "get_vehicle_status": lambda: result + ".",
        "set_drive_mode": lambda: f"Fahrmodus {a['mode']}.",
        "set_ambient_light": lambda: f"Ambientebeleuchtung {a['colour']}.",
        "open_trunk": lambda: "Kofferraum wird geöffnet.",
        "play_video": lambda: "Video wird abgespielt.",
        "manual_lookup": lambda: result,
    }
    return (de_ if de else en)[name]()


class Assistant:
    def __init__(self, backend="hybrid", threshold=0.55):
        self.backend = backend
        self.local = LocalNLU(threshold=threshold) if backend in ("hybrid", "local") else None
        self.cloud = ClaudeBackend() if backend in ("hybrid", "claude") and available() else None
        self.guard = SafetyGuard()

    def handle(self, text, vehicle: Vehicle):
        t0 = time.perf_counter()
        de = is_german(text)
        if self.local is not None:
            calls, conf = self.local.parse(text)
            if calls is not None:
                executed, blocked, replies = [], [], []
                for c in calls:
                    d = self.guard.check(c["name"], c["args"], vehicle.state)
                    if not d.allowed:
                        blocked.append(c)
                        replies.append(d.note)
                        continue
                    result = vehicle.execute(c["name"], d.args)
                    executed.append({"name": c["name"], "args": d.args})
                    replies.append(say(c["name"], d.args, result, de) + (f" ({d.note})" if d.note else ""))
                return {"route": "local", "confidence": round(conf, 3), "requested": calls, "executed": executed,
                        "blocked": blocked, "reply": " ".join(replies),
                        "latency_ms": (time.perf_counter() - t0) * 1000, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        if self.cloud is not None:
            return self.cloud.handle(text, vehicle)
        reply = ("Das kann ich offline leider nicht. Bitte anders formulieren." if de
                 else "I can't do that offline. Could you rephrase it as a command?")
        return {"route": "fallback", "requested": [], "executed": [], "blocked": [], "reply": reply,
                "latency_ms": (time.perf_counter() - t0) * 1000, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}


def voice_loop(assistant, vehicle):
    """Microphone -> Whisper speech-to-text -> assistant -> spoken reply. Downloads speech models on first use."""
    import numpy as np
    import pyttsx3
    import sounddevice as sd
    from faster_whisper import WhisperModel
    stt, tts = WhisperModel("small", compute_type="int8"), pyttsx3.init()
    print("Speak after the beep (5 s per command). Ctrl+C to stop.")
    while True:
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        segments, _ = stt.transcribe(np.squeeze(audio), beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        if not text:
            continue
        r = assistant.handle(text, vehicle)
        print(f"You: {text}\nCar [{r['route']}]: {r['reply']}")
        tts.say(r["reply"])
        tts.runAndWait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text")
    p.add_argument("--backend", default="hybrid", choices=["hybrid", "local", "claude"])
    p.add_argument("--speed", type=float, default=50.0, help="vehicle speed in km/h (safety rules depend on it)")
    p.add_argument("--voice", action="store_true")
    args = p.parse_args()
    a, v = Assistant(args.backend), Vehicle(VehicleState(speed_kmh=args.speed))
    if args.backend != "local" and a.cloud is None:
        print("(Claude not connected: set ANTHROPIC_API_KEY for the cloud part. Running offline only.)")
    if args.voice:
        return voice_loop(a, v)
    lines = [args.text] if args.text else iter(lambda: input("You: "), "")
    for text in lines:
        r = a.handle(text, v)
        print(f"Car [{r['route']}, {r['latency_ms']:.0f} ms]: {r['reply']}")


if __name__ == "__main__":
    main()
