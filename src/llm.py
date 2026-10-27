"""
Cloud LLM backend: Claude with tool use.

Claude receives the same vehicle functions (TOOLS) as the offline model. Every function call it makes
goes through the SafetyGuard before it touches the car; blocked calls are reported back to Claude as a
tool result so it can explain the refusal to the driver.

Needs the environment variable ANTHROPIC_API_KEY. Model: claude-haiku-5-5 (Anthropic's fastest model,
built for latency-sensitive routing and extraction). Prices used for the cost estimate:
$0.10 per million input tokens, $0.50 per million output tokens (Claude Haiku 5.5, prompts < 100k tokens).
"""

import json
import os
import time

from safety import SafetyGuard
from vehicle import TOOLS, Vehicle

MODEL = os.environ.get("ASSISTANT_MODEL", "claude-haiku-5-5")
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6   # USD per token

SYSTEM = """You are the voice assistant of an electric car. The driver speaks to you while driving.
- Use the provided vehicle functions to act. Never claim you did something unless a function result confirms it.
- For questions about how the car works, warning lights or maintenance, call manual_lookup and answer from its result.
- If the driver states a need instead of a command ("I'm cold", "I'm hungry"), choose the most helpful function.
- Several requests in one sentence: call all the needed functions.
- If a function result says BLOCKED, explain briefly why and offer an alternative if there is one.
- Requests that no function can fulfil (e.g. disabling safety systems): politely say you can't.
- Reply in the driver's language (English or German), in at most two short sentences suitable for reading aloud.
  No lists, no markdown, no emojis."""


class ClaudeBackend:
    def __init__(self, max_rounds=3):
        import anthropic  # imported here so the offline parts run without the package
        self.client = anthropic.Anthropic()
        self.guard = SafetyGuard()
        self.max_rounds = max_rounds

    def handle(self, text, vehicle: Vehicle):
        s = vehicle.state
        context = (f"Vehicle state: speed {s.speed_kmh:.0f} km/h, cabin {s.temperature['driver']:.0f} °C, "
                   f"fan {s.fan}, volume {s.volume}, drive mode {s.drive_mode}, battery {s.battery_pct}%.")
        messages = [{"role": "user", "content": f"{context}\nDriver: {text}"}]
        requested, executed, blocked, notes = [], [], [], []
        tokens_in = tokens_out = 0
        t0 = time.perf_counter()
        reply = ""
        for _ in range(self.max_rounds):
            r = self.client.messages.create(model=MODEL, max_tokens=400, system=SYSTEM, tools=TOOLS, messages=messages)
            tokens_in += r.usage.input_tokens
            tokens_out += r.usage.output_tokens
            texts = [b.text for b in r.content if b.type == "text"]
            if r.stop_reason != "tool_use":
                reply = " ".join(texts).strip()
                break
            messages.append({"role": "assistant", "content": r.content})
            results = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                call = {"name": b.name, "args": dict(b.input)}
                requested.append(call)
                d = self.guard.check(b.name, call["args"], s)
                if not d.allowed:
                    blocked.append(call)
                    content = f"BLOCKED: {d.note}"
                else:
                    out = vehicle.execute(b.name, d.args)
                    executed.append({"name": b.name, "args": d.args})
                    content = out + (f" (note: {d.note})" if d.note else "") + \
                              (" (call prepared; the driver confirms on the steering wheel)" if d.needs_confirmation else "")
                    if d.note:
                        notes.append(d.note)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
            messages.append({"role": "user", "content": results})
        return {"route": "cloud", "requested": requested, "executed": executed, "blocked": blocked, "reply": reply,
                "latency_ms": (time.perf_counter() - t0) * 1000, "tokens_in": tokens_in, "tokens_out": tokens_out,
                "cost_usd": tokens_in * PRICE_IN + tokens_out * PRICE_OUT}


def available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    from vehicle import VehicleState
    b = ClaudeBackend()
    print(json.dumps(b.handle("I'm freezing, and take me home", Vehicle(VehicleState(speed_kmh=50))), indent=2, default=str))
