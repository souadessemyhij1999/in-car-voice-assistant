"""
Tests the Claude backend through the real Anthropic Python SDK, against a simulated API server
(httpx.MockTransport) that replies in the documented Messages API format. No network or API key needed.

Checks: the request the SDK sends (model, tools, system prompt), the tool-use loop, the safety layer
blocking a call and the BLOCKED result being sent back to Claude, and the cost accounting.

    pytest -q tests
"""

import json
import sys
from pathlib import Path

import anthropic
try:
    import httpx2 as httpx  # anthropic SDK >= 2 uses httpx2
except ImportError:
    import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import llm  # noqa: E402
from vehicle import TOOL_NAMES, Vehicle, VehicleState  # noqa: E402


def message(content, stop_reason, n_in=900, n_out=40):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": llm.MODEL, "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": n_in, "output_tokens": n_out}}


def make_backend(script):
    """script: list of responses the fake API returns in order. Records every request body."""
    requests = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=script[len(requests) - 1])

    backend = llm.ClaudeBackend.__new__(llm.ClaudeBackend)
    backend.client = anthropic.Anthropic(api_key="test-key", base_url="https://api.test",
                                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    from safety import SafetyGuard
    backend.guard, backend.max_rounds = SafetyGuard(), 3
    return backend, requests


def test_request_contains_model_tools_and_system_prompt():
    backend, requests = make_backend([message([{"type": "text", "text": "Hello!"}], "end_turn")])
    r = backend.handle("hello", Vehicle(VehicleState(speed_kmh=50)))
    body = requests[0]
    assert body["model"] == "claude-haiku-5-5"
    assert [t["name"] for t in body["tools"]] == TOOL_NAMES
    assert "voice assistant of an electric car" in body["system"]
    assert "speed 50 km/h" in body["messages"][0]["content"]
    assert r["reply"] == "Hello!" and r["executed"] == []


def test_tool_call_is_executed_and_result_returned():
    script = [
        message([{"type": "text", "text": "Sure."},
                 {"type": "tool_use", "id": "toolu_1", "name": "set_temperature", "input": {"change": 2}}], "tool_use"),
        message([{"type": "text", "text": "I've made it a little warmer."}], "end_turn"),
    ]
    backend, requests = make_backend(script)
    v = Vehicle(VehicleState(speed_kmh=50))
    r = backend.handle("I'm freezing", v)
    assert r["executed"] == [{"name": "set_temperature", "args": {"celsius": 23.0}}]   # relative change made absolute
    assert v.state.temperature["driver"] == 23.0
    second = requests[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][1]["type"] == "tool_use"
    result = second[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert r["reply"] == "I've made it a little warmer."


def test_safety_layer_blocks_and_tells_claude():
    script = [
        message([{"type": "tool_use", "id": "toolu_9", "name": "play_video", "input": {"query": "a movie"}}], "tool_use"),
        message([{"type": "text", "text": "Videos only work when parked. Shall I play an audiobook instead?"}], "end_turn"),
    ]
    backend, requests = make_backend(script)
    v = Vehicle(VehicleState(speed_kmh=100))
    r = backend.handle("Play a movie", v)
    assert r["executed"] == [] and r["blocked"][0]["name"] == "play_video"
    assert v.state.video == ""                                            # nothing happened in the car
    assert requests[1]["messages"][2]["content"][0]["content"].startswith("BLOCKED")


def test_multiple_tools_in_one_turn_and_cost():
    script = [
        message([{"type": "tool_use", "id": "a", "name": "navigate_to", "input": {"destination": "home"}},
                 {"type": "tool_use", "id": "b", "name": "call_contact", "input": {"name": "Mom"}}], "tool_use", 1000, 60),
        message([{"type": "text", "text": "Route home started, calling Mom once you confirm."}], "end_turn", 1100, 20),
    ]
    backend, _ = make_backend(script)
    r = backend.handle("Navigate home and call Mom", Vehicle(VehicleState(speed_kmh=50)))
    assert [c["name"] for c in r["executed"]] == ["navigate_to", "call_contact"]
    assert r["tokens_in"] == 2100 and r["tokens_out"] == 80
    assert abs(r["cost_usd"] - (2100 * 0.10 + 80 * 0.50) / 1e6) < 1e-12


def test_unknown_function_cannot_run():
    script = [
        message([{"type": "tool_use", "id": "x", "name": "disable_airbag", "input": {}}], "tool_use"),
        message([{"type": "text", "text": "I can't do that."}], "end_turn"),
    ]
    backend, _ = make_backend(script)
    r = backend.handle("Switch off the airbags", Vehicle(VehicleState(speed_kmh=100)))
    assert r["executed"] == [] and r["blocked"][0]["name"] == "disable_airbag"


def test_empty_answer_gets_a_fallback_reply():
    backend, _ = make_backend([message([], "max_tokens", 900, 1024)])
    r = backend.handle("Ich verstehe die Musik kaum", Vehicle(VehicleState(speed_kmh=50)))
    assert r["executed"] == [] and r["reply"].startswith("Sorry")
