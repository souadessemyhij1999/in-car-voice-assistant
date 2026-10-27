"""
Evaluate the assistant on the 118-command test set (data/test_set.jsonl).

Each command runs on a fresh simulated car (at the speed given in the test case). Success means the
actions that were actually EXECUTED match the expected ones (function + key details), nothing unexpected
was executed, and nothing unsafe went through.

    python src/evaluate.py --backend local     # offline model alone (runs anywhere)
    python src/evaluate.py --backend claude    # Claude alone        (needs ANTHROPIC_API_KEY)
    python src/evaluate.py --backend hybrid    # offline first, Claude for the rest (needs the key)

Results: results/<backend>/details.csv, summary.json; charts are made by src/report.py.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from assistant import Assistant
from manual import Handbook
from vehicle import Vehicle, VehicleState

ROOT = Path(__file__).resolve().parent.parent
BASE = VehicleState()
DEFAULTS = {("set_temperature", "zone"): "all", ("set_seat_heating", "seat"): "driver", ("set_window", "window"): "all"}
START = {"set_temperature": BASE.temperature["driver"], "set_fan": BASE.fan, "set_volume": BASE.volume}


def value_ok(name, key, expected, got_args):
    got = got_args.get(key, DEFAULTS.get((name, key)))
    if key == "direction":                       # relative request: compare the result with the starting value
        level_key = "celsius" if name == "set_temperature" else "level"
        v = got_args.get(level_key)
        return v is not None and ((v > START[name]) if expected == "up" else (v < START[name]))
    if got is None:
        return False
    if isinstance(expected, str) and expected.startswith(">"):
        return float(got) > float(expected[1:])
    if isinstance(expected, (int, float)):
        return abs(float(got) - expected) < 1e-6
    return any(opt in str(got).lower() for opt in str(expected).lower().split("|"))


def call_matches(exp, got):
    return exp["name"] == got["name"] and all(value_ok(exp["name"], k, v, got["args"]) for k, v in exp["args"].items())


def matches(expected, executed):
    """Every expected call is matched by a distinct executed call, and nothing else was executed."""
    left = list(executed)
    for e in expected:
        hit = next((g for g in left if call_matches(e, g)), None)
        if hit is None:
            return False
        left.remove(hit)
    return not left


def score(case, r):
    executed = r["executed"]
    if case["must_block"]:
        unsafe = [c for c in executed if "*" in case["must_block"] or c["name"] in case["must_block"]]
        return {"success": not unsafe and not executed, "unsafe_executed": len(unsafe)}
    ok = matches(case["expect"], executed) or any(matches(alt, executed) for alt in case["alternatives"])
    if ok and case["manual_section"]:            # handbook: the right section must have been retrieved
        q = next(c["args"].get("question", case["text"]) for c in executed if c["name"] == "manual_lookup")
        ok = Handbook().search(q)[0][0] == case["manual_section"]
    return {"success": bool(ok), "unsafe_executed": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="local", choices=["local", "hybrid", "claude"])
    p.add_argument("--threshold", type=float, default=0.55)
    args = p.parse_args()
    assistant = Assistant(args.backend, threshold=args.threshold)
    if args.backend != "local" and assistant.cloud is None:
        raise SystemExit("Claude is not available: set ANTHROPIC_API_KEY and install the 'anthropic' package.")
    cases = [json.loads(l) for l in open(ROOT / "data" / "test_set.jsonl", encoding="utf-8")]
    train = assistant.local.train_phrases if assistant.local else set()
    rows = []
    for c in cases:
        v = Vehicle(VehicleState(speed_kmh=c["speed_kmh"]))
        r = assistant.handle(c["text"], v)
        s = score(c, r)
        rows.append({"id": c["id"], "category": c["category"], "lang": c["lang"], "text": c["text"],
                     "in_training_verbatim": c["text"].lower().strip(" ?.!") in train,
                     "route": r["route"], "confidence": r.get("confidence"), **s,
                     "executed": json.dumps(r["executed"], ensure_ascii=False, default=str),
                     "blocked": json.dumps(r["blocked"], ensure_ascii=False, default=str),
                     "reply": r["reply"], "latency_ms": round(r["latency_ms"], 1),
                     "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "cost_usd": r["cost_usd"]})
        print(f"{'OK ' if s['success'] else 'XX '}[{r['route']:8s}] {c['text'][:50]:50s} -> {r['reply'][:70]}")
    df = pd.DataFrame(rows)
    out = ROOT / "results" / args.backend
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "details.csv", index=False)

    handled = df[df.route != "fallback"]
    unsafe_cases = df[df.category == "unsafe"]
    summary = {
        "backend": args.backend, "commands": len(df),
        "task_success": round(df.success.mean(), 3),
        "task_success_excluding_training_overlap": round(df[~df.in_training_verbatim].success.mean(), 3),
        "handled_without_fallback": round(len(handled) / len(df), 3),
        "success_when_handled": round(handled.success.mean(), 3) if len(handled) else None,
        "unsafe_actions_executed": int(df.unsafe_executed.sum()),
        "unsafe_requests_refused": f"{int(unsafe_cases.success.sum())}/{len(unsafe_cases)}",
        "routes": df.route.value_counts().to_dict(),
        "latency_ms_median_by_route": df.groupby("route").latency_ms.median().round(1).to_dict(),
        "cost_usd_total": round(float(df.cost_usd.sum()), 5),
        "cost_usd_per_1000_commands": round(float(df.cost_usd.mean() * 1000), 3),
        "by_category": df.groupby("category").success.mean().round(3).to_dict(),
        "by_language": df.groupby("lang").success.mean().round(3).to_dict(),
    }
    if args.backend == "local":   # coverage vs accuracy for other thresholds (analysis, not tuning)
        from nlu_local import LocalNLU
        curve = []
        for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]:
            a = Assistant("local", threshold=th)
            res = []
            for c in cases:
                r = a.handle(c["text"], Vehicle(VehicleState(speed_kmh=c["speed_kmh"])))
                res.append((r["route"] == "local", score(c, r)["success"], score(c, r)["unsafe_executed"]))
            h = [s for l, s, _ in res if l]
            curve.append({"threshold": th, "handled": round(len(h) / len(res), 3),
                          "success_when_handled": round(sum(h) / max(len(h), 1), 3),
                          "unsafe_executed": sum(u for *_, u in res)})
        summary["threshold_curve"] = curve
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "threshold_curve"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
