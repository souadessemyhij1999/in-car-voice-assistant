"""
Charts and the results table for the README, from results/<backend>/summary.json + details.csv.
Backends that have not been run yet (e.g. claude / hybrid before the Colab notebook) are skipped.

    python src/report.py
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, ASSETS = ROOT / "results", ROOT / "assets"
LABELS = {"local": "Offline in-car model", "claude": "Claude only (cloud)", "hybrid": "Hybrid: offline + Claude"}
COLOURS = {"local": "#9aa5b8", "claude": "#f08c00", "hybrid": "#1c69d4"}
ORDER = ["climate", "comfort", "navigation", "media", "phone", "status", "modes", "multi_step", "indirect",
         "handbook", "out_of_scope", "unsafe"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ASSETS.mkdir(exist_ok=True)
    runs = {b: json.load(open(RES / b / "summary.json")) for b in LABELS if (RES / b / "summary.json").exists()}
    details = {b: pd.read_csv(RES / b / "details.csv") for b in runs}

    # success by category, one bar per backend
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(runs)
    for k, (b, s) in enumerate(runs.items()):
        vals = [s["by_category"].get(c, 0) * 100 for c in ORDER]
        ax.bar([i + (k - (len(runs) - 1) / 2) * w for i in range(len(ORDER))], vals, w, color=COLOURS[b],
               label=f"{LABELS[b]} ({s['task_success']:.0%} overall)")
    counts = details[next(iter(details))].category.value_counts()
    ax.set_xticks(range(len(ORDER)), [f"{c.replace('_', ' ')}\n({counts.get(c, 0)})" for c in ORDER], fontsize=8)
    ax.set(ylabel="commands done correctly (%)", ylim=(0, 108), title="Task success by category (number of test commands)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "success_by_category.png", dpi=150)

    # offline model: coverage vs accuracy for different confidence thresholds
    if "local" in runs:
        c = pd.DataFrame(runs["local"]["threshold_curve"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(c.handled * 100, c.success_when_handled * 100, "o-", color="#1c69d4")
        for _, r in c.iterrows():
            ax.annotate(f"{r.threshold:.2f}", (r.handled * 100, r.success_when_handled * 100), textcoords="offset points",
                        xytext=(-8, 8) if r.threshold == 0.6 else (5, -12), fontsize=8)
        cur = c[c.threshold == 0.55].iloc[0]
        ax.scatter([cur.handled * 100], [cur.success_when_handled * 100], s=160, facecolor="none", edgecolor="#d6336c",
                   lw=2, label="threshold used (0.55, set before testing)")
        ax.set(xlabel="commands the offline model handles itself (%)", ylabel="correct when it handles them (%)",
               title="Offline model: coverage vs. accuracy\n(labels = confidence threshold)", ylim=(90, 101))
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        style(ax)
        fig.tight_layout()
        fig.savefig(ASSETS / "offline_threshold.png", dpi=150)

    # comparison table (markdown) for the README
    rows = []
    for b, s in runs.items():
        rows.append({"Backend": LABELS[b], "Task success": f"{s['task_success']:.1%}",
                     "Unsafe actions executed": s["unsafe_actions_executed"],
                     "Unsafe requests refused": s["unsafe_requests_refused"],
                     "Handled in the car": f"{s['routes'].get('local', 0)}/{s['commands']}",
                     "Median latency": " / ".join(f"{k} {v:.0f} ms" for k, v in s["latency_ms_median_by_route"].items()),
                     "Cost per 1,000 commands": f"${s['cost_usd_per_1000_commands']:.2f}"})
    table = pd.DataFrame(rows)
    md = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n" + \
         "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in table.values)
    (RES / "comparison.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
