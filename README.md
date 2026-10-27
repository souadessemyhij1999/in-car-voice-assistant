# 🎙️ In-Car Voice Assistant: Offline Model + Claude + Safety Layer

A bilingual (English/German) voice assistant for a car that splits the work the way production systems do:

- a **small offline model in the car** handles everyday commands in **a few milliseconds**, without a network,
- **Claude** (cloud LLM) handles everything it isn't sure about: indirect requests ("I'm freezing"), handbook questions and unusual phrasing,
- a **deterministic safety layer** sits between both "brains" and the car, so no model, however it is prompted, can play a video while driving or open the boot at speed.

![Demo](assets/demo.gif)
<sub>Scripted drive in offline mode. Each reply shows where it was handled and how fast. Right: live vehicle state.</sub>

---

## How it works

```
 driver's voice ──► speech-to-text (Whisper, optional) ──► text
                                                            │
                         ┌──────────────────────────────────┴──────────────┐
                         ▼                                                 │ not confident
             Offline in-car model (~ms)                                    ▼
             intent: TF-IDF + logistic regression          Claude Haiku 5.5 with tool use
             details: rules (numbers, seats, places)       (same 16 vehicle functions)
                         │                                                 │
                         └──────────────► SAFETY LAYER ◄───────────────────┘
                                          stationary-only functions, value limits,
                                          call confirmation, unknown functions refused
                                                  │
                                                  ▼
                                   vehicle functions ──► spoken reply (text-to-speech, optional)
```

| Design choice | Why |
|---|---|
| **Hybrid routing** | Most in-car commands are simple ("Temperatur auf 21 Grad"). A tiny local model answers them instantly, free and offline (tunnels, car parks). The LLM is only paid for and waited for when it adds value. |
| **One function list for both brains** | The 16 vehicle functions are defined once and used by both the offline model and Claude, so behaviour and safety rules are identical whichever brain answers. |
| **"AI suggests, deterministic code decides"** | An LLM can be talked into things or make mistakes. The safety layer is plain code with fixed rules, and every action from either brain passes through it. |
| **Confidence threshold** | The offline model hands over when it isn't sure (threshold 0.55, set before testing) instead of guessing. A wrong action in a car is worse than a short delay. |
| **Bilingual** | English and German commands, replies in the driver's language. |

---

## Results

**Test set:** 118 commands (72 English, 46 German) in 12 categories, each with the vehicle action expected. A command counts as done correctly only if exactly the expected actions were executed, with the right details (for example, the right seat, temperature and destination).

### Headline: three ways to run it

| | Offline model only | Claude only (cloud) | **Hybrid: offline + Claude** |
|---|---|---|---|
| Commands done correctly | 81.4% (96/118) | 95.8% (113/118) | **94.9% (112/118)** |
| Answered in the car, offline | 87/118 | 0/118 | **87/118 (74%)** |
| Median response time | a few ms | 1.8 s | **3.5 ms** in the car / 2.0 s when Claude is asked |
| Cost per 1,000 commands | $0 | $0.52 | **$0.13** |
| Unsafe actions executed | 0 (8/8 refused) | 0 (8/8 refused) | **0 (8/8 refused)** |
| English / German | 82% / 80% | 94% / 98% | 93% / 98% |

<sub>Claude = `claude-haiku-5-5` via the Claude API. Measured with [`notebooks/evaluate_llm_colab.ipynb`](notebooks/evaluate_llm_colab.ipynb) on Google Colab, one run per setup. Raw results per command: [`results/`](results).</sub>

**The hybrid is within one command of Claude alone, at a quarter of the cost, and three in four commands never leave the car.**

![Success by category](assets/success_by_category.png)

### Offline in-car model

| Metric | Result |
|---|---|
| Commands it handles itself | **87 of 118 (74%)** |
| Correct when it handles them | **87 of 87 (100%)** |
| Handbook search (finding the right section) | 13 of 13 questions |

The offline model is strong exactly where it should be: direct commands, multi-step commands ("Navigate home and call Mom" is split and both actions run) and safety. It deliberately **passes on** indirect requests ("I'm freezing") and about half of the handbook questions, which is what Claude is for.

![Coverage vs accuracy](assets/offline_threshold.png)

The confidence threshold trades coverage for accuracy. At 0.55, chosen before testing, the model is still 100% correct while handling 74% of commands. Lowering the threshold to handle more drops accuracy quickly (93% correct at 0.30).

### What went wrong: every failure of the hybrid (6 of 118)

All six were commands passed on to Claude. The scoring is strict: exactly the expected actions, nothing extra.

| Command | What happened | Kind of error |
|---|---|---|
| "It's boiling in here" | cooled to 18 °C **and** raised the fan | extra action (4 cases) |
| "It's really stuffy, I need some fresh air" | opened the windows **and** raised the fan | extra action |
| "I'm getting sleepy, I could use a coffee" | found coffee places **and** raised the fan | extra action |
| "My back is cold" | seat heating **and** +2 °C | extra action |
| "Ich verstehe die Musik kaum" | no answer: the reply hit the 400-token output limit | **bug, fixed after the run** (see below) |
| "A yellow light with an exclamation mark came on…" | Claude rephrased it as "yellow warning light", so the search returned the general warning-light section instead of tyre pressure | retrieval |

Claude alone made 5 errors: 2 of the same "extra action" kind, the same handbook miss, plus "Open my window" (it opened **all** windows, a real mistake) and "Sitzheizung aus" (switched off every seat, not just the driver's; arguably reasonable).

**Run-to-run variation:** "It's boiling in here" was correct in the Claude-only run and wrong in the hybrid run, although the same model answered it. With one run per setup, a difference of 1–3 commands between Claude-only and hybrid (e.g. indirect requests: 10/12 vs 7/12) is within normal variation, not a real difference.

**Fixed after the evaluation (not yet re-measured):** the output limit is raised to 1,024 tokens, and if Claude ever returns no answer, the assistant now says "Sorry, I didn't get that. Could you say it again?" instead of staying silent (covered by a new test).

### Tests

The Claude backend is tested through the **real Anthropic Python SDK** against a simulated API server that answers in the documented Messages API format ([`tests/test_claude_backend.py`](tests/test_claude_backend.py)). The tests cover the request (model, the 16 functions, system prompt and vehicle state), the tool-use loop including several functions in one turn, a blocked call reported back to Claude as `BLOCKED`, a made-up function such as `disable_airbag` being refused, the empty-answer fallback and the cost accounting. Together with the safety-layer tests: **10 tests**.

---

## Key observations

- **Most of an in-car assistant doesn't need an LLM.** The hybrid answered 74% of commands in the car, in milliseconds and for free, and still reached 94.9%, against 95.8% for sending everything to Claude. It costs a quarter as much and keeps working in a tunnel for the commands drivers use most.
- **Knowing when not to answer is the important skill.** The offline model never acted on a request it got wrong; everything uncertain went to Claude. In a car, "let me ask the cloud" is far better than opening the wrong window.
- **Safety must not depend on the model.** All 8 unsafe requests were refused at 100 km/h in all three setups (video, boot, "switch off the airbags", "deactivate lane assist"). The same rule layer limits values (35 °C becomes 28 °C) and holds phone calls for confirmation, whichever brain made the request.
- **The LLM's typical mistake is doing too much, not too little.** 4 of the 6 hybrid errors were the right action plus an extra one (usually raising the fan). For a car, that raises a design question: should an assistant act on its own interpretation, or do the minimum and offer the rest ("Shall I also turn up the fan?")? A stricter system prompt is the obvious next experiment.
- **Testing found real bugs.** The German test cases found "Musik pausieren" skipping to the next track (fixed before the Claude run; the offline model was 98.9% correct when handling a command before the fix). The Claude run found that an empty answer was possible (fixed afterwards).
- **Claude was stronger in German than in English** (98% vs 94%). The offline model was balanced (80% vs 82%).

---

## Try it

```bash
pip install -r requirements.txt

python src/assistant.py                                  # chat by typing (offline mode)
python src/assistant.py --text "Sitzheizung Fahrer Stufe 2"
python src/assistant.py --speed 100 --text "Open the trunk"      # refused while moving

export ANTHROPIC_API_KEY=...                             # optional: connect Claude
python src/assistant.py --text "I'm freezing, and take me home"

pip install -r requirements-voice.txt                    # optional: real voice
python src/assistant.py --voice                          # microphone -> Whisper -> assistant -> speech

python -m pytest -q tests                                # safety rules + Claude backend tests (10)
python src/evaluate.py --backend local                   # offline evaluation
```

## Project structure

```
in-car-voice-assistant/
├── src/
│   ├── vehicle.py          # simulated car + the 16 vehicle functions (shared by both brains)
│   ├── safety.py           # deterministic safety layer
│   ├── nlu_local.py        # offline model: intent classifier + rule-based details, EN/DE
│   ├── llm.py              # Claude backend with tool use and cost tracking
│   ├── assistant.py        # routing, spoken replies, text / voice interface
│   ├── manual.py           # owner's handbook search
│   ├── evaluate.py         # runs the 118 test commands for a backend
│   ├── report.py           # charts + comparison table
│   └── demo.py             # renders the infotainment demo animation
├── data/test_set.jsonl     # 118 test commands with expected actions
├── tests/                  # safety-layer and Claude-backend tests (10 tests)
├── notebooks/evaluate_llm_colab.ipynb   # measures the Claude and hybrid results (Colab)
└── results/                # per-command details and summaries
```

---

## Limitations & next steps

- **The test set was written by me**, not recorded from real drivers, and the offline model's training phrases were generated from templates written by the same person. 8 of the 118 test commands also appear verbatim among the 914 training phrases (very short ones like "Sitzheizung aus"); without them, overall success is 80.0%.
- **Text-based evaluation.** Speech recognition errors (road noise, accents) aren't measured. Voice input and output are included as an option but weren't tested here, because the speech models couldn't be downloaded in my environment.
- **The vehicle and handbook are simulated.** The handbook is written for this project (a fictional electric car), because real manuals are copyrighted.
- **One run per setup.** Claude's answers vary between runs, so small differences between setups are not meaningful; several runs would give error bars.
- **Next:** re-run after the empty-answer fix, try a "do only what was asked" system prompt, pass the driver's original wording to the handbook search, add conversation memory ("make it a bit warmer *again*"), and test with recorded in-car audio.

## Tech stack

Python · scikit-learn · Claude API (tool use, `claude-haiku-5-5`) · Anthropic Python SDK · pytest · Whisper (faster-whisper) · pyttsx3 · Pillow · matplotlib

## Credits & licences

- Claude API and SDK: [Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- Project idea from [project-based-learning](https://github.com/practical-tutorials/project-based-learning) ("Create a Voice Assistant using Python")
- Code released under the MIT licence. The handbook text and test set were written for this project.

**Author:** Souad Essemyhij, MSc Robotics and Autonomous Systems, Universität zu Lübeck · [LinkedIn](https://www.linkedin.com/in/souad-essemyhi-742a9520b/) · [GitHub](https://github.com/souadessemyhij1999)
