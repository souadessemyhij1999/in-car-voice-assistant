# 🎙️ In-Car Voice Assistant: Offline Model + Claude + Safety Layer

A bilingual (English/German) voice assistant for a car that splits the work the way production systems do:

- a **small offline model in the car** handles everyday commands in about **1 millisecond**, without a network,
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
             Offline in-car model (~1 ms)                                  ▼
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

### Offline in-car model (measured)

| Metric | Result |
|---|---|
| Commands it handles itself | **87 of 118 (74%)** |
| Correct when it handles them | **87 of 87 (100%)** |
| Overall task success (the rest needs the cloud) | 81.4% (English 82%, German 80%) |
| Unsafe actions executed | **0** (8 of 8 unsafe requests refused) |
| Median response time | **1.3 ms** on a laptop CPU |
| Cost | €0 |
| Handbook search (finding the right section) | 13 of 13 questions |

![Success by category](assets/success_by_category.png)

The offline model is strong exactly where it should be: direct commands, multi-step commands ("Navigate home and call Mom" is split and both actions run) and safety. It deliberately **passes on** indirect requests (0% offline), and about half of the handbook questions. Those are the requests the LLM is for.

![Coverage vs accuracy](assets/offline_threshold.png)

The confidence threshold trades coverage for accuracy. At 0.55, chosen before testing, the model is still 100% correct while handling 74% of commands. Lowering the threshold to handle more drops accuracy quickly (94% correct at 0.30).

### Claude and the hybrid assistant

> ⏳ **Measured results pending.** I built this project in an environment without an LLM API key, so the Claude numbers have not been measured yet, and they aren't estimated here. They are produced by [`notebooks/evaluate_llm_colab.ipynb`](notebooks/evaluate_llm_colab.ipynb) (≈10 minutes, ≈ US$0.10–0.20 of API usage), and this section will be updated with them.

What is already verified for the Claude part: the backend is tested through the **real Anthropic Python SDK** against a simulated API server that answers in the documented Messages API format ([`tests/test_claude_backend.py`](tests/test_claude_backend.py)). The tests cover:
- the request (model `claude-haiku-5-5`, the 16 functions, the system prompt and the vehicle state),
- the tool-use loop, including several functions in one turn,
- a blocked call reported back to Claude as `BLOCKED` so it can explain the refusal,
- a made-up function such as `disable_airbag` being refused,
- token and cost accounting.

---

## Key observations

- **Most of an in-car assistant doesn't need an LLM.** Three in four test commands were handled locally, correctly, in about 1 ms and at no cost. Using a cloud LLM for those would add latency, cost and a dependency on mobile coverage.
- **Knowing when not to answer is the important skill.** The offline model never acted on a request it got wrong: everything uncertain was passed on. In a car, "let me ask the cloud" is far better than opening the wrong window.
- **Safety must not depend on the model.** All 8 unsafe requests were refused at 100 km/h by the rule layer: video, boot, "switch off the airbags" and "deactivate lane assist". The same layer limits values (a request for 35 °C becomes 28 °C) and holds phone calls for confirmation. The tests show it also blocks the cloud model.
- **The test set found a bug before deployment.** "Musik pausieren" skipped to the next track, because the rule only knew the English "pause". It's fixed now; before the fix the offline model was 98.9% correct when handling a command. It's a small example of why German needs its own test cases.
- **Simple retrieval was enough for the handbook.** TF-IDF search with German keywords found the right section for all 13 questions. The offline misses in that category come from routing, not from search, so sending every question straight to the handbook would be an easy next improvement.

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

python -m pytest -q tests                                # safety rules + Claude backend tests
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
├── tests/                  # safety-layer and Claude-backend tests (9 tests)
├── notebooks/evaluate_llm_colab.ipynb   # measure the Claude and hybrid results
└── results/                # per-command details and summaries
```

---

## Limitations & next steps

- **The test set was written by me**, not recorded from real drivers, and the offline model's training phrases were generated from templates written by the same person. 8 of the 118 test commands also appear verbatim among the 914 training phrases (very short ones like "Sitzheizung aus"); without them, overall success is 80.0%.
- **Text-based evaluation.** Speech recognition errors (road noise, accents) aren't measured. Voice input and output are included as an option but weren't tested here, because the speech models couldn't be downloaded in my environment.
- **The vehicle and handbook are simulated.** The handbook is written for this project (a fictional electric car), because real manuals are copyrighted.
- **Next:** measure the Claude and hybrid results (notebook), route every question straight to the handbook, add conversation memory ("make it a bit warmer *again*"), and test with recorded in-car audio.

## Tech stack

Python · scikit-learn · Claude API (tool use, `claude-haiku-5-5`) · Anthropic Python SDK · pytest · Whisper (faster-whisper) · pyttsx3 · Pillow · matplotlib

## Credits & licences

- Claude API and SDK: [Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- Project idea from [project-based-learning](https://github.com/practical-tutorials/project-based-learning) ("Create a Voice Assistant using Python")
- Code released under the MIT licence. The handbook text and test set were written for this project.

**Author:** Souad Essemyhij, MSc Robotics and Autonomous Systems, Universität zu Lübeck · [LinkedIn](https://www.linkedin.com/in/souad-essemyhi-742a9520b/) · [GitHub](https://github.com/souadessemyhij1999)
