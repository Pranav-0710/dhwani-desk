# Decision: week-1 gate — can Laya read Kannada?

Status: **go — Laya stays, but not zero-shot.** Build on the multilingual checkpoint with Kannada question text (variant B), fine-tune it (#14), and take the lookup-style decisions out of Laya. Measured 2026-10-13, Laya 0.4.6, laptop CPU (Intel Core Ultra 7 155H).

## What was compared

All 50 hand-labelled caller turns in `data/seed/seed.jsonl`, all 10 decisions in `data/decision_schema.json`. Reproduce with `uv run python laya_benchmark.py` in `training/` (or `01_laya_benchmark.ipynb` on Kaggle / Colab); every prediction is in `training/results/benchmark.json`.

| Variant | Caller text | Question text | Laya checkpoint |
| --- | --- | --- | --- |
| A | Kannada | English | multilingual |
| B | Kannada | Kannada | multilingual |
| C | English gloss | English | English |
| D | Kannada | — keyword rules, no model | — |

**C is a ceiling, not IndicTrans2.** It feeds Laya the hand-written English gloss, i.e. a perfect translation. Real IndicTrans2 output can only score lower and adds its own latency, so if the ceiling does not win, translate-first does not either. It did not win (see below), so IndicTrans2 itself was not run.

**D is optimistic.** The rules were written with the seed set in view, so they score near 100% here by construction. They are a reference point, not a measured competitor; their real accuracy shows only on the unseen test set (#13).

## Results

| Decision | Metric | A | B | C (ceiling) | D (fitted) |
| --- | --- | --- | --- | --- | --- |
| department | macro-F1 | 0.25 | 0.28 | **0.63** | 1.00 |
| ward | accuracy | 0.26 | 0.34 | 0.36 | 1.00 |
| urgency | accuracy | 0.24 | 0.26 | 0.38 | 0.82 |
| urgency | within one level | 0.74 | 0.70 | 0.92 | 1.00 |
| safety_hazard | recall | 0.07 | **0.73** | 0.40 | 1.00 |
| has_location | recall | 0.24 | 0.43 | 0.14 | 1.00 |
| repeat_complaint | recall | 0.00 | 0.50 | 0.67 | 1.00 |
| wants_human | recall | 0.00 | 0.00 | 1.00 | 1.00 |
| caller_state | macro-F1 | 0.25 | 0.23 | 0.34 | 0.94 |
| out_of_scope | recall | 0.00 | 0.60 | 0.00 | 1.00 |
| contains_personal_id | accuracy | 0.24 | 0.54 | 1.00 | 1.00 |
| **latency** | s/turn, median (p95) | 8.3 (9.3) | 4.5 (9.7) | 15.9 (17.8) | 0.0001 |

Where the errors come from:

- **Department collapses to one answer.** A answers `other` for 33 of 50 turns, B answers `sewage-drainage` for 30 of 50. Zero-shot Laya is not reading the Kannada complaint, it is picking a favourite option.
- **Ward: good at names, never says "unknown".** B finds the right ward in 16 of the 21 turns that name one, but for the 29 turns that name none it still picks a ward 28 times. A ward lookup against `wards.json` cannot make that mistake.
- **Safety flags are unreliable zero-shot.** B catches 73% of hazards but no `wants_human` request at all; A catches almost nothing. These two flags trigger escalation, so a miss is the worst error the system can make.
- The English checkpoint warns that some of its calibration temperatures are invalid, so C's confidences are uncalibrated.

A setup bug worth recording: an earlier run passed `max_len=128` to save CPU time. Laya counts the option descriptions inside that limit, so the department and ward options were cut off and both decisions fell to chance. The benchmark now uses Laya's default lengths and stops if Laya reports truncation.

## Decision

1. **Laya, multilingual checkpoint, Kannada question text (B)** is the base for the build. It is the best zero-shot Kannada variant on the decisions that matter for escalation (safety_hazard, out_of_scope) and the fastest.
2. **Fine-tuning is required, not optional.** No zero-shot variant is usable for routing. #14 fine-tunes B on the synthetic Kannada set (#7, #12) on Kaggle / Colab; the gain is measured on the unseen test set (#13).
3. **Lookup decisions move out of Laya** into deterministic code in the backend: `ward` (match ward names, numbers and landmarks from `wards.json`), `contains_personal_id` (digit-run check) and the ward part of `has_location`. That leaves Laya the decisions that need understanding and cuts 2–3 questions from every turn. This changes the decision schema, a shared contract, so it lands with the schema work in #15 rather than here.
4. **Keyword rules stay as a safety net** (#16). In the live path, a rule hit on `safety_hazard` or `wants_human` escalates even if Laya says no. Missing a live wire or a caller asking for a person costs more than an extra escalation.
5. **Translate-first is rejected.** Even with a perfect translation it reaches only 0.63 department macro-F1 zero-shot, and the English checkpoint is the slowest at 16 s per turn before any translation time.

**Kill criterion.** If the fine-tuned checkpoint does not beat the keyword rules on the unseen test set's department macro-F1 and safety recall by Oct 23 (start of week 3), the build routes with rules and keeps Laya only for `caller_state` and `urgency`, where rules are weakest.

## Latency: the main open problem

The PRD budgets 300 ms for all Laya decisions per turn. Zero-shot on this CPU, all 10 questions take 4.5–8 s median. Laya scores each question as its own sequence (caller text + question + options, ~100–170 tokens), so time grows with the number of questions and the length of their option text. #15 and #22 own this:

- move the lookup decisions to code (point 3), leaving ~7 questions
- shorten option descriptions once fine-tuning carries the meaning
- serve through Laya's ONNX backend (needs the export script from the Laya repo; not in the pip package)
- if that is still too slow: score `safety_hazard`, `wants_human` and `out_of_scope` first and play an acknowledgement line while the rest run
