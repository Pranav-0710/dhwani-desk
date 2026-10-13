# Decision: week-1 gate — can Laya read Kannada?

Status: **go — Laya stays, but only fine-tuned.** No zero-shot variant is good enough to route a call. Build on the multilingual checkpoint reading the Kannada transcript with English question text, fine-tune it (#14), take the lookup-style decisions out of Laya, and keep translate-first as the measured fallback. Measured 2026-10-13, Laya 0.4.6, laptop CPU (Intel Core Ultra 7 155H).

## What was compared

All 50 hand-labelled caller turns in `data/seed/seed.jsonl`, all 10 decisions in `data/decision_schema.json`. Reproduce with `uv run python laya_benchmark.py` in `training/` (or `01_laya_benchmark.ipynb` on Kaggle / Colab). Every prediction, with Laya's raw probabilities, is in `training/results/benchmark.json`.

| Variant | Caller text | Question text | Laya checkpoint |
| --- | --- | --- | --- |
| A | Kannada | English | multilingual |
| B | Kannada | Kannada | multilingual |
| C | English gloss | English | English |
| E | English gloss | English | multilingual |
| D | Kannada | keyword rules, no model | none |

**C and E stand in for IndicTrans2.** They read the hand-written English gloss, which is a perfect translation. Real IndicTrans2 output on Prisma transcripts can only score lower, and it adds a model and its latency to every turn. E keeps the checkpoint fixed, so A against E isolates the effect of the language. C against E isolates the effect of the checkpoint.

**D is optimistic.** The rules were written with the seed set in view, so they score near 100% here by construction. They are a reference point, not a measured competitor. Their real accuracy shows only on the unseen test set (#13).

**Small numbers.** The seed set has few positives for some flags: wants_human 3, contains_personal_id 2, out_of_scope 5, repeat_complaint 6. A difference of one or two turns swings those rows a lot.

## Results

Flags show *raised / truly positive* so false alarms are visible next to misses.

| Decision | Metric | A | B | E | C | D (fitted) |
| --- | --- | --- | --- | --- | --- | --- |
| department | macro-F1 | 0.25 | 0.28 | 0.62 | 0.63 | 1.00 |
| ward | accuracy | 0.26 | 0.34 | 0.24 | 0.36 | 1.00 |
| urgency | accuracy | 0.24 | 0.36 | 0.24 | 0.34 | 0.82 |
| caller_state | macro-F1 | 0.25 | 0.23 | 0.20 | 0.34 | 0.94 |
| safety_hazard | raised / true 15 | 4 (1 right) | 31 (11 right) | 3 (2 right) | 9 (6 right) | 16 (15 right) |
| wants_human | raised / true 3 | 3 (0 right) | 11 (0 right) | 3 (3 right) | 6 (3 right) | 3 (3 right) |
| out_of_scope | raised / true 5 | 6 (0 right) | 22 (3 right) | 0 | 8 (0 right) | 5 (5 right) |
| repeat_complaint | raised / true 6 | 0 | 19 (3 right) | 0 | 6 (4 right) | 8 (6 right) |
| has_location | raised / true 21 | 7 (5 right) | 15 (9 right) | 0 | 3 (3 right) | 22 (21 right) |
| contains_personal_id | raised / true 2 | 40 (2 right) | 25 (2 right) | 19 (1 right) | 2 (2 right) | 2 (2 right) |
| latency | s/turn median (p95) | 5.0 (8.4) | 9.4 (10.5) | 7.2 (12.4) | 15.3 (31.5) | 0.0001 |

What the numbers say:

- **Language is the bottleneck, not the checkpoint.** On department, A → E (Kannada → English, same checkpoint) goes from 0.25 to 0.62. E → C (switching to the English checkpoint) adds only 0.01. The multilingual checkpoint understands an English complaint and not a Kannada one.
- **Zero-shot Kannada picks a favourite.** A answers `other` for 33 of 50 turns and B answers `sewage-drainage` for 30 of 50. A and E answer `urgent` for all 50 turns.
- **No variant can be trusted with the flags.** B's question wording makes it say yes to everything: it would send 19 genuine complaints away as out of scope and flag 31 turns as hazards to catch 11. A and E mostly say no and miss hazards. C is the most balanced, and it still catches only 6 of 15 hazards. These flags trigger escalation, so neither failure mode is acceptable.
- **Ward and personal ID are lookups, not judgements.** B finds the right ward in 16 of the 21 turns that name one, but in the 29 turns that name none it still picks a ward 28 times. A flags a personal ID in 40 of 50 turns, though only 2 contain one. A table lookup and a digit check cannot make these mistakes.
- **Kannada question text does not help.** B against A shows a little on department and urgency, but the flags collapse into "yes", and the questions cost about 35% more tokens (1,701 against 1,260 per turn).
- **Latency cannot rank the variants on this laptop.** The same variant's median moved by up to 2× between two runs (A 8.3 → 5.0 s, B 4.5 → 9.4 s). Token count is the steadier proxy. All variants are 15–50× over the budget anyway (see below).
- The English checkpoint warns that some of its calibration temperatures are invalid, so C's confidences are uncalibrated.

Two benchmark bugs were caught before this result. Recording them so they are not repeated:

- `max_len=128` cut off the option descriptions. Laya counts them inside that limit, so department and ward fell to chance. The benchmark now uses Laya's defaults and stops if Laya reports truncation.
- Urgency was first decoded by rounding Laya's `score`, which is the expected level. That pulls every unsure answer to the middle. It is now decoded as the most likely level.

## Decision

1. **Base for the build: multilingual checkpoint, Kannada transcript in, English question text (variant A's format).**
   - English questions are shorter, easier to keep in sync with the schema, and did no worse than Kannada ones.
   - Reading Kannada directly keeps a translation model out of the live path.
2. **Fine-tuning is required, not optional.** #14 fine-tunes this format on the synthetic Kannada set (#7, #12) on Kaggle / Colab, and measures the gain on the unseen test set (#13). E shows the checkpoint can do the task when it understands the language. Fine-tuning is how it learns to understand Kannada complaints.
3. **Translate-first is the fallback, not rejected.** Zero-shot it is clearly better (department 0.62 against 0.25). #14 also runs real IndicTrans2 on the test set's Kannada and scores it through the base checkpoint, so the comparison at the kill date uses real translation output, not the perfect gloss.
4. **Lookup decisions move out of Laya** into deterministic backend code:
   - `ward`: match ward names, numbers and landmarks from `wards.json`, and answer `unknown` when nothing matches.
   - `contains_personal_id`: a digit-run check.
   - the ward/landmark part of `has_location`.

   This cuts 2–3 questions from every turn. It changes the decision schema, a shared contract, so it lands with the schema work in #15 rather than here.
5. **Keyword rules stay as a safety net** (#16). In the live path, a rule hit on `safety_hazard` or `wants_human` escalates even if Laya says no. Missing a live wire or a caller asking for a person costs more than an extra escalation.

**Kill criterion (Oct 23, start of week 3).** On the unseen test set, the fine-tuned checkpoint must beat both:
- the keyword rules
- IndicTrans2 + the base checkpoint

on department macro-F1 and safety_hazard recall. If IndicTrans2 wins, the live path translates first. If the rules win, the build routes with rules and keeps Laya only for `caller_state` and `urgency`, where rules are weakest.

## Latency: the main open problem

The PRD budgets ≤ 300 ms p95 for all Laya decisions per turn on CPU. Zero-shot on this laptop, all 10 questions take 5–15 s median. Laya scores each question as its own sequence (caller text + question + options, about 100–170 tokens), so time grows with the number of questions and the length of their option text. #15 and #22 own this:

- move the lookup decisions to code (point 4), leaving about 7 questions
- shorten option descriptions once fine-tuning carries the meaning
- serve through Laya's ONNX backend (needs the export script from the Laya repo; it is not in the pip package)
- if that is still too slow: score `safety_hazard`, `wants_human` and `out_of_scope` first, and play an acknowledgement line while the rest run
