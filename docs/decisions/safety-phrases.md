# Decision: fixed safety phrases around Laya

Status: **not used.** The list was committed and pushed before it was scored (commit `90555db`). It was then scored once on the held-out set on 2026-11-06, and `eval/compare.py` said KEEP (`eval/reports/switch-kn-ft-v1-phrases.md`). Live calls run `kn-ft-v1` without it. The code stays, so `eval/run.py --laya-phrases` can score a revised list.

## Result on the held-out set

| | `kn-ft-v1` | with phrases |
| --- | --- | --- |
| Hazards caught, 16 kHz | 30 of 34 (52 raised) | 30 of 34 (53 raised) |
| Hazards caught, 8 kHz + noise | 29 of 34 | 29 of 34 |
| Officer requests caught, 16 kHz | 3 of 7 | 3 of 7 |
| Officer requests caught, 8 kHz + noise | 2 of 7 | 3 of 7 |

It caught no extra hazards and added one false alarm. The gain on the generated rows (below) came mostly from the generator reusing the same request wordings, as feared: the test set's callers ask in other words.

## Why

`kn-ft-v1` catches 30 of 34 hazards and 3 of 7 officer requests on the held-out set, against a target of 0.90 for both, and a second fine-tune did not fix that (`eval/reports/switch-kn-ft-v2.md`). Some wordings should not depend on a model trained on noisy text: a caller asking for an officer, or reporting a spark, a bite or a fire.

## What it does

`decisions/safety_phrases.py` wraps a decider. A hit sets `safety_hazard` or `wants_human` to 1.0. It can only raise an answer, never lower one, and it changes nothing else. `revision` gains `+safety-phrases-1`, so a stored decision says the list was in use.

This changes what the project can claim: Laya no longer makes every escalation decision, because code can now raise two of them. Code already answers the ward and personal numbers, for the same reason.

## Where the list came from

- Hazards: the labelling rule in `data/README.md`, the scenarios in `data/generation/scenarios.json` and the seed set.
- Officer requests: an officer word followed within three words by an asking word. "The officer never came" must not hand a call off.
- The held-out sentences were not opened.

## What it did on data that is not the test set

853 generated rows `kn-ft-v1` never trained on, from the `v2.0` generator run:

| | Recall | Precision | False alarms |
| --- | --- | --- | --- |
| Hazard, Laya | 0.918 | 0.840 | 43 |
| Hazard, Laya or phrases | 0.927 | 0.841 | 43 |
| Officer request, Laya | 0.805 | 1.000 | 0 |
| Officer request, Laya or phrases | 0.899 | 1.000 | 0 |

These flatter the officer rule. The generator was shown example requests with these same words and was told not to mention officers in other rows, so the rows cannot test a caller who mentions an officer without asking for one. Unit tests cover those wordings instead.

## Limits

- A non-native reader wrote and checked the list. A native speaker should check it.
- A phrase hit cannot read negation: "no spark now" still counts.
- Prisma may spell a word differently from the list, and then the phrase simply does not fire.
