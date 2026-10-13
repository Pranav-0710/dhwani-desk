# Data

All data here is **synthetic**: invented sentences, a fictional city (Hosapura) and fictional wards. No real people, phone numbers, ID numbers or recorded calls. The phone and Aadhaar numbers in the seed set are obviously fake placeholders.

| File | What it is |
| --- | --- |
| `decision_schema.json` | The 10 questions Laya answers per caller turn, with their options. Single source of truth for labels, the benchmark (#6), data generation (#7, #12) and the backend decision layer (#15). Changing it is a shared-contract change. |
| `wards.json` | The 10 fictional wards of Hosapura and their landmarks. Every landmark belongs to exactly one ward. Ward names are also Prisma boost words (`backend/gnani/stt_bias_words.json`). |
| `seed/seed.jsonl` | 50 hand-written, hand-labelled caller turns for the week-1 Laya Kannada benchmark (#6). |
| `generation/scenarios.json` | Templates for synthetic training data (#7): scenarios per department with their urgency and hazard labels, out-of-scope requests, place phrases that belong to no ward, and placeholder ID numbers. Used by `training/generate_data.py`; the generated rows live on the private Hugging Face dataset repo, not here. |

`backend/tests/test_seed_data.py` checks every row against the schema and the ward list in CI.

## Seed row format

```json
{"id": "seed-001", "text": "<what Prisma would write>", "gloss": "<English meaning>", "labels": {"department": "water", "ward": "w01", "urgency": "soon", "safety_hazard": false, "has_location": true, "repeat_complaint": false, "wants_human": false, "out_of_scope": false, "contains_personal_id": false, "caller_state": "calm"}}
```

## Writing rules

- **Write what Prisma would output**, not what a person would type: Kannada script only, English loanwords transliterated (ಸ್ಟ್ರೀಟ್ ಲೈಟ್, not "street light"), little punctuation, spoken colloquial forms (ಬರ್ತಿದೆ, ಆಗ್ತಿಲ್ಲ). A test rejects any Latin letter.
- Numbers: Prisma turns Kannada number words into digits (ವಾರ್ಡ್ 4) but leaves English ones as words (ವಾರ್ಡ್ ಫೋರ್). Use both.
- Keep each row to one caller turn of a few seconds.

## Labelling rules

- `ward`: a ward id only if the caller names the ward (by name or number) or a landmark unique to it; otherwise `unknown`.
- `has_location`: true if the caller gave a ward, street or landmark. Naming a ward always counts.
- `urgency`: `routine` (information, minor nuisance) → `soon` (service missing for days) → `urgent` (damage spreading, risk building) → `emergency` (someone may be hurt now).
- `safety_hazard`: immediate risk to life or health — live wires, open manholes, sewage in homes, contaminated drinking water, animal attacks, fire.
- `repeat_complaint`: the caller says it was reported before or has been pending despite complaints.
- `caller_state`: `abusive` only for insults aimed at staff; anger without insults is `frustrated`.
- `out_of_scope`: police, medical, electricity bills, ration cards, private disputes. Its `department` is always `other`.
- `contains_personal_id`: a phone, Aadhaar or account number. A ticket number is not personal.

## Seed set at a glance

| Label | Distribution |
| --- | --- |
| department | water 8 · sewage-drainage 8 · solid-waste 7 · roads 7 · streetlights 7 · stray-animals 5 · other 8 (5 of them out of scope) |
| urgency | routine 13 · soon 19 · urgent 12 · emergency 6 |
| caller_state | calm 29 · frustrated 10 · distressed 9 · abusive 2 |
| ward | named or landmarked in 21 rows; all 10 wards appear |
| flags | safety_hazard 15 · repeat_complaint 6 · wants_human 3 · out_of_scope 5 · contains_personal_id 2 |
