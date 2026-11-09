# Dhwani Desk: one-page pitch

**A Kannada voice helpline for civic complaints that hands danger to a person first.**
Jey Pranav and Lakshmi Narasimman · Great Indian AI Internship Challenge · [Repository](https://github.com/Pranav-0710/dhwani-desk)

## The problem

City helplines get thousands of calls a day about water, drains, garbage, roads and streetlights. Most callers speak their own language on an ordinary phone line. Agents can't scale, menu trees frustrate callers, and the rare emergency (a fallen live wire, a child near an open drain) waits in the same queue as a missed garbage pickup.

## The solution

A caller rings and just talks, in Kannada. **Gnani Prisma** transcribes the call, a fine-tuned **Laya** model decides what the caller needs, and **Gnani Timbre** speaks the reply.

- **Routine complaints** end with a ticket number, routed to one of seven departments and the right ward.
- **Danger goes to a person first.** Laya's safety questions run before anything else, and a hazard hands the call to an officer at once.
- **Officers stay in control.** A live console shows every call with each decision's confidence. An officer can take over and type in Kannada, and it is spoken in the agent's voice.

## Why it is different

- **No LLM in the call.** Replies come from fixed Kannada templates, so the agent never invents an answer.
- **Every decision runs locally** on a laptop CPU: ₹0 per decision, and only speech goes to the Gnani APIs.
- **Auditable.** Every ticket stores each answer's probability and the exact model revision that made it.
- **Private by design.** Spoken ID numbers are masked before storage, transcripts are deleted after 7 days, and all data is synthetic.

## Results (held-out test set: 150 calls never seen in training)

| | Laya out of the box | **Laya fine-tuned** |
| --- | --- | --- |
| Hazards caught | 0 of 34 | **30 of 34** |
| Department routing (macro-F1) | 0.13 | **0.71** |
| Cost of a noisy 8 kHz phone line | | **0.01** F1 |

The agent says "ಸರಿ" ("okay") about 1.5 s after the caller stops. Every model change must pass a rule written before scoring: no fall in safety recall, and a real gain somewhere. Three challengers (two more fine-tunes and Prisma v3.0) were rejected under it, because each caught fewer hazards.

## Built properly

FastAPI backend, React console, 618 tests in CI, Laya fine-tuned on Kaggle and pinned to a Hugging Face tag, and Docker Compose to run it. Synthetic data only; the demo city, *Nagara Seva, City Corporation of Hosapura*, is fictional.

**Next:** native-speaker training data, real voices, and a SIP phone line.
