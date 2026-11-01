# Dhwani Desk

A Kannada-first voice helpline for civic complaints. A caller speaks Kannada over a phone-quality line; **Gnani Prisma v2.5** transcribes it, **Laya** decides department, ward, urgency and whether a human must step in — locally, with no LLM in the call path — and **Gnani Timbre v2.5** speaks the reply. A supervisor console shows every decision with its confidence and lets an officer take over the call.

Built for the Great Indian AI Internship Challenge by Gnani AI.

> The city in the demo, "Nagara Seva, City Corporation of Hosapura", is fictional, and all data is synthetic.

**Write-up, results and models used: [docs/WRITEUP.md](docs/WRITEUP.md).** Demo video on [LinkedIn](https://lnkd.in/p/gNMNPkh4) and [X](https://x.com/jey_pranav53258/status/2116888213851963672).

## Results

Held-out test set (150 sentences, spoken by Timbre voices, transcribed by Prisma at 16 kHz); full report in [eval/reports/](eval/reports/test-kn-ft-v1-timbre.md).

| Metric | Target | Laya zero-shot | Laya fine-tuned |
| --- | --- | --- | --- |
| Department macro-F1 | ≥ 0.80 | 0.13 | 0.71 |
| Safety hazard recall | ≥ 0.90 | 0.00 | 0.88 |
| Wants a human, recall | ≥ 0.90 | 0.14 | 0.43 |
| Macro-F1 drop on an 8 kHz noisy line | reported | 0.00 | 0.01 |

## Run it

Copy `.env.example` to `.env`, set `GNANI_API_KEY`, `HF_TOKEN` (read access to the checkpoint) and `CONSOLE_TOKEN`, then follow [docs/runbook.md](docs/runbook.md). The phone page and the supervisor console are served on one port, and a Cloudflare quick tunnel can share them.

## How it works

- **Speech in / out:** Gnani Prisma (STT, `kn-IN`, 8 kHz telephony) and Timbre (TTS, `kn-IN`)
- **Decisions:** Laya multilingual decision engine, fine-tuned on synthetic Kannada civic complaints, running on CPU
- **Dialogue:** a deterministic state machine with Kannada reply templates, escalation to a human officer
- **Console:** live transcript, decision bars, alerts, take-over, tickets

Full design: [docs/PRD.md](docs/PRD.md).

## Team

- Jey Pranav — voice pipeline and serving
- Lakshmi Narasimman — data, fine-tuning, evaluation and console

## Contributing

Every change goes through an issue, its own branch and a reviewed pull request. See [STRUCTURE.md](STRUCTURE.md) and [AGENTS.md](AGENTS.md).

## License

[Apache 2.0](LICENSE)
