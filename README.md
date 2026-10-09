# Dhwani Desk

A Kannada-first voice helpline for civic complaints. A caller speaks Kannada over a phone-quality line; **Gnani Prisma v2.5** transcribes it, **Laya** decides department, ward, urgency and whether a human must step in — locally, with no LLM in the call path — and **Gnani Timbre v2.5** speaks the reply. A supervisor console shows every decision with its confidence and lets an officer take over the call.

Built for the Great Indian AI Internship Challenge by Gnani AI.

> Status: in development (Oct–Nov 2026). The city in the demo, "Nagara Seva, City Corporation of Hosapura", is fictional, and all data is synthetic.

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
