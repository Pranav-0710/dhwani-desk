# 60-second demo: shot list (#23)

Follows the script in `docs/PRD.md` ("Demo, posting and submission"). Two calls on the phone page, the supervisor console beside it, one slide, credits. Record the screen with system audio and the microphone (OBS: a Display Capture, plus Desktop Audio and Mic/Aux), 1080p, 30 fps. Keep a full take that works as the fallback (PRD risks: "Live demo fails on stage").

## Before recording

1. `.env` from `.env.example`, with `GNANI_API_KEY`, `CONSOLE_TOKEN` and a read `HF_TOKEN`. `.env.example` already runs the fine-tuned Laya (`DECIDER=laya`, `LAYA=1`, `LAYA_CHECKPOINT_REVISION=kn-ft-v1`).
2. `docker compose up --build` and wait for the backend to report healthy (`docs/runbook.md`).
3. Two browser windows side by side, at 50% zoom each:
   - left: the phone page, http://localhost:8080, line noise set to **heavy**;
   - right: the console, http://localhost:8080/supervisor, signed in with `CONSOLE_TOKEN`.
4. A headset, so the agent's voice does not reach the mic (barge-in listens while the agent speaks).
5. Rehearse both calls once; the ticket numbers go up with every call, which is fine.

## Shots

| Time | Screen | Caller says (Kannada, then what it means) | What should happen |
| --- | --- | --- | --- |
| 0–8 s | Phone page, full screen, noise on | "ನಮ್ಮ ಮನೆ ಹತ್ತಿರ ಶಾಲೆ ಮುಂದೆ ಚರಂಡಿ ತೆರೆದಿದೆ" (there's an open drain in front of the school near our house) | The greeting plays first; the caller's line shows up as Prisma's transcript |
| 8–25 s | Split: phone page and console | "ಹೂವಿನಕೇರಿ, ಸರ್ಕಾರಿ ಶಾಲೆ ಹತ್ತಿರ" (Hoovinakeri, near the government school) | The agent asks where it is, then reads back the drainage complaint in Hoovinakeri ward and the ticket number. Zoom on the console: the decision bars for the call fill in, the department chip reads drainage |
| 25–40 s | Split | "ಮಳೆ ಬಂದ ಮೇಲೆ ಕಂಬದಿಂದ ಕರೆಂಟ್ ವೈರ್ ಕೆಳಗೆ ಬಿದ್ದು ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ, ಕೆರೆಕೋಡಿ ಮೇನ್ ರೋಡ್" (after the rain a live wire fell from the pole and is sparking, Kerekodi main road) | The agent says the safety line (stay away, connecting you to an officer); the console turns the call red with the safety alert. Click **Take over**, type "ದಯವಿಟ್ಟು ಆ ವೈರ್ ಹತ್ತಿರ ಹೋಗಬೇಡಿ, ಸಿಬ್ಬಂದಿ ಬರ್ತಿದ್ದಾರೆ" (please don't go near the wire, the crew is on its way) and **Say**: it is spoken in the agent's voice |
| 40–52 s | One slide | — | The held-out results from `eval/reports/`: department macro-F1 zero-shot vs fine-tuned, safety recall, the 8 kHz + noise drop; "No LLM in the call path. ₹0 per decision." |
| 52–60 s | Credits | — | Gnani Prisma (speech to text) and Timbre (voice), Laya (decisions, fine-tuned on Kaggle), the repo link |

The caller's lines are short on purpose: one breath each, so the turn closes cleanly at the 0.6 s pause.

If the safety call is handed off on the first line, as it should be, the console shows the safety step's bars first and the rest of the decision a moment later. Keep that in shot: it is Laya answering the safety questions before anything else.

## Slide

Take every number from the newest held-out report in `eval/reports/`, never by hand. Say on the slide how the test audio was made (the report's opening line says it).

## After recording

- Trim to 60 s, add captions in English for the Kannada lines (the "means" column above).
- Export 1080p MP4; keep the raw take as the fallback.
- Post on LinkedIn and X tagging Gnani AI, with #GnaniAI #GreatIndianAIInternshipChallenge, by Nov 4 (PRD). Engagement must be organic.
