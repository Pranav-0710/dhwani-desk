# Decision: Timbre voice for the agent

Status: **decided — Saanvi** (Jey Pranav, by ear, 2026-10-13). Measured 2026-10-13.

## Candidates

Timbre v2.5 has 42 voices; only two list Kannada (`kn-IN`) as their preferred language: **Kavin** (male, professional) and **Saanvi** (female, expressive/energetic). Three other-language voices were tried speaking Kannada for comparison: Chitra (Hindi, warm/empathetic), Kaveri (English, professional) and Lavanya (Telugu, warm/empathetic).

Test script (greeting, ward and ticket number as words, and a code-mixed line in Kannada script):

> ನಮಸ್ಕಾರ, ನಗರ ಸೇವಾ ಸಹಾಯವಾಣಿಗೆ ಸ್ವಾಗತ. ನಿಮ್ಮ ದೂರು ದಾಖಲಾಗಿದೆ. ವಾರ್ಡ್ ಸಂಖ್ಯೆ ನಾಲ್ಕು, ಗಾಂಧಿ ನಗರ. ನಿಮ್ಮ ಟಿಕೆಟ್ ಸಂಖ್ಯೆ ಎರಡು ಸಾವಿರದ ಮುನ್ನೂರ ನಲವತ್ತೈದು. ನಿಮ್ಮ ಏರಿಯಾದ ಸ್ಟ್ರೀಟ್ ಲೈಟ್ ರಿಪೇರಿಗೆ ನಾಳೆ ಟೀಮ್ ಬರುತ್ತದೆ.

Regenerate the clips with `uv run python -m scripts.tts_voices` (from `backend/`); they land in `samples/tts/`.

## Results

Intelligibility check: each clip was fed back into Prisma (`kn-IN`, 16 kHz) and the transcript compared with the script.

| Voice | Native Kannada | Prisma round trip |
| --- | --- | --- |
| Kavin | Yes | Word-perfect |
| Saanvi | Yes | Word-perfect |
| Chitra | No | Word-perfect |
| Kaveri | No | Word-perfect once; one run lost the last sentence |
| Lavanya | No | One colloquial word change (ಬರುತ್ತದೆ → ಬರ್ತದೆ) |

Time to first streamed audio for a short reply (ದಯವಿಟ್ಟು ಹತ್ತಿರದ ಲ್ಯಾಂಡ್‌ಮಾರ್ಕ್ ಹೇಳಿ), 4 runs each:

| Voice | 8 kHz (ms) | 16 kHz (ms) | Median |
| --- | --- | --- | --- |
| Kavin | 812, 4672, 844, 765 | 1032, 875, 750, 546 | ~0.8 s |
| Saanvi | 2188, 812, 828, 672 | 797, 907, 765, 766 | ~0.8 s |

## Findings

- **Digits are read correctly.** "ನಿಮ್ಮ ಟಿಕೆಟ್ ಸಂಖ್ಯೆ 2345. ವಾರ್ಡ್ 4." produced the same speech as the number written in words (Prisma heard "2,000 ಮುನ್ನೂರ 45" and "4" in both cases), so templates can use digits.
- **Prisma's ITN does convert Kannada number words** (ನಾಲ್ಕು → 4) even though it leaves an English "four" as ಫೋರ್.
- **Time to first audio is ~0.8 s typical with spikes up to 4.7 s.** That alone uses half of the 1.5 s end-of-speech → first-audio budget. The dialogue layer (#9) should pre-synthesise fixed template lines at startup and cache them; only lines with ticket numbers or landmarks need live synthesis.

## Decision

**Saanvi** is the agent voice. Jey Pranav listened to all five voices, at 16 kHz and at phone-quality 8 kHz, and picked Saanvi.

- Native Kannada voice, word-perfect Prisma round trip, and time to first audio on a par with Kavin.
- The analysis above had recommended Kavin for his "professional" persona. Saanvi's "expressive" persona was the concern for serious calls; re-check her tone on the emergency and hand-off lines once the templates exist (#9, #16).
- Kavin stays the fallback. The voice is a setting (`GNANI_TTS_VOICE`), so switching is one line.
