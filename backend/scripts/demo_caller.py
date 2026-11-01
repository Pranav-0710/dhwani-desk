"""Make the caller's side of the two demo calls as WAV files a browser can use as its microphone.

The lines come from docs/demo/script.md, spoken by a Timbre voice, with silence around them for
the greeting and the agent's replies. Chrome plays such a file as the microphone:

    chrome --use-fake-device-for-media-stream --use-fake-ui-for-media-stream ^
        --use-file-for-fake-audio-capture="<out>/call1.wav%noloop"

Usage, from backend/:
    uv run python -m scripts.demo_caller --out ../eval/out/demo

Costs a few seconds of TTS credits. The caller is a synthetic voice, so say so in the video.
"""

import argparse
import asyncio
import wave
from pathlib import Path

from config import get_settings
from gnani.timbre import TimbreClient, TimbreConfig

CALLER_VOICE = "Kavin"
RATE = 16000

# Each call: the lines, and how long to wait before each (the greeting, then the agent's reply).
CALLS: dict[str, list[tuple[float, str]]] = {
    "call1": [
        (7.0, "ಕಸದ ಗಾಡಿ ಒಂದು ವಾರದಿಂದ ಬಂದಿಲ್ಲ"),
        (10.0, "ಹೂವಿನಕೇರಿ, ಸರ್ಕಾರಿ ಶಾಲೆ ಹತ್ತಿರ"),
    ],
    "call2": [
        (
            7.0,
            "ಮಳೆ ಬಂದ ಮೇಲೆ ಕಂಬದಿಂದ ಕರೆಂಟ್ ವೈರ್ ಕೆಳಗೆ ಬಿದ್ದು ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ, ಕೆರೆಕೋಡಿ ಮೇನ್ ರೋಡ್",
        ),
    ],
}
TAIL_S = 6.0


def silence(seconds: float) -> bytes:
    return bytes(int(RATE * seconds) * 2)


async def main(out: Path, gap_scale: float) -> None:
    settings = get_settings()
    config = TimbreConfig(
        api_key=settings.gnani_api_key.get_secret_value(),
        base_url=settings.gnani_tts_url,
        voice=CALLER_VOICE,
        language=settings.gnani_tts_lang,
        sample_rate=RATE,
    )
    out.mkdir(parents=True, exist_ok=True)
    async with TimbreClient(config) as tts:
        for name, lines in CALLS.items():
            pcm = b""
            for wait_s, text in lines:
                pcm += silence(wait_s * gap_scale) + await tts.synthesize(text)
            pcm += silence(TAIL_S)
            path = out / f"{name}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(RATE)
                wav.writeframes(pcm)
            print(f"{path}  {len(pcm) / (RATE * 2):.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gap-scale", type=float, default=1.0, help="stretch every wait")
    args = parser.parse_args()
    asyncio.run(main(args.out, args.gap_scale))
