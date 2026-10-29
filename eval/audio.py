"""The phone line the eval puts recordings through: 16 kHz speech down to 8 kHz with line noise,
the same as the phone page's noise toggle (console/src/audio/noise.ts). Plain Python: the
backend environment has no numpy, and 150 short clips do not need it."""

import array
import math
import random
import wave
from pathlib import Path

# The phone page's levels (console/src/audio/noise.ts): RMS on a -1..1 scale.
NOISE_RMS = {"light": 0.02, "heavy": 0.063}
# RMS of Paul Kellet's economy pink filter for unit white noise (noise.ts PINK_RMS).
PINK_RMS = 2.97
# Half-band low-pass before dropping every other sample: windowed sinc, cut-off at 3.6 kHz.
_TAPS = 63


def read_wav(path: Path) -> tuple[list[float], int]:
    """Mono 16-bit PCM as floats in -1..1, and the sample rate."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: needs 16-bit mono")
        samples = array.array("h", wav.readframes(wav.getnframes()))
        return [s / 32768 for s in samples], wav.getframerate()


def write_wav(path: Path, samples: list[float], rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(to_pcm16(samples))


def to_pcm16(samples: list[float]) -> bytes:
    return array.array("h", (max(-32768, min(32767, round(s * 32767))) for s in samples)).tobytes()


def _low_pass(cutoff: float) -> list[float]:
    """`cutoff` as a fraction of the sample rate; Blackman-windowed sinc, unit gain."""
    middle = (_TAPS - 1) / 2
    taps = []
    for n in range(_TAPS):
        x = n - middle
        sinc = 2 * cutoff if x == 0 else math.sin(2 * math.pi * cutoff * x) / (math.pi * x)
        window = 0.42 - 0.5 * math.cos(2 * math.pi * n / (_TAPS - 1)) + 0.08 * math.cos(
            4 * math.pi * n / (_TAPS - 1)
        )
        taps.append(sinc * window)
    total = sum(taps)
    return [t / total for t in taps]


_HALF_BAND = _low_pass(3600 / 16000)


def to_8k(samples: list[float], rate: int) -> list[float]:
    """16 kHz to 8 kHz (8 kHz passes through)."""
    if rate == 8000:
        return list(samples)
    if rate != 16000:
        raise ValueError(f"only 16 or 8 kHz, not {rate}")
    middle = len(_HALF_BAND) // 2
    out = []
    for i in range(0, len(samples), 2):
        acc = 0.0
        for k, tap in enumerate(_HALF_BAND):
            j = i + k - middle
            if 0 <= j < len(samples):
                acc += tap * samples[j]
        out.append(acc)
    return out


def pink_noise(count: int, rms: float, rng: random.Random) -> list[float]:
    """Pink noise at `rms`, with noise.ts's filter, so recordings get the phone page's line."""
    b0 = b1 = b2 = 0.0
    scale = rms / PINK_RMS
    out = []
    for _ in range(count):
        white = rng.gauss(0.0, 1.0)
        b0 = 0.99765 * b0 + white * 0.099046
        b1 = 0.963 * b1 + white * 0.2965164
        b2 = 0.57 * b2 + white * 1.0526913
        out.append((b0 + b1 + b2 + white * 0.1848) * scale)
    return out


def phone_line(samples: list[float], rate: int, level: str = "heavy", seed: int = 0) -> list[float]:
    """The clip as an 8 kHz call on a noisy line: the same noise for the same seed."""
    narrow = to_8k(samples, rate)
    noise = pink_noise(len(narrow), NOISE_RMS[level], random.Random(seed))
    return [max(-1.0, min(1.0, s + n)) for s, n in zip(narrow, noise, strict=True)]


def rms(samples: list[float]) -> float:
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
