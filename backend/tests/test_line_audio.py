import importlib.util
import math
import random
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "eval"
spec = importlib.util.spec_from_file_location("eval_line_audio", EVAL / "line_audio.py")
assert spec and spec.loader
line = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = line
spec.loader.exec_module(line)


def tone(seconds: float = 0.5, rate: int = 8000, level: float = 0.3) -> list[float]:
    return [level * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]


def test_the_phone_codec_keeps_speech_close_but_not_exact() -> None:
    clean = tone()
    coded = line.mu_law(clean)
    error = line.rms([a - b for a, b in zip(clean, coded, strict=True)])
    assert len(coded) == len(clean)
    assert coded != clean
    assert 0 < error < 0.1 * line.rms(clean)


def test_the_phone_codec_keeps_silence_silent_and_stays_in_range() -> None:
    assert all(abs(s) < 1e-3 for s in line.mu_law([0.0] * 100))
    assert all(-1.0 <= s <= 1.0 for s in line.mu_law([-2.0, 2.0, 0.999, -0.999]))


def test_lost_packets_are_whole_20_ms_silences_at_about_the_share_asked() -> None:
    clean = [0.5] * (8000 * 20)  # 20 s: 1,000 packets
    lost = line.drop_packets(clean, 8000, 0.03, random.Random(1))
    packets = [lost[i : i + 160] for i in range(0, len(lost), 160)]
    silent = [p for p in packets if all(s == 0.0 for s in p)]
    assert all(all(s == 0.0 for s in p) or all(s == 0.5 for s in p) for p in packets)
    assert 15 <= len(silent) <= 45


def test_lost_packets_are_the_same_for_the_same_seed() -> None:
    clean = tone(2.0)
    first = line.drop_packets(clean, 8000, 0.03, random.Random(7))
    assert first == line.drop_packets(clean, 8000, 0.03, random.Random(7))
    assert line.drop_packets(clean, 8000, 0.0, random.Random(7)) == clean


def test_loud_noise_is_as_loud_as_the_median_test_voice() -> None:
    noise = line.pink_noise(80000, line.NOISE_RMS["loud"], random.Random(3))
    assert abs(line.rms(noise) - 0.126) < 0.02
    assert line.NOISE_RMS["loud"] > line.NOISE_RMS["heavy"]
