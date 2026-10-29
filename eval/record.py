"""Read the test set aloud, one sentence at a time, into 16 kHz WAVs (#13).

Shows each sentence that has no recording yet. Press Enter to start, Enter again to stop, then
Enter to keep it, `r` to record it again, `p` to play it back, `s` to skip it, `q` to stop.
Recordings go to `eval/out/audio/<id>.wav` (private, git-ignored); run again to carry on.

Usage, from backend/ (sounddevice is only needed here, so it is not a backend dependency):
    uv run --with sounddevice python ../eval/record.py --test-file ../eval/out/test.jsonl

Speak as a caller would: at your usual pace, in your usual voice, on a quiet laptop mic.
"""

import argparse
import sys
from pathlib import Path

from common import OUT, read_jsonl
from line_audio import write_wav

RATE = 16000
AUDIO = OUT / "audio"


def capture() -> list[float]:
    import sounddevice as sd

    frames: list[float] = []

    def collect(indata, _frames, _time, _status) -> None:
        frames.extend(float(x) for x in indata[:, 0])

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=collect):
        input("  recording… Enter to stop ")
    return frames


def play(samples: list[float]) -> None:
    import sounddevice as sd

    sd.play([[s] for s in samples], RATE)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-file", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = read_jsonl(args.test_file)
    todo = [r for r in rows if not (AUDIO / f"{r['id']}.wav").exists()]
    print(f"{len(rows) - len(todo)} of {len(rows)} recorded; {len(todo)} to go.\n")
    for number, row in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {row['id']}\n\n    {row['text']}\n")
        if record_one(row["id"]) == "quit":
            break


def record_one(row_id: str) -> str:
    """Record one sentence until the reader keeps or skips it; returns "kept", "skipped" or
    "quit"."""
    while True:
        start = input("  Enter to record (s skip, q stop) ").strip().lower()
        if start == "q":
            return "quit"
        if start == "s":
            return "skipped"
        samples = capture()
        seconds = len(samples) / RATE
        answer = (
            input(f"  {seconds:.1f} s. Enter keeps it; r again, p play, s skip ").strip().lower()
        )
        while answer == "p":
            play(samples)
            answer = input("  Enter keeps it; r again, s skip ").strip().lower()
        if answer == "r":
            continue
        if answer == "s":
            return "skipped"
        if seconds < 0.5:
            print("  too short; again")
            continue
        write_wav(AUDIO / f"{row_id}.wav", samples, RATE)
        print("  saved\n")
        return "kept"


if __name__ == "__main__":
    main()
