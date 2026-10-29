"""Read the test set aloud, one sentence at a time, into 16 kHz WAVs (#13).

Shows each sentence that has no recording yet. Press Enter to start, Enter again to stop, then
Enter to keep it, `r` to record it again, `p` to play it back, `s` to skip it, `q` to stop.
Recordings go to `eval/out/audio/<id>.wav` (private, git-ignored), each with `<id>.json` naming
the version of the sentence it was read from; run again to carry on. A sentence corrected since it
was recorded comes up again.

Usage, from backend/ (sounddevice is only needed here, so it is not a backend dependency):
    uv run --with sounddevice python ../eval/record.py --test-file ../eval/out/test.jsonl

Speak as a caller would: at your usual pace, in your usual voice, on a quiet laptop mic.
"""

import argparse
import json
import sys
from pathlib import Path

from common import OUT, fingerprint, read_jsonl
from line_audio import write_wav

RATE = 16000
AUDIO = OUT / "audio"


def capture() -> tuple[list[float], bool]:
    """The samples, and whether the sound card dropped any of them."""
    import sounddevice as sd

    frames: list[float] = []
    dropped = False

    def collect(indata, _frames, _time, status) -> None:
        nonlocal dropped
        dropped = dropped or bool(status)
        frames.extend(float(x) for x in indata[:, 0])

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=collect):
        input("  recording… Enter to stop ")
    return frames, dropped


def recorded(row: dict) -> bool:
    """A recording of this version of the sentence exists."""
    sidecar = AUDIO / f"{row['id']}.json"
    if not (AUDIO / f"{row['id']}.wav").exists() or not sidecar.exists():
        return False
    return json.loads(sidecar.read_text("utf-8")).get("sha256_16") == fingerprint(row["text"])


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
    todo = [r for r in rows if not recorded(r)]
    print(f"{len(rows) - len(todo)} of {len(rows)} recorded; {len(todo)} to go.\n")
    for number, row in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {row['id']}\n\n    {row['text']}\n")
        if record_one(row["id"], fingerprint(row["text"])) == "quit":
            break


def record_one(row_id: str, text_hash: str) -> str:
    """Record one sentence until the reader keeps or skips it; returns "kept", "skipped" or
    "quit"."""
    while True:
        start = input("  Enter to record (s skip, q stop) ").strip().lower()
        if start == "q":
            return "quit"
        if start == "s":
            return "skipped"
        samples, dropped = capture()
        if dropped:
            print("  the sound card dropped audio; again")
            continue
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
        (AUDIO / f"{row_id}.json").write_text(json.dumps({"sha256_16": text_hash}) + "\n", "utf-8")
        print("  saved\n")
        return "kept"


if __name__ == "__main__":
    main()
