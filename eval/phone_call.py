"""Put the spoken test set through a real mobile call, and cut the far end back into clips (#100).

`playlist` joins every `out/audio/<id>.wav` into one WAV to play into a phone during a call: each
clip is led by a short chirp, the marker, so the recording at the other end can be cut back into
the same clips however the network delays or stretches it. `cut RECORDING` finds the markers in
the far end's recording, matches them to the playlist, and writes one 8 kHz clip per sentence to
`out/audio_phone/<id>.wav`, which `transcribe.py --only phone` sends to Prisma.

Usage, from backend/:
    uv run python ../eval/phone_call.py playlist
    uv run python ../eval/phone_call.py cut path/to/call-recording.m4a

`cut` decodes the recording with ffmpeg (any format the phone saves). Prints counts and timings
only, never the sentences.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import OUT
from line_audio import read_wav, write_wav

AUDIO = OUT / "audio"
WORK = OUT / "phone_call"
PHONE_AUDIO = OUT / "audio_phone"
PLAYLIST_RATE = 16000
LEAD_IN_S = 6.0  # time for the call to settle and any "this call is recorded" announcement
MARKER_S = 0.4
MARKER_HZ = (500.0, 3000.0)  # inside the phone band
MARKER_LEVEL = 0.25
AFTER_MARKER_S = 0.6
AFTER_CLIP_S = 1.2
TAIL_S = 3.0
# A peak this alike to the marker counts as one; speech stays well below it.
MIN_LIKENESS = 0.35
# How far a marker may sit from where the timing fit expects it.
FIRST_MATCH_S, REFIT_MATCH_S = 0.3, 0.15
# Cut a little before and after each clip: the network's timing wobbles by tens of ms.
CUT_BEFORE_S, CUT_AFTER_S = 0.15, 0.5
MIN_MATCHED_SHARE = 0.9


def marker(rate: int) -> Any:
    """A linear chirp across the phone band with 10 ms fades: easy to find after a codec."""
    import numpy as np

    t = np.arange(int(MARKER_S * rate)) / rate
    low, high = MARKER_HZ
    sweep = np.sin(2 * np.pi * (low * t + (high - low) * t**2 / (2 * MARKER_S)))
    fade = np.minimum(1.0, np.minimum(t, MARKER_S - t) / 0.01)
    return MARKER_LEVEL * sweep * fade


def build_playlist(clips: list[tuple[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """One track at PLAYLIST_RATE from (id, samples) clips, and where each marker and clip sits."""
    import numpy as np

    rate = PLAYLIST_RATE
    chirp = marker(rate)
    parts, schedule, at = [np.zeros(int(LEAD_IN_S * rate))], [], LEAD_IN_S
    for clip_id, samples in clips:
        clip = np.asarray(samples, dtype=float)
        clip_s = at + MARKER_S + AFTER_MARKER_S
        schedule.append(
            {"id": clip_id, "marker_s": at, "clip_s": clip_s, "length_s": len(clip) / rate}
        )
        parts += [chirp, np.zeros(int(AFTER_MARKER_S * rate)), clip]
        parts.append(np.zeros(int(AFTER_CLIP_S * rate)))
        at = clip_s + len(clip) / rate + AFTER_CLIP_S
    # A last marker, so the timing fit has both ends of the call.
    parts += [chirp, np.zeros(int(TAIL_S * rate))]
    return np.concatenate(parts), {"rate": rate, "clips": schedule, "end_marker_s": at}


def find_markers(recording: Any, rate: int) -> list[tuple[float, float]]:
    """(time, likeness) of every stretch that looks like the marker, likeness 0..1.

    Matched against the marker's analytic signal, so the envelope of the match is found whatever
    the line does to the chirp's phase; normalised by the recording's local energy, so a loud
    stretch of speech does not look like a marker."""
    import numpy as np

    recording = np.asarray(recording, dtype=float)
    chirp = marker(rate)
    size = 1 << (len(recording) + len(chirp) - 1).bit_length()
    spectrum = np.fft.fft(chirp, size)
    spectrum[size // 2 + 1 :] = 0  # analytic: only the positive frequencies, doubled
    spectrum[1 : size // 2] *= 2
    analytic = np.fft.ifft(spectrum)[: len(chirp)]
    match = np.fft.ifft(np.fft.fft(recording, size) * np.conj(np.fft.fft(analytic, size)))
    match = np.abs(match[: len(recording) - len(chirp) + 1])
    energy = np.concatenate([[0.0], np.cumsum(recording**2)])
    window = energy[len(chirp) :] - energy[: -len(chirp)]
    likeness = match / (np.sqrt(np.maximum(window, 1e-12)) * np.linalg.norm(analytic))
    likeness[window < 1e-8] = 0.0
    peaks: list[tuple[int, float]] = []
    gap = int(1.0 * rate)
    for index in np.argsort(likeness)[::-1]:
        if likeness[index] < MIN_LIKENESS:
            break
        if all(abs(index - p) >= gap for p, _ in peaks):
            peaks.append((int(index), float(likeness[index])))
    return sorted((p / rate, like) for p, like in peaks)


def align(schedule: dict[str, Any], peaks: list[float]) -> tuple[dict[int, float], float, float]:
    """Match found markers to the playlist's: {marker index: found time}, and the fit
    found = scale * playlist + offset over the matched ones (the delay and any clock drift)."""
    import numpy as np

    expected = [c["marker_s"] for c in schedule["clips"]] + [schedule["end_marker_s"]]
    found = np.asarray(peaks, dtype=float)

    def matches(scale: float, offset: float, tolerance: float) -> dict[int, float]:
        out: dict[int, float] = {}
        if not len(found):
            return out
        for k, t in enumerate(expected):
            predicted = scale * t + offset
            nearest = float(found[int(np.argmin(np.abs(found - predicted)))])
            if abs(nearest - predicted) <= tolerance:
                out[k] = nearest
        return out

    best: dict[int, float] = {}
    for p in found[:8]:
        for t in expected[:8]:
            candidate = matches(1.0, float(p - t), FIRST_MATCH_S)
            if len(candidate) > len(best):
                best = candidate
    if len(best) < 2:
        return best, 1.0, 0.0
    ks = sorted(best)
    scale, offset = np.polyfit([expected[k] for k in ks], [best[k] for k in ks], 1)
    return matches(float(scale), float(offset), REFIT_MATCH_S), float(scale), float(offset)


def cut(recording: Any, rate: int, schedule: dict[str, Any], peaks: list[float]) -> dict[str, Any]:
    """Each clip from the recording, timed from its own marker where that was found, else from
    the fit. Returns {"clips": {id: samples}, "matched": n, "markers": n, ...}."""
    matched, scale, offset = align(schedule, peaks)
    clips: dict[str, Any] = {}
    for k, entry in enumerate(schedule["clips"]):
        marker_at = matched.get(k, scale * entry["marker_s"] + offset)
        start = marker_at + scale * (entry["clip_s"] - entry["marker_s"]) - CUT_BEFORE_S
        end = start + CUT_BEFORE_S + scale * entry["length_s"] + CUT_AFTER_S
        first, last = int(start * rate), int(end * rate)
        if first >= 0 and last <= len(recording):
            clips[entry["id"]] = recording[first:last]
    markers = len(schedule["clips"]) + 1
    return {
        "clips": clips, "matched": len(matched), "markers": markers,
        "scale": scale, "offset_s": offset,
    }  # fmt: skip


def decode(path: Path, rate: int) -> Any:
    """Any audio file as mono floats at `rate`, through ffmpeg."""
    import numpy as np

    command = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate)]
    raw = subprocess.run([*command, "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(float) / 32768


def make_playlist() -> None:
    clips = []
    for path in sorted(AUDIO.glob("*.wav")):
        samples, rate = read_wav(path)
        if rate != PLAYLIST_RATE:
            raise SystemExit(f"{path.name} is {rate} Hz; the playlist is {PLAYLIST_RATE} Hz")
        clips.append((path.stem, samples))
    if not clips:
        raise SystemExit(f"No recordings in {AUDIO}")
    track, schedule = build_playlist(clips)
    WORK.mkdir(parents=True, exist_ok=True)
    write_wav(WORK / "playlist.wav", list(track), PLAYLIST_RATE)
    (WORK / "playlist.json").write_text(json.dumps(schedule, indent=1), "utf-8")
    minutes = len(track) / PLAYLIST_RATE / 60
    print(f"{len(clips)} clips, {minutes:.1f} minutes: {WORK / 'playlist.wav'}")


def cut_recording(path: Path) -> None:
    schedule = json.loads((WORK / "playlist.json").read_text("utf-8"))
    recording = decode(path, 8000)
    peaks = [t for t, _ in find_markers(recording, 8000)]
    result = cut(recording, 8000, schedule, peaks)
    share = result["matched"] / result["markers"]
    print(
        f"markers matched {result['matched']}/{result['markers']} ({len(peaks)} found); "
        f"delay {result['offset_s']:.2f} s, clock {1e6 * (result['scale'] - 1):+.0f} ppm"
    )
    if share < MIN_MATCHED_SHARE:
        raise SystemExit(
            f"Only {share:.0%} of the markers were found: check the recording is the far end of "
            "the call and covers the whole playlist. Nothing was written."
        )
    for clip_id, samples in result["clips"].items():
        write_wav(PHONE_AUDIO / f"{clip_id}.wav", list(samples), 8000)
    print(f"{len(result['clips'])}/{len(schedule['clips'])} clips written to {PHONE_AUDIO}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("playlist", help="write the playlist to play into the call")
    cut_parser = commands.add_parser("cut", help="cut the far end's recording into clips")
    cut_parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "playlist":
        make_playlist()
    else:
        cut_recording(args.recording)


if __name__ == "__main__":
    main()
