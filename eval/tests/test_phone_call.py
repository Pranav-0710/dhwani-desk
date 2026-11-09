"""Cutting a real call's recording back into the test clips (#100), on a simulated call.
Run from backend/: uv run pytest ../eval/tests"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import line_audio  # noqa: E402
import phone_call  # noqa: E402

RATE = 8000


def fake_clips(count: int) -> list[tuple[str, list[float]]]:
    """Speech-like clips: noise under a syllable-rate envelope, a different length each."""
    rng = np.random.default_rng(4)
    clips = []
    for n in range(count):
        length = int((0.8 + 0.2 * n) * phone_call.PLAYLIST_RATE)
        t = np.arange(length) / phone_call.PLAYLIST_RATE
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t + n)
        clips.append((f"test-{n:03}", list(0.2 * envelope * rng.standard_normal(length))))
    return clips


def through_a_call(track, delay_s: float, drift: float, wipe_marker_s: float | None):
    """The playlist as the far end hears it: 8 kHz, later, a slightly different clock, noise and a
    mu-law codec. `wipe_marker_s` silences one marker, as a dropout on the line would."""
    taps = line_audio._HALF_BAND
    narrow = np.convolve(track, taps, mode="same")[::2]
    if wipe_marker_s is not None:
        start = int(wipe_marker_s * RATE)
        narrow[start : start + int(phone_call.MARKER_S * RATE)] = 0
    times = np.arange(int(len(narrow) * drift)) / drift
    stretched = np.interp(times, np.arange(len(narrow)), narrow)
    rng = np.random.default_rng(9)
    line = np.concatenate([np.zeros(int(delay_s * RATE)), stretched])
    line = line + 0.02 * rng.standard_normal(len(line))
    return np.asarray(line_audio.mu_law(list(np.clip(line, -1, 1))))


def test_every_clip_is_cut_where_it_was_played_despite_delay_drift_and_a_lost_marker() -> None:
    clips = fake_clips(6)
    track, schedule = phone_call.build_playlist(clips)
    lost = schedule["clips"][3]["marker_s"]
    recording = through_a_call(track, delay_s=1.37, drift=1.0003, wipe_marker_s=lost)

    peaks = [t for t, _ in phone_call.find_markers(recording, RATE)]
    result = phone_call.cut(recording, RATE, schedule, peaks)

    assert result["matched"] == result["markers"] - 1  # all but the wiped one
    assert result["offset_s"] == pytest.approx(1.37, abs=0.02)
    assert set(result["clips"]) == {clip_id for clip_id, _ in clips}
    for clip_id, samples in clips:
        original = np.convolve(samples, line_audio._HALF_BAND, mode="same")[::2]
        segment = result["clips"][clip_id]
        match = np.correlate(segment, original, mode="valid")
        # The clip starts CUT_BEFORE_S into its segment, give or take the line's wobble.
        assert np.argmax(np.abs(match)) / RATE == pytest.approx(phone_call.CUT_BEFORE_S, abs=0.02)


def test_speech_and_noise_alone_are_not_taken_for_markers() -> None:
    speech = np.concatenate([np.asarray(s) for _, s in fake_clips(4)])[::2]

    assert phone_call.find_markers(speech, RATE) == []


def test_the_playlist_leads_each_clip_with_a_marker_and_ends_with_one() -> None:
    clips = fake_clips(3)
    track, schedule = phone_call.build_playlist(clips)

    assert [c["id"] for c in schedule["clips"]] == [clip_id for clip_id, _ in clips]
    for entry, (_, samples) in zip(schedule["clips"], clips, strict=True):
        assert entry["clip_s"] - entry["marker_s"] == pytest.approx(
            phone_call.MARKER_S + phone_call.AFTER_MARKER_S
        )
        assert entry["length_s"] == pytest.approx(len(samples) / phone_call.PLAYLIST_RATE)
    found = [t for t, _ in phone_call.find_markers(track, phone_call.PLAYLIST_RATE)]
    expected = [c["marker_s"] for c in schedule["clips"]] + [schedule["end_marker_s"]]
    assert found == pytest.approx(expected, abs=0.002)


def test_a_new_cut_replaces_the_last_and_names_each_clip_s_sentence_version(tmp_path) -> None:
    import transcribe

    schedule = {"clips": [{"id": "t001", "sha256_16": "v1"}, {"id": "t002", "sha256_16": "v2"}]}
    phone_call.write_clips({"t001": [0.0] * 80, "t002": [0.0] * 80}, schedule, "cut-a", tmp_path)
    # The second call lost t002: its clip from the first call must not be scored again.
    phone_call.write_clips({"t001": [0.0] * 80}, schedule, "cut-b", tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["t001.json", "t001.wav"]
    assert transcribe.phone_clip("t001", "v1", tmp_path) == (tmp_path / "t001.wav", "cut-b")
    assert transcribe.phone_clip("t002", "v2", tmp_path) is None
    # A sentence re-recorded since the call: its clip says another version.
    assert transcribe.phone_clip("t001", "v9", tmp_path) is None


def test_a_phone_transcript_from_another_cut_is_done_again() -> None:
    import transcribe

    clip = (Path("t001.wav"), "cut-b")
    old = {"id": "t001", transcribe.PHONE: "x", transcribe.PHONE_CUT: "cut-a"}
    transcribe.forget_stale_phone(old, clip)
    assert transcribe.PHONE not in old and transcribe.PHONE_CUT not in old

    lost = {"id": "t001", transcribe.PHONE: "x", transcribe.PHONE_CUT: "cut-a"}
    transcribe.forget_stale_phone(lost, None)  # the new call did not carry it
    assert transcribe.PHONE not in lost

    current = {"id": "t001", transcribe.PHONE: "x", transcribe.PHONE_CUT: "cut-b"}
    transcribe.forget_stale_phone(current, clip)
    assert current[transcribe.PHONE] == "x"
