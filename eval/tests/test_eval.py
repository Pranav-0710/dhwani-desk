"""The eval's own arithmetic: metrics, the phone line, and turning the review sheet into the set.
Run from backend/: uv run pytest ../eval/tests"""

import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_test_set as build  # noqa: E402
import line_audio  # noqa: E402
import run  # noqa: E402
from common import SEED_FILE, label_errors, read_jsonl  # noqa: E402

SEED = read_jsonl(SEED_FILE)


def test_macro_f1_averages_only_the_departments_the_set_asks_about() -> None:
    gold = ["roads", "roads", "water", "water"]
    classes = ["roads", "water", "other"]

    assert run.macro_f1(gold, gold, classes) == 1.0
    # roads: tp 1, fp 0, fn 1 -> 2/3; water: tp 2, fp 1, fn 0 -> 4/5; "other" never asked
    guess = ["roads", "water", "water", "water"]
    assert run.macro_f1(gold, guess, classes) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_flag_scores_show_misses_and_false_alarms() -> None:
    scores = run.flag_scores([True, True, False, False], [True, False, True, True])

    assert scores == {"recall": 0.5, "precision": 1 / 3, "raised": 3, "true": 2}
    assert run.flag_scores([False], [False])["recall"] is None


def test_p95_is_a_value_that_was_measured() -> None:
    assert run.p95([float(v) for v in range(1, 101)]) == 95.0
    assert run.p95([0.3]) == 0.3


def test_a_decision_is_scored_on_the_thresholds_the_call_acts_on() -> None:
    from dialogue.decision import Decision

    decision = Decision(
        department="roads", department_confidence=0.9, ward="w04", urgency="urgent",
        safety_hazard=0.6, has_location=0.51, repeat_complaint=0.2, wants_human=0.71,
        caller_state="calm", out_of_scope=0.1, contains_personal_id=0.0,
    )  # fmt: skip

    labels = run.labels_of(decision)

    assert labels["safety_hazard"] is False  # exactly at the threshold is not acted on
    assert labels["has_location"] is True
    assert labels["wants_human"] is True
    assert label_errors({k: v for k, v in labels.items() if k != run.ROUTED}) == []


def test_the_keyword_stand_in_runs_through_both_steps() -> None:
    labels, safety_s, whole_s = run.predict(run.keyword_decider(), SEED[0]["text"])

    assert label_errors({k: v for k, v in labels.items() if k != run.ROUTED}) == []
    assert 0 <= safety_s <= whole_s


def test_the_phone_line_is_8_khz_with_the_noise_of_the_phone_page() -> None:
    silence = [0.0] * 16000

    line = line_audio.phone_line(silence, 16000, "heavy", seed=1)

    assert len(line) == 8000
    assert line_audio.rms(line) == pytest.approx(line_audio.NOISE_RMS["heavy"], rel=0.1)
    assert line_audio.phone_line(silence, 16000, "heavy", seed=1) == line


def test_downsampling_keeps_speech_band_tones_and_drops_aliasing_ones() -> None:
    def tone(hz: float) -> list[float]:
        return [0.5 * math.sin(2 * math.pi * hz * n / 16000) for n in range(16000)]

    kept = line_audio.rms(line_audio.to_8k(tone(1000), 16000)[100:-100])
    dropped = line_audio.rms(line_audio.to_8k(tone(6000), 16000)[100:-100])

    assert kept == pytest.approx(0.5 / math.sqrt(2), rel=0.05)
    assert dropped < 0.02


def test_pink_noise_has_its_target_level() -> None:
    noise = line_audio.pink_noise(80_000, 0.02, random.Random(3))

    assert line_audio.rms(noise) == pytest.approx(0.02, rel=0.1)


def test_label_fixes_take_sheet_or_schema_names() -> None:
    assert build.parse_fixes("hazard=YES; urgency=urgent ;ward=w04") == {
        "safety_hazard": True, "urgency": "urgent", "ward": "w04",
    }  # fmt: skip
    assert build.parse_fixes("drop") is None
    assert build.parse_fixes("") == {}
    for bad in ("hazard", "colour=red", "officer=maybe"):
        with pytest.raises(ValueError):
            build.parse_fixes(bad)


def test_the_sheet_corrections_are_applied_and_checked() -> None:
    draft = [json.loads(json.dumps(row)) for row in SEED[:3]]
    sheet = [
        {"id": draft[0]["id"], "kannada": "ಹೊಸ ವಾಕ್ಯ", "fix": "urgency=emergency"},
        {"id": draft[1]["id"], "kannada": "", "fix": "DROP"},
        {"id": draft[2]["id"], "kannada": "", "fix": ""},
    ]

    rows, log = build.corrected(draft, sheet)

    assert [r["id"] for r in rows] == [draft[0]["id"], draft[2]["id"]]
    assert rows[0]["text"] == "ಹೊಸ ವಾಕ್ಯ"
    assert rows[0]["labels"]["urgency"] == "emergency"
    assert rows[1] == SEED[2]
    assert any("dropped" in line for line in log)

    sheet[0]["fix"] = "department=electricity"
    with pytest.raises(SystemExit, match="electricity"):
        build.corrected(draft, sheet)


def test_the_sheet_must_cover_the_draft_once() -> None:
    draft = SEED[:2]
    line = {"kannada": "", "fix": ""}

    with pytest.raises(SystemExit, match="not the sheet"):
        build.corrected(draft, [{"id": draft[0]["id"], **line}])
    with pytest.raises(SystemExit, match="twice"):
        build.corrected(draft, [{"id": r["id"], **line} for r in (*draft, draft[0])])
    with pytest.raises(SystemExit, match="not the draft"):
        build.corrected(draft, [{"id": r["id"], **line} for r in draft] + [{"id": "x", **line}])


def test_a_department_counts_only_when_the_call_would_route_it() -> None:
    from dialogue.decision import Decision

    unsure = Decision(
        department="roads", department_confidence=0.4, ward="unknown", urgency="emergency",
        safety_hazard=0.1, has_location=0.0, repeat_complaint=0.0, wants_human=0.0,
        caller_state="calm", out_of_scope=0.0, contains_personal_id=0.0,
    )  # fmt: skip
    gold = {**SEED[0], "labels": {**SEED[0]["labels"], "department": "roads"}}

    labels = run.labels_of(unsure)
    result = run.score([gold], [labels])

    assert labels["department"] == "roads"
    assert labels[run.ROUTED] == run.UNROUTED
    assert result["department"]["macro_f1"] == 1.0
    assert result["department"]["routed_macro_f1"] == 0.0
    assert result["safety_handoff"]["raised"] == 1  # an emergency hands off without a hazard


def test_spoken_conditions_are_scored_on_the_same_sentences() -> None:
    rows = SEED[:3]
    transcripts = {
        rows[0]["id"]: {"prisma_16k": "a", "prisma_8k_noise": "b"},
        rows[1]["id"]: {"prisma_16k": "c"},
    }

    by_condition = run.inputs(rows, transcripts)

    assert len(by_condition["text"]) == 3
    assert [r["id"] for r, _ in by_condition["prisma_16k"]] == [rows[0]["id"]]
    assert [r["id"] for r, _ in by_condition["prisma_8k_noise"]] == [rows[0]["id"]]


def test_the_manifest_pins_the_set_without_its_text() -> None:
    manifest = build.manifest(SEED[:5])
    text = json.dumps(manifest, ensure_ascii=False)

    assert manifest["rows"] == 5
    assert all(row["text"] not in text for row in SEED[:5])
    assert sum(manifest["label_counts"]["department"].values()) == 5
