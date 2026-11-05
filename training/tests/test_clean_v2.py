import json
from pathlib import Path
from typing import Any

import pytest

import clean_v2 as cv

COMMON = "ಸರಿ"
RARE_A, RARE_B = "ಅಪರೂಪ", "ವಿರಳ"


def row(row_id: str, text: str, department: str = "water", hazard: bool = False) -> dict[str, Any]:
    labels = {flag: False for flag in cv.FLAGS}
    labels |= {"department": department, "safety_hazard": hazard}
    return {"id": row_id, "text": text, "labels": labels}


def check(sampled: list[str], good: list[str]) -> dict[str, list[str]]:
    return {
        "sampled": sampled,
        "all_three": good,
        "natural": good,
        "not_scenario": [],
        "wrong_label": [],
    }


def padding(word: str, count: int) -> list[dict[str, Any]]:
    """Rows that make `word` common across the set, so it is not rare."""
    return [row(f"pad-{word}-{i}", word) for i in range(count)]


def test_a_checked_row_follows_its_verdict_whatever_its_words() -> None:
    rows = [row("good", f"{RARE_A} {RARE_B}"), row("bad", COMMON), *padding(COMMON, 3)]
    kept, dropped = cv.clean(rows, check(["good", "bad"], ["good"]))
    assert "good" in [r["id"] for r in kept]
    assert [(r["id"], r["dropped"]) for r in dropped] == [("bad", "hand_check_failed")]


def test_an_unchecked_row_goes_only_with_more_than_one_rare_word() -> None:
    rows = [row("one", f"{RARE_A} {COMMON}"), row("two", f"{RARE_A} {RARE_B}"), *padding(COMMON, 3)]
    kept, dropped = cv.clean(rows, check([], []))
    assert "one" in [r["id"] for r in kept]
    assert [(r["id"], r["dropped"]) for r in dropped] == [("two", "rare_words")]


def test_a_word_is_rare_only_up_to_the_cut() -> None:
    twice = [row(f"t{i}", f"{RARE_A} {RARE_B}") for i in range(2)]
    thrice = [row(f"t{i}", f"{RARE_A} {RARE_B}") for i in range(3)]
    assert len(cv.clean(twice, check([], []))[1]) == 2
    assert cv.clean(thrice, check([], []))[1] == []


def test_english_and_digits_are_not_counted_as_rare_words() -> None:
    assert cv.clean([row("r", "pipe 42 burst")], check([], []))[1] == []


def test_no_sentence_or_label_is_changed() -> None:
    rows = [row("a", COMMON, hazard=True), *padding(COMMON, 3)]
    kept, _ = cv.clean(rows, check([], []))
    assert kept[0] == rows[0]


def test_shares_count_labels_and_hazards_by_department() -> None:
    rows = [row("a", COMMON, "water", True), row("b", COMMON, "water"), row("c", COMMON, "roads")]
    shares = cv.label_shares(rows)
    assert shares["n"] == 3
    assert shares["safety_hazard"] == pytest.approx(0.333)
    assert shares["dept_rows"] == {"roads": 1, "water": 2}
    assert shares["dept_hazard"] == {"roads": 0.0, "water": 0.5}


def test_the_command_writes_the_three_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_dir, out = tmp_path / "rows", tmp_path / "out"
    rows_dir.mkdir()
    rows = [row("keep", COMMON), row("drop", f"{RARE_A} {RARE_B}"), *padding(COMMON, 3)]
    (rows_dir / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    (rows_dir / "report.json").write_text('{"model": "m"}', "utf-8")
    hand_check = tmp_path / "check.json"
    hand_check.write_text(json.dumps(check([], [])), "utf-8")
    monkeypatch.setattr(cv, "HAND_CHECK", hand_check)
    monkeypatch.setattr("sys.argv", ["clean_v2.py", "--rows", str(rows_dir), "--out", str(out)])

    cv.main()

    assert len((out / "train.jsonl").read_text("utf-8").splitlines()) == 4
    assert json.loads((out / "filtered_out.jsonl").read_text("utf-8"))["dropped"] == "rare_words"
    summary = json.loads((out / "report.json").read_text("utf-8"))
    assert summary["model"] == "m"
    assert summary["filter"]["dropped"] == {"rare_words": 1}
    assert summary["filter"]["kept"] == 4
