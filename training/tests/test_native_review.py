"""The native-review sheet and its merge, on synthetic rows (no real training or test data)."""

import collections
from pathlib import Path
from typing import Any

import pytest

import native_review as nr

DEPARTMENTS = ("water", "roads", "solid-waste")


def row(index: int, department: str, text: str = "ನೀರು ಬರ್ತಿಲ್ಲ ವಾರ್ಡ್ 4") -> dict[str, Any]:
    labels = {
        "department": department,
        "urgency": "soon",
        "safety_hazard": False,
        "wants_human": False,
        "has_location": True,
        "contains_personal_id": False,
    }
    return {"id": f"gen-{index:05d}", "text": text, "labels": labels}


def rows(count: int = 30) -> list[dict[str, Any]]:
    return [row(i, DEPARTMENTS[i % 3]) for i in range(count)]


def marked(source: list[dict[str, Any]], **by_id: dict[str, str]) -> list[dict[str, str]]:
    sheet = nr.sheet_rows(source)
    for line in sheet:
        line.update(by_id.get(line["id"].replace("-", "_"), {}))
    return sheet


def test_the_sample_is_spread_over_the_departments_and_repeatable() -> None:
    source = [row(i, "water") for i in range(40)] + [row(100 + i, "roads") for i in range(4)]

    chosen = nr.sample(source, 10, seed=0)

    counts = collections.Counter(r["labels"]["department"] for r in chosen)
    assert counts == {"water": 6, "roads": 4}  # roads has only 4 rows: it gives all it has
    assert [r["id"] for r in chosen] == [r["id"] for r in nr.sample(source, 10, seed=0)]
    assert [r["id"] for r in chosen] != [r["id"] for r in nr.sample(source, 10, seed=1)]
    assert [r["id"] for r in chosen] == sorted(r["id"] for r in chosen)


def test_the_sheet_round_trips_kannada_through_a_csv(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    nr.write_sheet(path, nr.sheet_rows(rows(3)))

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # Excel reads the Kannada as UTF-8
    back = nr.read_sheet(path)
    assert [line["text"] for line in back] == ["ನೀರು ಬರ್ತಿಲ್ಲ ವಾರ್ಡ್ 4"] * 3
    assert back[0]["department"] == "water" and back[0]["safety_hazard"] == "no"
    assert set(back[0]) == set(nr.COLUMNS)


def test_an_unmarked_sheet_changes_nothing() -> None:
    source = rows()
    kept, dropped, summary = nr.apply(source, nr.sheet_rows(source))

    assert kept == source and dropped == []
    assert (summary["reviewed"], summary["share_natural"]) == (0, None)


def test_each_verdict_does_what_it_says() -> None:
    source = rows(6)
    sheet = nr.sheet_rows(source)
    sheet[0]["verdict"] = "ok"
    sheet[1].update(verdict="fix", corrected_text="ನೀರು ಬರುತ್ತಿಲ್ಲ ವಾರ್ಡ್ 4")
    sheet[2].update(verdict="label_wrong", note="this is a road complaint")
    sheet[3]["verdict"] = " DROP "  # case and spaces do not matter

    kept, dropped, summary = nr.apply(source, sheet)

    assert [r["id"] for r in kept] == ["gen-00000", "gen-00001", "gen-00004", "gen-00005"]
    assert kept[0]["reviewed"] == "native" and "reviewed" not in kept[2]
    assert (kept[1]["text"], kept[1]["reviewed"]) == ("ನೀರು ಬರುತ್ತಿಲ್ಲ ವಾರ್ಡ್ 4", "native-fixed")
    assert kept[1]["labels"] == source[1]["labels"]  # a fix never changes a label
    assert [(r["id"], r["dropped"]) for r in dropped] == [
        ("gen-00002", "label_wrong"),
        ("gen-00003", "drop"),
    ]
    assert dropped[0]["note"] == "this is a road complaint"
    assert summary["verdicts"] == {"ok": 1, "fix": 1, "label_wrong": 1, "drop": 1}
    assert summary["share_natural"] == 0.25
    assert (summary["kept"], summary["dropped"]) == (4, 2)


@pytest.mark.parametrize(
    ("verdict", "corrected", "why"),
    [
        ("fix", "", "needs corrected_text"),
        ("fix", "water is not coming ward 4", "no Kannada"),
        ("fix", "ನೀರು ಬರ್ತಿಲ್ಲ ವಾರ್ಡ್ 5", "changes a number"),
        ("fix", "ನೀರು ಬರ್ತಿಲ್ಲ ವಾರ್ಡ್", "changes a number"),
        ("maybe", "", "not one of"),
    ],
)
def test_a_wrongly_filled_sheet_is_refused_before_anything_is_written(
    verdict: str, corrected: str, why: str
) -> None:
    source = rows(3)
    sheet = nr.sheet_rows(source)
    sheet[1].update(verdict=verdict, corrected_text=corrected)

    with pytest.raises(SystemExit, match=why):
        nr.apply(source, sheet)


def test_a_sheet_from_other_rows_or_with_edited_text_is_refused() -> None:
    source = rows(3)
    stranger = nr.sheet_rows([row(99, "water")])
    with pytest.raises(SystemExit, match="not in the rows"):
        nr.apply(source, stranger)

    edited = nr.sheet_rows(source)
    edited[0]["text"] = "ಬದಲಾದ ಸಾಲು"
    with pytest.raises(SystemExit, match="use corrected_text"):
        nr.apply(source, edited)

    twice = nr.sheet_rows(source) + nr.sheet_rows(source[:1])
    with pytest.raises(SystemExit, match="appears twice"):
        nr.apply(source, twice)


def test_the_commands_write_a_sheet_then_the_reviewed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = rows(9)
    train = tmp_path / "train.jsonl"
    nr.write_jsonl(train, source)
    sheet = tmp_path / "review.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["native_review", "sheet", "--rows", str(train), "--n", "6", "--out", str(sheet)],
    )
    nr.main()
    filled = nr.read_sheet(sheet)
    assert len(filled) == 6
    filled[0]["verdict"], filled[1]["verdict"] = "ok", "drop"
    nr.write_sheet(tmp_path / "filled.csv", filled)

    monkeypatch.setattr(
        "sys.argv",
        [
            "native_review",
            "apply",
            "--rows",
            str(train),
            "--sheet",
            str(tmp_path / "filled.csv"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    nr.main()

    kept = nr.read_jsonl(tmp_path / "out" / "train.jsonl")
    assert len(kept) == 8 and len(nr.read_jsonl(tmp_path / "out" / "reviewed_out.jsonl")) == 1
    assert (tmp_path / "out" / "review.json").exists()
