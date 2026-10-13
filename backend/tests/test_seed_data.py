"""Checks the hand-labelled data in data/ against the decision schema, so the benchmark (#6)
and training never run on a mislabelled or malformed sentence."""

import json
import re

import pytest

from config import BACKEND_DIR, REPO_ROOT

DATA = REPO_ROOT / "data"
SCHEMA = json.loads((DATA / "decision_schema.json").read_text(encoding="utf-8"))["questions"]
WARDS = json.loads((DATA / "wards.json").read_text(encoding="utf-8"))["wards"]
SEED = [
    json.loads(line)
    for line in (DATA / "seed" / "seed.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
WARD_IDS = {ward["id"] for ward in WARDS}


def allowed_values(question: str) -> set:
    spec = SCHEMA[question]
    if question == "ward":
        return WARD_IDS | {"unknown"}
    if spec["type"] == "noul":
        return {True, False}
    criteria = spec["criteria"]
    return set(criteria) if isinstance(criteria, dict) else set(criteria)


def test_seed_has_fifty_uniquely_numbered_rows() -> None:
    assert len(SEED) == 50
    assert [row["id"] for row in SEED] == [f"seed-{i:03d}" for i in range(1, 51)]


@pytest.mark.parametrize("row", SEED, ids=lambda row: row["id"])
def test_every_row_labels_every_question_with_an_allowed_value(row: dict) -> None:
    assert set(row["labels"]) == set(SCHEMA)
    for question, value in row["labels"].items():
        assert value in allowed_values(question), f"{question}={value!r}"


@pytest.mark.parametrize("row", SEED, ids=lambda row: row["id"])
def test_text_is_kannada_script_like_prisma_output(row: dict) -> None:
    # Prisma writes English words in Kannada script, so the labelled text must too.
    assert not re.search(r"[A-Za-z]", row["text"]), row["text"]
    assert row["gloss"], "every row needs an English gloss for reviewers"


@pytest.mark.parametrize("row", SEED, ids=lambda row: row["id"])
def test_labels_are_consistent_with_each_other(row: dict) -> None:
    labels = row["labels"]
    long_number = re.search(r"\d(?:[\d ]{8,})\d", row["text"])
    if long_number and len(re.sub(r"\D", "", long_number.group())) >= 10:
        assert labels["contains_personal_id"], "a 10+ digit number is a personal ID"
    if labels["ward"] != "unknown":
        assert labels["has_location"], "naming a ward is a usable location"
    if labels["out_of_scope"]:
        assert labels["department"] == "other"


def test_seed_covers_every_department_ward_urgency_and_caller_state() -> None:
    for question in ("department", "ward", "urgency", "caller_state"):
        used = {row["labels"][question] for row in SEED}
        assert used == allowed_values(question), (
            f"{question} missing {allowed_values(question) - used}"
        )
    for question in ("safety_hazard", "repeat_complaint", "wants_human", "out_of_scope"):
        assert any(row["labels"][question] for row in SEED), f"no positive {question}"
    assert any(row["labels"]["contains_personal_id"] for row in SEED)


def test_wards_are_unique_and_every_landmark_identifies_one_ward() -> None:
    assert [ward["id"] for ward in WARDS] == [f"w{i:02d}" for i in range(1, 11)]
    landmarks = [landmark["name"] for ward in WARDS for landmark in ward["landmarks"]]
    assert len(landmarks) == len(set(landmarks))


def test_every_ward_name_word_is_boosted_in_prisma() -> None:
    bias = set(json.loads((BACKEND_DIR / "gnani" / "stt_bias_words.json").read_text("utf-8")))
    for ward in WARDS:
        for word in ward["name"].split():
            assert word in bias, f"add {word} to stt_bias_words.json"
