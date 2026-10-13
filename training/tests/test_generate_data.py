"""The parts of the generator that decide labels and keep them honest; no model needed."""

import random

import pytest

import generate_data as gd

SPECS = gd.sample_specs(3000, seed=7)


def allowed_values(question: str) -> set:
    spec = gd.SCHEMA[question]
    if question == "ward":
        return {ward["id"] for ward in gd.WARDS} | {"unknown"}
    if spec["type"] == "noul":
        return {True, False}
    return set(spec["criteria"])


def spec_for(text_ward: str = "unknown", form: str = "none", phrase: str | None = None) -> dict:
    """A minimal spec, with the location parts `problems` reads."""
    labels = {name: False for name in gd.SCHEMA}
    labels.update(department="water", ward=text_ward, urgency="soon", caller_state="calm")
    labels["has_location"] = form != "none"
    return {"labels": labels, "location": {"form": form, "kn": phrase}, "personal_id": None}


def test_specs_are_reproducible() -> None:
    assert gd.sample_specs(50, seed=7) == SPECS[:50]


def test_every_label_is_an_allowed_value() -> None:
    for spec in SPECS:
        assert set(spec["labels"]) == set(gd.SCHEMA)
        for question, value in spec["labels"].items():
            assert value in allowed_values(question), (spec["id"], question, value)


def test_labels_follow_the_labelling_rules() -> None:
    for spec in SPECS:
        labels, location = spec["labels"], spec["location"]
        if labels["out_of_scope"]:
            assert labels["department"] == "other"
            assert not labels["repeat_complaint"]
        assert labels["has_location"] == (location["form"] != "none")
        assert (labels["ward"] != "unknown") == (location["form"] in gd.WARD_FORM_WEIGHTS)
        assert labels["contains_personal_id"] == (spec["personal_id"] is not None)


def test_a_large_sample_covers_every_value() -> None:
    for question in gd.SCHEMA:
        used = {spec["labels"][question] for spec in SPECS}
        assert used == allowed_values(question), question
    forms = {spec["location"]["form"] for spec in SPECS}
    assert forms == {"name", "landmark", "number", "generic", "none"}


def test_scenarios_use_valid_urgency_and_cover_every_department() -> None:
    urgency = set(gd.SCHEMA["urgency"]["criteria"])
    assert set(gd.TEMPLATES["scenarios"]) == set(gd.SCHEMA["department"]["criteria"])
    pools = list(gd.TEMPLATES["scenarios"].values()) + [gd.TEMPLATES["out_of_scope"]]
    for scenario in (s for pool in pools for s in pool):
        assert scenario["urgency"] in urgency
        assert isinstance(scenario["safety_hazard"], bool)


def test_generic_places_and_fake_ids_cannot_be_mistaken_for_a_ward() -> None:
    for place in gd.TEMPLATES["generic_places"]:
        assert not gd.wards_mentioned(place["kn"]), place
        assert "ವಾರ್ಡ್" not in place["kn"]
    for number in gd.TEMPLATES["fake_id_numbers"]:
        assert gd.LONG_NUMBER.search(number["digits"])


def test_prompt_states_the_location_and_ends_with_the_json_contract() -> None:
    rng = random.Random(1)
    for spec in SPECS[:200]:
        user = gd.build_messages(spec, rng)[1]["content"]
        if spec["location"]["kn"]:
            assert f'"{spec["location"]["kn"]}"' in user
        if spec["personal_id"]:
            assert spec["personal_id"]["digits"] in user
        assert user.rstrip().endswith('"gloss": "<its meaning in English>"}')


def test_seed_sentences_pass_the_checks_for_their_own_labels() -> None:
    row = next(r for r in gd.SEED if r["id"] == "seed-001")  # names Hoovinakeri, w01
    assert gd.problems(row["text"], spec_for("w01", "name", "ಹೂವಿನಕೇರಿ")) == []


def test_checks_reject_text_that_contradicts_its_labels() -> None:
    text = "ನಮ್ಮ ಏರಿಯಾದಲ್ಲಿ ಮೂರು ದಿನದಿಂದ ನೀರು ಬಂದಿಲ್ಲ"
    assert gd.problems(text, spec_for()) == []
    assert "latin_letters" in gd.problems(text + " water", spec_for())
    assert "length" in gd.problems("ನೀರು", spec_for())
    # A landmark names its ward, so a "no ward" row cannot contain one.
    assert "ward_mentioned_but_unknown" in gd.problems(text + " ಮಸೀದಿ ಹತ್ತಿರ", spec_for())
    assert "ward_mentioned_but_unknown" in gd.problems(text + " ವಾರ್ಡ್ 4", spec_for())
    assert "ward_phrase_missing" in gd.problems(text, spec_for("w04", "name", "ಗಾಂಧಿನಗರ"))
    assert "other_ward_mentioned" in gd.problems(
        text + " ಗಾಂಧಿನಗರ ಚರ್ಚ್ ಹತ್ತಿರ", spec_for("w04", "name", "ಗಾಂಧಿನಗರ")
    )
    assert "place_phrase_missing" in gd.problems(text, spec_for("unknown", "generic", "ಬೇಕರಿ ಪಕ್ಕ"))


def test_checks_tie_long_numbers_to_the_personal_id_flag() -> None:
    text = "ಲೈಟ್ ಆಫ್ ಆಗಿದೆ ನನ್ನ ನಂಬರ್ 9876543210 ಅಪ್ಡೇಟ್ ಕೊಡಿ"
    assert "unexpected_long_number" in gd.problems(text, spec_for())
    spec = spec_for()
    spec["personal_id"] = {"kind": "phone number", "digits": "9876543210"}
    assert gd.problems(text, spec) == []
    spec["personal_id"] = {"kind": "Aadhaar number", "digits": "1234 5678 9012"}
    assert "id_number_missing" in gd.problems(text, spec)
    assert gd.problems("ಟಿಕೆಟ್ ನಂಬರ್ 4521 ಇನ್ನೂ ಲೈಟ್ ಸರಿ ಮಾಡಿಲ್ಲ", spec_for()) == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            '{"text": "ನೀರು ಬಂದಿಲ್ಲ.", "gloss": "No water."}',
            {"text": "ನೀರು ಬಂದಿಲ್ಲ", "gloss": "No water."},
        ),
        (
            '```json\n{"text": "ಲೈಟ್ ಇಲ್ಲ, ಬೇಗ!", "gloss": "No light"}\n```',
            {"text": "ಲೈಟ್ ಇಲ್ಲ ಬೇಗ", "gloss": "No light"},
        ),
        ("Sure! Here it is: ನೀರು ಬಂದಿಲ್ಲ", None),
        ('{"text": "", "gloss": "x"}', None),
        ('{"text": 3, "gloss": "x"}', None),
        ('{"text": "ನೀರು", "gloss": "water"', None),
    ],
)
def test_parse_reply(reply: str, expected: dict | None) -> None:
    assert gd.parse_reply(reply) == expected


def test_style_examples_never_show_a_ward_or_id_the_spec_does_not_have() -> None:
    rng = random.Random(3)
    for spec in SPECS[:300]:
        user = gd.build_messages(spec, rng)[1]["content"]
        examples = user.split("Style examples from other calls:")[1].split("Reply with JSON")[0]
        assert examples.count("\n- ") == 3
        if spec["labels"]["ward"] == "unknown":
            assert not gd.wards_mentioned(examples), spec["id"]
        if not spec["personal_id"]:
            assert not gd.LONG_NUMBER.search(examples), spec["id"]
