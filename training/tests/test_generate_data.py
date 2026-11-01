"""The parts of the generator that decide labels and keep them honest; no model needed."""

import json
import random
import re

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
    assert "ward_missing" in gd.problems(text, spec_for("w04", "name", "ಗಾಂಧಿನಗರ"))
    assert "other_ward_mentioned" in gd.problems(
        text + " ಗಾಂಧಿನಗರ ಚರ್ಚ್ ಹತ್ತಿರ", spec_for("w04", "name", "ಗಾಂಧಿನಗರ")
    )
    assert "place_phrase_missing" in gd.problems(text, spec_for("unknown", "generic", "ಬೇಕರಿ ಪಕ್ಕ"))


def test_ward_numbers_are_read_whole_and_cross_checked() -> None:
    text = "ನೀರು ಬಂದಿಲ್ಲ ಮೂರು ದಿನ ಆಯ್ತು"
    w01 = spec_for("w01", "number", "ವಾರ್ಡ್ 1")
    assert gd.problems("ವಾರ್ಡ್ 1 " + text, w01) == []
    # "ವಾರ್ಡ್ 1" is a substring of "ವಾರ್ಡ್ 10"; ward 10 must not pass for ward 1.
    assert "ward_missing" in gd.problems("ವಾರ್ಡ್ 10 " + text, w01)
    assert gd.wards_mentioned("ವಾರ್ಡ್ ನಂಬರ್ 10 ಅಲ್ಲಿ") == {"w10"}
    assert gd.wards_mentioned("ವಾರ್ಡ್ ಫೋರ್ ಅಲ್ಲಿ") == {"w04"}
    assert gd.wards_mentioned("ವಾರ್ಡ್ 4ರಲ್ಲಿ") == {"w04"}
    assert gd.wards_mentioned("ವಾರ್ಡ್ 11 ಅಲ್ಲಿ") == {"ward-11"}
    w05 = spec_for("w05", "name", "ಆಲದಮರಹಳ್ಳಿ")
    assert "other_ward_mentioned" in gd.problems("ವಾರ್ಡ್ 2 ಆಲದಮರಹಳ್ಳಿ " + text, w05)


def test_places_match_across_spacing_and_word_endings() -> None:
    text = "ನೀರು ಬಂದಿಲ್ಲ ಮೂರು ದಿನ ಆಯ್ತು"
    assert gd.wards_mentioned("ಗಾಂಧಿ ನಗರ ಅಲ್ಲಿ") == {"w04"}
    assert gd.wards_mentioned("ಚರ್ಚಿನ ಹತ್ತಿರ") == {"w06"}
    assert gd.wards_mentioned("ಆಸ್ಪತ್ರೆ ಮುಂದೆ") == {"w04"}  # alias, as labelled in seed-014
    assert "ward_mentioned_but_unknown" in gd.problems("ಆಸ್ಪತ್ರೆ ಮುಂದೆ " + text, spec_for())
    assert "place_mentioned_but_none" in gd.problems("ಫಿಫ್ತ್ ಕ್ರಾಸ್ ಅಲ್ಲಿ " + text, spec_for())
    generic = spec_for("unknown", "generic", "ಬೇಕರಿ ಪಕ್ಕ")
    assert gd.problems("ಬೇಕರಿ ಪಕ್ಕದಲ್ಲಿ " + text, generic) == []


def test_text_must_be_kannada_script_only() -> None:
    text = "ನೀರು ಬಂದಿಲ್ಲ ಮೂರು ದಿನ ಆಯ್ತು"
    assert "not_kannada" in gd.problems(text + " पानी", spec_for())  # Devanagari
    assert "not_kannada" in gd.problems(text + " నీరు", spec_for())  # Telugu
    assert gd.problems(text + " 3", spec_for()) == []


@pytest.mark.parametrize("row", gd.SEED, ids=lambda row: row["id"])
def test_ward_matcher_agrees_with_every_hand_label(row: dict) -> None:
    ward = row["labels"]["ward"]
    assert gd.wards_mentioned(row["text"]) == (set() if ward == "unknown" else {ward})


def test_checks_tie_long_numbers_to_the_personal_id_flag() -> None:
    text = "ಲೈಟ್ ಆಫ್ ಆಗಿದೆ ನನ್ನ ನಂಬರ್ 1234567890 ಅಪ್ಡೇಟ್ ಕೊಡಿ"
    assert "unexpected_long_number" in gd.problems(text, spec_for())
    spec = spec_for()
    spec["personal_id"] = {"kind": "phone number", "digits": "1234567890"}
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


def test_inflection_matching_does_not_reach_other_words() -> None:
    text = "ಇನ್ನೂ ಬಂದಿಲ್ಲ ಸರ್ ಬೇಗ ಕಳಿಸಿ"
    # ನೀರಿನ ಟ್ಯಾಂಕ್ (w09's water tank) is not ನೀರಿನ ಟ್ಯಾಂಕರ್ (a water tanker).
    assert gd.wards_mentioned("ನೀರಿನ ಟ್ಯಾಂಕರ್ " + text) == set()
    assert gd.wards_mentioned("ನೀರಿನ ಟ್ಯಾಂಕು ತುಂಬಿ ಹರೀತಿದೆ") == {"w09"}
    w09 = spec_for("w09", "landmark", "ನೀರಿನ ಟ್ಯಾಂಕ್")
    assert "ward_missing" in gd.problems("ನೀರಿನ ಟ್ಯಾಂಕರ್ " + text, w09)
    # ಚರ್ಚ್ (w06's church) is not ಚರ್ಚೆ ("discussion").
    assert gd.wards_mentioned("ಮೀಟಿಂಗ್ ಅಲ್ಲಿ ಚರ್ಚೆ ಮಾಡಿದ್ವಿ") == set()
    assert gd.wards_mentioned("ಚರ್ಚ್ನಲ್ಲಿ") == {"w06"}


def test_ward_rows_need_the_asked_for_phrase() -> None:
    text = "ನೀರು ಬಂದಿಲ್ಲ ಮೂರು ದಿನ ಆಯ್ತು"
    by_name = spec_for("w04", "name", "ಗಾಂಧಿನಗರ")
    assert "ward_missing" in gd.problems("ಗಾಂಧಿ ಪ್ರತಿಮೆ ಹತ್ತಿರ " + text, by_name)
    by_number = spec_for("w04", "number", "ವಾರ್ಡ್ 4")
    assert gd.problems("4ನೇ ವಾರ್ಡ್ ಅಲ್ಲಿ " + text, by_number) == []


def test_ward_numbers_in_ordinal_and_kannada_forms() -> None:
    assert gd.wards_mentioned("ಆಲದಮರಹಳ್ಳಿ 6ನೇ ವಾರ್ಡ್") == {"w05", "w06"}
    assert gd.wards_mentioned("6 ನೇ ವಾರ್ಡ್ ಅಲ್ಲಿ") == {"w06"}
    assert gd.wards_mentioned("ಆರನೇ ವಾರ್ಡ್ ಅಲ್ಲಿ") == {"w06"}
    assert gd.wards_mentioned("ವಾರ್ಡ್ ಎರಡು ಅಲ್ಲಿ") == {"w02"}
    assert gd.wards_mentioned("ಮೂರನೇ ಸಲ ಕಾಲ್ ಮಾಡ್ತಿದೀನಿ") == set()


def test_only_street_phrases_count_as_a_location() -> None:
    text = "ನಾಯಿಗಳು ಮಕ್ಕಳನ್ನ ಅಟ್ಟಿಸ್ಕೊಂಡು ಬರ್ತವೆ"
    assert gd.problems("ರೋಡ್ ಕ್ರಾಸ್ ಮಾಡೋಕೆ ಭಯ " + text, spec_for()) == []
    for street in ("ಸೆಕೆಂಡ್ ಕ್ರಾಸ್", "3ನೇ ಕ್ರಾಸ್", "ಥರ್ಡ್ ಮೇನ್", "ಮೇನ್ ರೋಡ್ ಅಲ್ಲಿ"):
        assert "place_mentioned_but_none" in gd.problems(f"{street} {text}", spec_for()), street


def test_scenario_texts_are_unique_so_the_meaning_check_can_tell_them_apart() -> None:
    assert len(gd.ALL_SCENARIOS) == len(set(gd.ALL_SCENARIOS))


def test_meaning_check_offers_the_right_scenario_once_among_its_pool_and_distractors() -> None:
    rng = random.Random(5)
    for spec in SPECS[:300]:
        messages, options, right = gd.build_check("ನೀರು ಬಂದಿಲ್ಲ", spec, rng)
        assert options[right - 1] == spec["scenario"]
        assert options.count(spec["scenario"]) == 1
        assert options[-1] == gd.NONE_OF_THESE
        labels = spec["labels"]
        pool_name = "out_of_scope" if labels["out_of_scope"] else labels["department"]
        pool = (
            gd.TEMPLATES["out_of_scope"]
            if pool_name == "out_of_scope"
            else gd.TEMPLATES["scenarios"][pool_name]
        )
        assert len(options) == len(pool) + gd.CHECK_DISTRACTORS + 1
        user = messages[1]["content"]
        # The model must not see the request it wrote the sentence for.
        assert "ನೀರು ಬಂದಿಲ್ಲ" in user and "What happened" not in user
        assert f"{right}. {spec['scenario']}" in user


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("3", 3), (" 3.", 3), ("Option 12", 12), ("", None), ("three", None)],
)
def test_parse_choice(reply: str, expected: int | None) -> None:
    assert gd.parse_choice(reply) == expected


class FakeModel:
    """Stands in for the language model. Every sentence it writes ends in a running number,
    and its meaning check fails exactly the sentences with an odd number."""

    scenario = gd.TEMPLATES["scenarios"]["water"][0]["en"]

    def __init__(self, monkeypatch) -> None:
        self.written = 0
        monkeypatch.setattr(gd, "load_model", lambda name, four_bit: (None, None))
        monkeypatch.setattr(gd, "complete", self.complete)

    def specs(self, count: int) -> list[dict]:
        return [
            {"id": f"gen-{i:05d}", "scenario": self.scenario, **spec_for()}
            for i in range(1, count + 1)
        ]

    def complete(self, model, tokenizer, conversations, temperature, max_new_tokens):
        replies = []
        for messages in conversations:
            user = messages[-1]["content"]
            if messages[0]["content"] == gd.CHECK_SYSTEM:
                text = user.split("\n\n")[1]
                options = dict(re.findall(r"^(\d+)\. (.*)$", user, re.M))
                right = next(n for n, option in options.items() if option == self.scenario)
                replies.append(str(len(options)) if int(text.split()[-1]) % 2 else right)
            else:
                self.written += 1
                text = f"ನೀರು ಬಂದಿಲ್ಲ ಮೂರು ದಿನ ಆಯ್ತು {self.written}"
                replies.append(json.dumps({"text": text, "gloss": "x"}))
        return replies


def last_number(row: dict) -> str:
    return row["text"].split()[-1]


def test_generation_keeps_only_rows_whose_meaning_check_picks_their_scenario(monkeypatch) -> None:
    model = FakeModel(monkeypatch)
    accepted, rejected, finished = gd.generate(model.specs(6), "fake", 4, 1, 0.9, False, 0)

    assert finished
    assert [last_number(row) for row in accepted] == ["2", "4", "6"]
    assert all(row["meta"]["meaning_check"] for row in accepted)
    assert [row["reasons"] for row in rejected] == [["meaning_check_failed"]] * 3
    assert all(row["picked"] == gd.NONE_OF_THESE for row in rejected)

    accepted, rejected, _ = gd.generate(
        FakeModel(monkeypatch).specs(6), "fake", 4, 1, 0.9, False, 0, meaning_check=False
    )
    assert len(accepted) == 6 and not rejected


def test_each_spec_gets_its_own_attempts(monkeypatch) -> None:
    # Specs 1, 3, 5 write sentences 1, 3, 5 and fail; on retry they write 7, 8, 9, so spec 3
    # passes on its 2nd try; then 10, 11: spec 1 passes on its 3rd, spec 5 runs out.
    model = FakeModel(monkeypatch)
    accepted, rejected, finished = gd.generate(model.specs(6), "fake", 4, 3, 0.9, False, 0)

    assert finished
    attempts = {row["id"]: row["meta"]["attempt"] for row in accepted}
    assert attempts == {
        "gen-00001": 3,
        "gen-00002": 1,
        "gen-00003": 2,
        "gen-00004": 1,
        "gen-00006": 1,
    }
    assert [row["attempt"] for row in rejected if row["id"] == "gen-00005"] == [1, 2, 3]


def test_a_run_saves_checkpoints_and_stops_at_its_deadline(monkeypatch) -> None:
    model = FakeModel(monkeypatch)
    saved: list[tuple[int, int]] = []
    accepted, rejected, finished = gd.generate(
        model.specs(6),
        "fake",
        2,
        1,
        0.9,
        False,
        0,
        checkpoint=lambda kept, failed: saved.append((len(kept), len(failed))),
        checkpoint_every=1,
        deadline=0.0,  # already passed: stop after the first batch
    )
    assert not finished
    assert (len(accepted), len(rejected)) == (1, 1)
    assert saved == [(1, 1)]


def test_a_resumed_run_skips_kept_specs_and_counts_earlier_failures(monkeypatch) -> None:
    model = FakeModel(monkeypatch)
    model.written = 1  # continue the numbering as if sentence 1 had been written before
    kept = [{"id": "gen-00002", "text": "ನೀರು ಬಂದಿಲ್ಲ ಹಳೆಯದು", "labels": {}, "meta": {}}]
    failed = [gd.rejection({"id": "gen-00001"}, n, ["meaning_check_failed"], "") for n in (1, 2)]

    accepted, rejected, finished = gd.generate(
        model.specs(2), "fake", 4, 3, 0.9, False, 0, previous=(kept, failed)
    )

    # Only spec 1 is asked for again, and only once: it had used two of its three attempts.
    assert model.written == 2
    assert finished
    assert {row["id"] for row in accepted} == {"gen-00001", "gen-00002"}
    assert next(r for r in accepted if r["id"] == "gen-00001")["meta"]["attempt"] == 3


def test_a_checkpoint_round_trips_and_refuses_a_different_run(tmp_path) -> None:
    run = {"n": 6, "seed": 0, "model": "fake", "attempts": 3, "meaning_check": True}
    kept = [{"id": "gen-00002", "text": "ನೀರು", "labels": {}, "meta": {}}]
    failed = [gd.rejection({"id": "gen-00001"}, 1, ["unparseable"], "x")]

    assert gd.load_checkpoint(tmp_path, run, None) == ([], [])
    gd.save_checkpoint(tmp_path, run, kept, failed, None)
    assert gd.load_checkpoint(tmp_path, run, None) == (kept, failed)
    with pytest.raises(SystemExit):
        gd.load_checkpoint(tmp_path, run | {"n": 3000}, None)


class FakeHub:
    """Stands in for huggingface_hub.HfApi. A token like the Kaggle one that failed: it can
    write to an existing repo but may not create repos in the organization."""

    instances: list["FakeHub"] = []

    def __init__(self, token: str | None = None, *, exists: bool = True, upload_error=None):
        self.exists, self.upload_error = exists, upload_error
        self.created, self.uploads = False, []
        FakeHub.instances.append(self)

    def repo_info(self, repo, repo_type):
        from huggingface_hub.errors import RepositoryNotFoundError

        if not self.exists:
            # The real error wants an HTTP response to build; the type is what matters here.
            error = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
            Exception.__init__(error, "missing")
            raise error
        return type("Info", (), {"private": True})()

    def create_repo(self, repo, repo_type, private):
        if self.exists:
            raise PermissionError("403: no rights to create under the namespace")
        self.created, self.exists = True, True

    def upload_folder(self, **kwargs):
        if self.upload_error:
            raise self.upload_error
        self.uploads.append(kwargs)


def use_fake_hub(monkeypatch, **behaviour) -> None:
    import huggingface_hub

    gd.hub.cache_clear()
    FakeHub.instances.clear()
    monkeypatch.setenv("HF_TOKEN", "test")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: FakeHub(token, **behaviour))


RUN = {"n": 6, "seed": 0, "model": "fake", "attempts": 3, "meaning_check": True}


def test_an_existing_repo_is_never_created(monkeypatch, tmp_path) -> None:
    use_fake_hub(monkeypatch)
    gd.save_checkpoint(tmp_path, RUN, [], [], "Org/data", strict=True)
    hub = FakeHub.instances[0]
    assert not hub.created and hub.uploads[0]["path_in_repo"] == gd.CHECKPOINT


def test_a_missing_repo_is_created_private(monkeypatch, tmp_path) -> None:
    use_fake_hub(monkeypatch, exists=False)
    gd.save_checkpoint(tmp_path, RUN, [], [], "Org/data", strict=True)
    assert FakeHub.instances[0].created


def test_a_token_that_cannot_write_stops_the_run_before_it_starts(monkeypatch, tmp_path) -> None:
    use_fake_hub(monkeypatch, upload_error=PermissionError("403 Forbidden"))
    with pytest.raises(SystemExit) as stopped:
        gd.save_checkpoint(tmp_path, RUN, [], [], "Org/data", strict=True)
    assert "write access to Org/data" in str(stopped.value)


def test_a_failed_upload_mid_run_keeps_the_local_checkpoint_and_goes_on(
    monkeypatch, tmp_path, capsys
) -> None:
    use_fake_hub(monkeypatch, upload_error=ConnectionError("network down"))
    kept = [{"id": "gen-00002", "text": "ನೀರು", "labels": {}, "meta": {}}]
    gd.save_checkpoint(tmp_path, RUN, kept, [], "Org/data")
    assert "WARNING: checkpoint upload failed" in capsys.readouterr().out
    assert gd.load_checkpoint(tmp_path, RUN, None) == (kept, [])


def test_a_failed_final_upload_says_where_the_files_are(monkeypatch, tmp_path) -> None:
    use_fake_hub(monkeypatch, upload_error=PermissionError("403 Forbidden"))
    with pytest.raises(SystemExit) as stopped:
        gd.push(tmp_path, "Org/data", "rows")
    assert str(tmp_path) in str(stopped.value)


def test_a_token_that_cannot_see_the_repo_gets_plain_instructions(monkeypatch, tmp_path) -> None:
    # A private repo looks "not found" to a token without access, and creating it is refused.
    use_fake_hub(monkeypatch, exists=False)
    monkeypatch.setattr(
        FakeHub,
        "create_repo",
        lambda self, *a, **k: (_ for _ in ()).throw(
            PermissionError("403: no rights to create under the namespace")
        ),
    )
    with pytest.raises(SystemExit) as stopped:
        gd.load_checkpoint(tmp_path, RUN, "Org/data")
    assert "cannot see Org/data" in str(stopped.value)
    assert "write access to the repos of the Org organization" in str(stopped.value)
