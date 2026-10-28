import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from laya.revisions import PINNED_REVISIONS
from laya.train import target_from_expected, to_internal

import finetune_laya as ft

SEED = ft.read_jsonl(ft.REPO / "data" / "seed" / "seed.jsonl")


def test_trains_the_questions_the_backend_asks_laya() -> None:
    # backend/decisions/schema.py: SAFETY_QUESTIONS + OTHER_QUESTIONS
    backend_asks = {
        "safety_hazard", "urgency", "wants_human", "out_of_scope",
        "department", "has_location", "repeat_complaint", "caller_state",
    }  # fmt: skip

    assert set(ft.LAYA_ANSWERS) == backend_asks
    for name, question in ft.QUESTIONS.items():
        spec = ft.SCHEMA[name]
        assert question == {k: spec[k] for k in ("type", "instructions", "criteria") if k in spec}


def test_every_seed_label_becomes_a_laya_target() -> None:
    for row in SEED:
        training = ft.training_row(row)
        assert training["state"] == row["text"]
        for name, question in training["questions"].items():
            target = target_from_expected(to_internal(name, question), training["expected"][name])
            assert sum(target) == pytest.approx(1.0)
            assert max(target) == 1.0


def test_urgency_is_trained_as_its_level_index() -> None:
    row = next(r for r in SEED if r["labels"]["urgency"] == "emergency")

    assert ft.training_row(row)["expected"]["urgency"] == 3


def test_a_label_outside_the_schema_stops_the_run() -> None:
    row = json.loads(json.dumps(SEED[0]))
    row["labels"]["department"] = "electricity"

    with pytest.raises(SystemExit, match="department='electricity'"):
        ft.check_labels([row])


def test_the_base_is_layas_reviewed_commit() -> None:
    assert PINNED_REVISIONS[ft.BASE_REPO] == ft.BASE_REVISION


class FakeHub:
    def __init__(self, tags: dict[str, str]) -> None:
        self.tags = tags
        self.created: list[tuple[str, str]] = []

    def list_repo_refs(self, repo: str, repo_type: str = "model") -> SimpleNamespace:
        tags = [SimpleNamespace(name=n, target_commit=c) for n, c in self.tags.items()]
        return SimpleNamespace(tags=tags)

    def dataset_info(self, repo: str) -> SimpleNamespace:
        return SimpleNamespace(sha="head123")

    def create_tag(self, repo: str, tag: str, revision: str, **_: object) -> None:
        self.created.append((tag, revision))


def test_the_dataset_tag_pins_its_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHub({"v1.0": "abc"})
    monkeypatch.setattr(ft, "hub", lambda: fake)

    assert ft.pinned_dataset("v1.0", create=False) == "abc"
    assert fake.created == []


def test_a_missing_dataset_tag_is_made_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHub({})
    monkeypatch.setattr(ft, "hub", lambda: fake)

    with pytest.raises(SystemExit, match="--tag-dataset"):
        ft.pinned_dataset("v1.0", create=False)
    assert ft.pinned_dataset("v1.0", create=True) == "head123"
    assert fake.created == [("v1.0", "head123")]


def touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")


def test_the_upload_has_the_layout_the_backend_loads(tmp_path: Path) -> None:
    checkpoint, export = tmp_path / "ckpt", tmp_path / "export" / "ckpt"
    for name in ("model.safetensors", "rl_agent_config.json", "questions.json", "train_run.json"):
        touch(checkpoint / name)
    touch(checkpoint / "tokenizer" / "tokenizer.json")
    touch(checkpoint / "encoder" / "config.json")
    touch(checkpoint / "checkpoint_latest" / "model.safetensors")
    touch(export / "rl_agent_config.json", "exported")
    for name in ("model.onnx", "model.onnx.data", "fixtures.npz"):
        touch(export / name)

    ft.stage_upload(checkpoint, export, tmp_path / "stage")

    staged = {p.relative_to(tmp_path / "stage").as_posix() for p in (tmp_path / "stage").rglob("*")}
    assert staged == {
        "rl_agent_config.json", "model.safetensors", "questions.json", "train_run.json",
        "tokenizer", "tokenizer/tokenizer.json", "encoder", "encoder/config.json",
        "onnx", "onnx/model.onnx", "onnx/model.onnx.data",
    }  # fmt: skip
    # The config the ONNX graph was checked against, not the training copy.
    assert (tmp_path / "stage" / "rl_agent_config.json").read_text("utf-8") == "exported"


def test_a_tag_already_on_the_model_repo_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft, "hub", lambda: FakeHub({"kn-ft-v1": "old"}))

    with pytest.raises(SystemExit, match="already has 'kn-ft-v1'"):
        ft.publish(Path("unused"), "kn-ft-v1", "note")


@pytest.mark.parametrize("flag", [["--limit", "64"], ["--dry-run"], ["--no-export"]])
def test_only_a_full_run_is_published(monkeypatch: pytest.MonkeyPatch, flag: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["finetune_laya.py", "--push-tag", "kn-ft-v1", *flag])

    with pytest.raises(SystemExit) as stopped:
        ft.main()

    assert stopped.value.code == 2


def test_a_yes_no_label_must_be_a_bool() -> None:
    row = json.loads(json.dumps(SEED[0]))
    row["labels"]["safety_hazard"] = "no"

    with pytest.raises(SystemExit, match="safety_hazard='no'"):
        ft.check_labels([row])


def laya_answers(department: str, hazard: float, urgency: str) -> dict:
    levels = {str(i): 0.1 for i in range(4)}
    levels[str(ft.SCHEMA["urgency"]["criteria"].index(urgency))] = 0.7
    answers = {name: {"noul": 0.1} for name in ft.LAYA_ANSWERS}
    answers["safety_hazard"] = {"noul": hazard}
    answers["department"] = {"choice": department}
    answers["caller_state"] = {"choice": "calm"}
    answers["urgency"] = {"probabilities": levels}
    return answers


def test_answers_are_read_as_the_backend_reads_them() -> None:
    labels = ft.decode(laya_answers("roads", hazard=0.8, urgency="emergency"))

    assert labels["department"] == "roads"
    assert labels["safety_hazard"] is True
    assert labels["urgency"] == "emergency"  # the most likely level, not the expected one
    assert labels["wants_human"] is False


def test_a_model_that_answers_one_department_is_not_published() -> None:
    rows = SEED[:20]
    right = [dict(row["labels"]) for row in rows]
    collapsed = [dict(guess, department="water") for guess in right]

    assert ft.good_enough(ft.score(rows, right))
    assert not ft.good_enough(ft.score(rows, collapsed))


class PublishHub(FakeHub):
    def whoami(self) -> dict:
        return {"name": "pranav"}

    def repo_info(self, repo: str) -> None:
        if repo != ft.MODEL_REPO:
            raise AssertionError(repo)


def test_publishing_is_checked_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft, "hub", lambda: PublishHub({}))
    ft.ready_to_publish("kn-ft-v1")

    monkeypatch.setattr(ft, "hub", lambda: PublishHub({"kn-ft-v1": "old"}))
    with pytest.raises(SystemExit, match="already has"):
        ft.ready_to_publish("kn-ft-v1")


def test_a_token_that_cannot_reach_the_repo_stops_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoAccess(PublishHub):
        def repo_info(self, repo: str) -> None:
            raise PermissionError("403")

    monkeypatch.setattr(ft, "hub", lambda: NoAccess({}))

    with pytest.raises(SystemExit, match="Cannot reach"):
        ft.ready_to_publish("kn-ft-v1")


def test_the_exporter_checks_against_our_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(ft.subprocess, "run", lambda argv, **_: calls.append(argv))
    checkpoint = tmp_path / "work" / "ckpt"
    touch(checkpoint / "model.safetensors", "ours")
    (tmp_path / "laya-src").mkdir()  # already cloned

    export = ft.export_onnx(checkpoint, tmp_path / "work")

    assert export == tmp_path / "work" / "export" / "ckpt"
    # Beside the exporter's output, where it looks before the Hugging Face cache.
    assert (tmp_path / "work" / "export" / "model.safetensors").read_text("utf-8") == "ours"
    assert calls[0][-3:] == ["checkout", "-q", ft.EXPORTER_COMMIT]
    assert calls[1][1:] == [
        str(tmp_path / "laya-src" / ft.EXPORTER), str(checkpoint),
        "--out", str(tmp_path / "work" / "export"), "--verify", "--force",
    ]  # fmt: skip
