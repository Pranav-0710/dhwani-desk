"""The agent's Kannada lines (templates/kn.yaml) and the names it reads back."""

import json
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

from config import REPO_ROOT

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
WARDS_FILE = REPO_ROOT / "data" / "wards.json"
SCHEMA_FILE = REPO_ROOT / "data" / "decision_schema.json"

# Every line the state machine can say; loading fails if one is missing.
LINE_KEYS = frozenset(
    {
        "greeting",
        "ask_problem",
        "ask_location",
        "ask_ward",
        "apology",
        "repeat_noted",
        "read_back_ward",
        "read_back",
        "closing",
        "safety",
        "handoff_human",
        "handoff_calm",
        "handoff_unclear",
        "redirect",
    }
)


@dataclass(frozen=True)
class Templates:
    lines: dict[str, str]
    departments: dict[str, str]  # department id -> how a read-back line names it
    wards: dict[str, str]  # ward id -> ward name in Kannada

    def line(self, key: str, **values: str) -> str:
        return self.lines[key].format(**values)

    def fixed_lines(self) -> list[str]:
        """Lines with no placeholders: the same audio every time, so worth caching."""
        return [text for text in self.lines.values() if not placeholders(text)]


def placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def load_templates(language: str = "kn") -> Templates:
    data = yaml.safe_load((TEMPLATE_DIR / f"{language}.yaml").read_text("utf-8"))
    lines, departments = data["lines"], data["departments"]
    if missing := LINE_KEYS - lines.keys():
        raise ValueError(f"{language}.yaml is missing lines: {sorted(missing)}")
    schema = json.loads(SCHEMA_FILE.read_text("utf-8"))["questions"]
    if missing := schema["department"]["criteria"].keys() - departments.keys():
        raise ValueError(f"{language}.yaml is missing department names: {sorted(missing)}")
    wards = json.loads(WARDS_FILE.read_text("utf-8"))["wards"]
    return Templates(
        lines=lines,
        departments=departments,
        wards={ward["id"]: ward["name"] for ward in wards},
    )
