"""Week-1 gate (#6): can Laya read Kannada caller turns well enough to build on?

Runs Laya zero-shot on data/seed/seed.jsonl in four variants and scores every decision in
data/decision_schema.json against the hand labels:

  A  kn-state/en-questions   Kannada text, English questions (multilingual checkpoint)
  B  kn-state/kn-questions   Kannada text, Kannada questions (multilingual checkpoint)
  C  en-gloss/en-questions   the English gloss, English questions (English checkpoint):
                             the ceiling for a translate-first pipeline with perfect MT
  D  keyword-rules           hand-written Kannada keyword rules, no model

Usage (CPU is fine; ~30 minutes for all variants):
    uv run python laya_benchmark.py                     # all variants
    uv run python laya_benchmark.py --variants A D --limit 10
    python laya_benchmark.py --device cuda                # Kaggle / Colab GPU

Writes results/benchmark.json and prints a markdown table.
"""

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = Path(__file__).resolve().parent / "results"

SCHEMA: dict[str, Any] = json.loads((DATA / "decision_schema.json").read_text("utf-8"))["questions"]
WARDS: list[dict[str, Any]] = json.loads((DATA / "wards.json").read_text("utf-8"))["wards"]
URGENCY = SCHEMA["urgency"]["criteria"]
NOUL = [name for name, spec in SCHEMA.items() if spec["type"] == "noul"]

# Kannada wording of every question, for variant B.
KANNADA_QUESTIONS: dict[str, dict[str, Any]] = {
    "department": {
        "instructions": "ಈ ದೂರನ್ನು ನಗರಸಭೆಯ ಯಾವ ವಿಭಾಗ ನೋಡಿಕೊಳ್ಳಬೇಕು?",
        "criteria": {
            "water": "ಕುಡಿಯುವ ನೀರು: ನೀರು ಬರದಿರುವುದು, ಕಡಿಮೆ ಒತ್ತಡ, ಒಡೆದ ಪೈಪ್, ಕಲುಷಿತ ನೀರು, ನೀರಿನ ಟ್ಯಾಂಕರ್",
            "sewage-drainage": "ಒಳಚರಂಡಿ, ಚರಂಡಿ ಬ್ಲಾಕ್ ಅಥವಾ ಉಕ್ಕಿ ಹರಿಯುವುದು, ತೆರೆದ ಮ್ಯಾನ್ ಹೋಲ್, "
            "ಒಡೆದ ಚರಂಡಿ ಸ್ಲ್ಯಾಬ್, ರಸ್ತೆಯಲ್ಲಿ ಮಳೆ ನೀರು ನಿಲ್ಲುವುದು",
            "solid-waste": "ಕಸ ಸಂಗ್ರಹಣೆ, ಕಸದ ರಾಶಿ, ತುಂಬಿದ ಡಸ್ಟ್ ಬಿನ್, ಸುರಿದ ಕಟ್ಟಡ ತ್ಯಾಜ್ಯ, "
            "ಕಸಕ್ಕೆ ಬೆಂಕಿ, ರಸ್ತೆಯಲ್ಲಿ ಸತ್ತ ಪ್ರಾಣಿ",
            "roads": "ರಸ್ತೆ ಗುಂಡಿ, ಹಾಳಾದ ಅಥವಾ ಅಗೆದ ರಸ್ತೆ, ಫುಟ್ ಪಾತ್ ಒತ್ತುವರಿ, ಸ್ಪೀಡ್ ಬ್ರೇಕರ್, ಮರ ಬಿದ್ದು ರಸ್ತೆ ಬ್ಲಾಕ್",
            "streetlights": "ಬೀದಿ ದೀಪ ಹತ್ತದಿರುವುದು ಅಥವಾ ಹಗಲಲ್ಲಿ ಉರಿಯುವುದು, ಕಂಬದಿಂದ ಬಿದ್ದ ವೈರ್, ಬಾಗಿದ ಕಂಬ",
            "stray-animals": "ಬೀದಿ ನಾಯಿ, ನಾಯಿ ಕಡಿತ, ರಸ್ತೆಯಲ್ಲಿ ದನಗಳು, ಮಂಗಗಳು, ಹಾವು",
            "other": "ಬೇರೆ ನಾಗರಿಕ ವಿಷಯ (ಪಾರ್ಕ್, ಸೊಳ್ಳೆ ಔಷಧಿ, ಕಟ್ಟಡ ಅನುಮತಿ) ಅಥವಾ ನಾಗರಿಕ ದೂರು ಅಲ್ಲದ್ದು",
        },
    },
    "ward": {"instructions": "ಈ ದೂರು ಹೊಸಪುರದ ಯಾವ ವಾರ್ಡ್ ಬಗ್ಗೆ?"},
    "urgency": {
        "instructions": "ಈ ದೂರು ಎಷ್ಟು ತುರ್ತು?",
        "criteria": ["ಸಾಮಾನ್ಯ", "ಬೇಗ ಬೇಕು", "ತುರ್ತು", "ಅತಿ ತುರ್ತು"],
    },
    "safety_hazard": {
        "instructions": "ಯಾರದ್ದಾದರೂ ಜೀವಕ್ಕೆ ಅಥವಾ ಆರೋಗ್ಯಕ್ಕೆ ತಕ್ಷಣದ ಅಪಾಯ ಇದೆಯೇ? ಉದಾಹರಣೆಗೆ ಕರೆಂಟ್ ವೈರ್, "
        "ತೆರೆದ ಮ್ಯಾನ್ ಹೋಲ್, ಮನೆಗೆ ಕೊಳಚೆ ನೀರು, ಕಲುಷಿತ ಕುಡಿಯುವ ನೀರು, ಪ್ರಾಣಿ ದಾಳಿ"
    },
    "has_location": {"instructions": "ಕರೆ ಮಾಡಿದವರು ವಾರ್ಡ್, ಬೀದಿ ಅಥವಾ ಗುರುತಿನ ಸ್ಥಳ ಹೇಳಿದ್ದಾರೆಯೇ?"},
    "repeat_complaint": {
        "instructions": "ಈ ಸಮಸ್ಯೆ ಬಗ್ಗೆ ಮೊದಲೇ ದೂರು ಕೊಟ್ಟಿದ್ದೇವೆ ಅಥವಾ ಬಹಳ ದಿನದಿಂದ ಬಾಕಿ ಇದೆ ಅಂತ ಹೇಳ್ತಿದ್ದಾರೆಯೇ?"
    },
    "wants_human": {"instructions": "ಸ್ವಯಂಚಾಲಿತ ವ್ಯವಸ್ಥೆ ಬದಲು ಅಧಿಕಾರಿಯ ಜೊತೆ ಮಾತಾಡಬೇಕು ಅಂತ ಕೇಳ್ತಿದ್ದಾರೆಯೇ?"},
    "caller_state": {
        "instructions": "ಕರೆ ಮಾಡಿದವರ ಮನಸ್ಥಿತಿ ಹೇಗಿದೆ?",
        "criteria": {
            "calm": "ಶಾಂತವಾಗಿ ವಿಷಯ ಹೇಳ್ತಿದ್ದಾರೆ",
            "frustrated": "ತಡ ಆಗಿದ್ದಕ್ಕೆ ಬೇಸರ ಅಥವಾ ಸಿಟ್ಟು, ಆದರೆ ಬೈಯುತ್ತಿಲ್ಲ",
            "distressed": "ಯಾರಿಗಾದರೂ ಅಪಾಯ ಆಗಬಹುದು ಅಂತ ಭಯ ಅಥವಾ ಗಾಬರಿ",
            "abusive": "ಸಿಬ್ಬಂದಿಯನ್ನು ಬೈಯುತ್ತಿದ್ದಾರೆ ಅಥವಾ ಅವಮಾನಿಸುತ್ತಿದ್ದಾರೆ",
        },
    },
    "out_of_scope": {
        "instructions": "ಇದು ನಗರಸಭೆ ದೂರು ಅಲ್ಲವೇ? ಉದಾಹರಣೆಗೆ ಪೊಲೀಸ್, ವೈದ್ಯಕೀಯ ತುರ್ತು, ಕರೆಂಟ್ ಬಿಲ್, "
        "ರೇಷನ್ ಕಾರ್ಡ್, ಅಕ್ಕಪಕ್ಕದವರ ಜಗಳ"
    },
    "contains_personal_id": {
        "instructions": "ಫೋನ್ ನಂಬರ್, ಆಧಾರ್ ನಂಬರ್, ಖಾತೆ ನಂಬರ್ ಅಥವಾ ಬೇರೆ ವೈಯಕ್ತಿಕ ಗುರುತಿನ ಸಂಖ್ಯೆ ಹೇಳಿದ್ದಾರೆಯೇ?"
    },
}


def ward_criteria(kannada: bool) -> dict[str, str]:
    criteria = {}
    for ward in WARDS:
        if kannada:
            landmarks = ", ".join(landmark["name"] for landmark in ward["landmarks"])
            criteria[ward["id"]] = f"{ward['name']}, ವಾರ್ಡ್ {ward['number']}: {landmarks}"
        else:
            landmarks = ", ".join(
                f"{landmark['name_en']} ({landmark['name']})" for landmark in ward["landmarks"]
            )
            criteria[ward["id"]] = (
                f"{ward['name_en']} ({ward['name']}), ward {ward['number']}: {landmarks}"
            )
    criteria["unknown"] = (
        "ಯಾವ ವಾರ್ಡ್, ಬೀದಿ ಅಥವಾ ಗುರುತು ಹೇಳಿಲ್ಲ" if kannada else "no ward, street or landmark mentioned"
    )
    return criteria


def laya_questions(kannada: bool) -> dict[str, dict[str, Any]]:
    questions = {}
    for name, spec in SCHEMA.items():
        source = KANNADA_QUESTIONS[name] if kannada else spec
        question = {"type": spec["type"], "instructions": source["instructions"]}
        if name == "ward":
            question["criteria"] = ward_criteria(kannada)
        elif "criteria" in spec:
            question["criteria"] = source["criteria"]
        questions[name] = question
    return questions


def decode(answers: dict[str, Any]) -> dict[str, Any]:
    """Laya answers -> the same label space as the seed file."""
    labels: dict[str, Any] = {}
    for name, spec in SCHEMA.items():
        answer = answers[name]
        if spec["type"] == "noul":
            labels[name] = answer["noul"] >= 0.5
        elif spec["type"] == "score":
            level = min(len(URGENCY) - 1, max(0, round(answer["score"])))
            labels[name] = URGENCY[level]
        else:
            labels[name] = answer["choice"]
    return labels


# --- variant D: keyword rules ------------------------------------------------------------

DEPARTMENT_WORDS = {
    "water": ["ನೀರು ಬಂದಿಲ್ಲ", "ಕುಡಿಯೋ ನೀರ", "ಪೈಪ್", "ಪ್ರೆಶರ್", "ಟ್ಯಾಂಕರ್", "ನಲ್ಲಿ", "ನೀರಿನ ಸಮಸ್ಯೆ", "ನೀರು ಬರ್ತಿಲ್ಲ"],
    "sewage-drainage": ["ಚರಂಡಿ", "ಮ್ಯಾನ್ ಹೋಲ್", "ಡ್ರೈನೇಜ್", "ಯುಜಿಡಿ", "ಕೊಳಚೆ", "ನೀರು ನಿಲ್ಲುತ್ತೆ"],
    "solid-waste": ["ಕಸ", "ಡಸ್ಟ್ ಬಿನ್", "ಡೆಬ್ರಿಸ್", "ಸತ್ತ"],
    "roads": ["ಗುಂಡಿ", "ರೋಡ್ ರಿಪೇರಿ", "ಫುಟ್ ಪಾತ್", "ಸ್ಪೀಡ್ ಬ್ರೇಕರ್", "ಮರ ಬಿದ್ದು", "ರೋಡ್ ತುಂಬಾ ಹಾಳಾಗಿದೆ"],
    "streetlights": ["ಲೈಟ್", "ಕಂಬ", "ವೈರ್"],
    "stray-animals": ["ನಾಯಿ", "ದನ", "ಮಂಗ", "ಹಾವು"],
}
OUT_OF_SCOPE_WORDS = ["ಜಗಳ", "ಹೊಡೆದಾಡ", "ಎದೆ ನೋವು", "ಆಂಬುಲೆನ್ಸ್ ಬೇಕು", "ಕರೆಂಟ್ ಬಿಲ್", "ರೇಷನ್", "ಪಕ್ಕದ ಮನೆಯವರು"]
HAZARD_WORDS = [
    "ಮ್ಯಾನ್ ಹೋಲ್",
    "ಸ್ಪಾರ್ಕ್",
    "ವೈರ್",
    "ಕಚ್ಚಿದೆ",
    "ಹಾವು",
    "ಬೆಂಕಿ",
    "ಹುಷಾರಿಲ್ಲ",
    "ಮನೆ ಒಳಗೆ",
    "ಬಿದ್ರೆ",
    "ಬಿದ್ರು",
    "ಬೀಳೋ",
    "ಎದೆ ನೋವು",
    "ಹೊಡೆದಾಡ",
    "ಕೆಂಪು",
    "ಆಂಬುಲೆನ್ಸ್",
    "ಉಕ್ಕಿ",
]
EMERGENCY_WORDS = ["ಕಚ್ಚಿದೆ", "ಸ್ಪಾರ್ಕ್", "ಎದೆ ನೋವು", "ಮುಚ್ಚಳ ಇಲ್ಲ", "ಹುಷಾರಿಲ್ಲ", "ಆಂಬುಲೆನ್ಸ್"]
URGENT_WORDS = ["ಬೇಗ", "ಒಡೆದು", "ಬೆಂಕಿ", "ಮನೆ ಒಳಗೆ", "ಬಿದ್ರು", "ಉಕ್ಕಿ", "ಭಯ", "ಬೀಳೋ"]
ROUTINE_WORDS = ["ಯಾವಾಗ", "ಅನ್ಸುತ್ತೆ", "ಹಗಲು", "ಬೊಗಳ", "ಕಡಿಮೆ", "ಮಿಕ್ಸ್ ಮಾಡ್ತಾರೆ", "ಫುಟ್ ಪಾತ್"]
REPEAT_WORDS = ["ಕಂಪ್ಲೈಂಟ್ ಕೊಟ್ಟಿದ್ದೆ", "ಸಲ", "ಇನ್ನೂ", "ಹಾಗೇ ಇದೆ", "ಇಷ್ಟು ದಿನ"]
HUMAN_WORDS = ["ಆಫೀಸರ್"]
ABUSIVE_WORDS = ["ಸೋಮಾರಿ", "ಕೆಲಸಕ್ಕೆ ಬಾರದ", "ಸುಳ್ಳು"]
DISTRESS_WORDS = [
    "ಬೇಗ ಬನ್ನಿ",
    "ಕಚ್ಚಿದೆ",
    "ಹುಷಾರಿಲ್ಲ",
    "ಪ್ಲೀಸ್",
    "ಎದೆ ನೋವು",
    "ಹೊಡೆದಾಡ",
    "ಮನೆ ಒಳಗೆ",
    "ಮುಚ್ಚಳ ಇಲ್ಲ",
    "ಆಂಬುಲೆನ್ಸ್",
]
FRUSTRATION_WORDS = ["ಇನ್ನೂ", "ಯಾರೂ", "ಬಾಟ್", "ದುಡ್ಡು", "ತಿಂಗಳಾಯ್ತು", "ಸಲ", "ಕೇಳ್ತಿಲ್ಲ"]
NUMBER_WORDS = {
    "ಒನ್": 1,
    "ಟೂ": 2,
    "ತ್ರೀ": 3,
    "ಫೋರ್": 4,
    "ಫೈವ್": 5,
    "ಸಿಕ್ಸ್": 6,
    "ಸೆವೆನ್": 7,
    "ಎಯ್ಟ್": 8,
    "ನೈನ್": 9,
    "ಟೆನ್": 10,
}


def has_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def keyword_rules(text: str) -> dict[str, Any]:
    ward = "unknown"
    for candidate in WARDS:
        names = [candidate["name"]] + [landmark["name"] for landmark in candidate["landmarks"]]
        if any(name in text for name in names) or ("ಆಸ್ಪತ್ರೆ" in text and candidate["id"] == "w04"):
            ward = candidate["id"]
            break
    number = re.search(r"ವಾರ್ಡ್ (\d+|\S+)", text)
    if ward == "unknown" and number:
        value = number.group(1)
        n = int(value) if value.isdigit() else NUMBER_WORDS.get(value)
        if n and 1 <= n <= len(WARDS):
            ward = f"w{n:02d}"

    out_of_scope = has_any(text, OUT_OF_SCOPE_WORDS)
    department = "other"
    if not out_of_scope:
        for name, words in DEPARTMENT_WORDS.items():
            if has_any(text, words):
                department = name
                break
    urgency = (
        "emergency"
        if has_any(text, EMERGENCY_WORDS)
        else "urgent"
        if has_any(text, URGENT_WORDS)
        else "routine"
        if has_any(text, ROUTINE_WORDS)
        else "soon"
    )
    digits = re.sub(r"\D", "", " ".join(re.findall(r"\d[\d ]*\d", text)))
    state = (
        "abusive"
        if has_any(text, ABUSIVE_WORDS)
        else "distressed"
        if has_any(text, DISTRESS_WORDS)
        else "frustrated"
        if has_any(text, FRUSTRATION_WORDS)
        else "calm"
    )
    return {
        "department": department,
        "ward": ward,
        "urgency": urgency,
        "safety_hazard": has_any(text, HAZARD_WORDS),
        "has_location": ward != "unknown" or has_any(text, ["ಹತ್ತಿರ", "ಮುಂದೆ", "ಹಿಂದೆ"]),
        "repeat_complaint": has_any(text, REPEAT_WORDS),
        "wants_human": has_any(text, HUMAN_WORDS),
        "caller_state": state,
        "out_of_scope": out_of_scope,
        "contains_personal_id": len(digits) >= 10,
    }


# --- scoring -----------------------------------------------------------------------------


def macro_f1(gold: list[str], pred: list[str]) -> float:
    scores = []
    for label in sorted(set(gold)):
        tp = sum(g == p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(p == label != g for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label != p for g, p in zip(gold, pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return statistics.mean(scores)


def recall(gold: list[bool], pred: list[bool]) -> float | None:
    positives = [p for g, p in zip(gold, pred, strict=True) if g]
    return sum(positives) / len(positives) if positives else None


def score(rows: list[dict], predictions: list[dict]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in SCHEMA:
        gold = [row["labels"][name] for row in rows]
        pred = [p[name] for p in predictions]
        entry = {"accuracy": sum(g == p for g, p in zip(gold, pred, strict=True)) / len(rows)}
        if name in ("department", "ward", "caller_state"):
            entry["macro_f1"] = macro_f1(gold, pred)
        if name == "urgency":
            entry["within_one_level"] = sum(
                abs(URGENCY.index(g) - URGENCY.index(p)) <= 1
                for g, p in zip(gold, pred, strict=True)
            ) / len(rows)
        if name in NOUL:
            entry["recall"] = recall(gold, pred)
        metrics[name] = entry
    return metrics


# --- runner ------------------------------------------------------------------------------


DEVICE = "cpu"


def laya_variant(kannada_questions: bool, use_gloss: bool, model: str | None) -> Callable:
    from laya import Router

    router = Router(device=DEVICE)
    questions = laya_questions(kannada_questions)

    def predict(row: dict) -> tuple[dict, float]:
        state = row["gloss"] if use_gloss else row["text"]
        started = time.perf_counter()
        # Laya's default lengths: a lower max_len cuts the option descriptions off and the
        # model answers without seeing most of the options.
        result = router.predict(state, questions, model=model)
        elapsed = time.perf_counter() - started
        if result["usage"]["truncated"]:
            raise RuntimeError(
                f"{row['id']}: Laya truncated {result['usage']['truncated_questions']}"
            )
        return decode(result["answers"]), elapsed

    return predict


def rules_variant() -> Callable:
    def predict(row: dict) -> tuple[dict, float]:
        started = time.perf_counter()
        return keyword_rules(row["text"]), time.perf_counter() - started

    return predict


VARIANTS: dict[str, tuple[str, Callable[[], Callable]]] = {
    "A": ("kn-state/en-questions", lambda: laya_variant(False, False, "multilingual")),
    "B": ("kn-state/kn-questions", lambda: laya_variant(True, False, "multilingual")),
    "C": ("en-gloss/en-questions", lambda: laya_variant(False, True, "english")),
    "D": ("keyword-rules", rules_variant),
}


def run(variant_keys: list[str], limit: int | None) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (DATA / "seed" / "seed.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ][:limit]
    report: dict[str, Any] = {"rows": len(rows), "variants": {}}
    for key in variant_keys:
        name, build = VARIANTS[key]
        predict = build()
        predict(rows[0])  # warm-up: model download and first-call overhead
        predictions, seconds = [], []
        for row in rows:
            labels, elapsed = predict(row)
            predictions.append(labels)
            seconds.append(elapsed)
        report["variants"][key] = {
            "name": name,
            "metrics": score(rows, predictions),
            "seconds_per_turn": {
                "median": statistics.median(seconds),
                "p95": sorted(seconds)[max(0, round(0.95 * len(seconds)) - 1)],
            },
            "predictions": [
                {"id": row["id"], **p} for row, p in zip(rows, predictions, strict=True)
            ],
        }
        print(f"{key} {name}: done", flush=True)
    return report


def markdown(report: dict[str, Any]) -> str:
    keys = list(report["variants"])
    header = (
        "| Decision | Metric | "
        + " | ".join(f"{k}: {report['variants'][k]['name']}" for k in keys)
        + " |"
    )
    lines = [header, "|" + " --- |" * (len(keys) + 2)]

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    for name in SCHEMA:
        metrics = report["variants"][keys[0]]["metrics"][name]
        for metric in metrics:
            values = [report["variants"][k]["metrics"][name][metric] for k in keys]
            lines.append(f"| {name} | {metric} | " + " | ".join(cell(v) for v in values) + " |")
    medians = [report["variants"][k]["seconds_per_turn"]["median"] for k in keys]
    lines.append(
        "| latency | s/turn (median, CPU) | " + " | ".join(f"{m:.2f}" for m in medians) + " |"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu, or cuda on Kaggle / Colab")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    report = run(args.variants, args.limit)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8"
    )
    print(markdown(report))


if __name__ == "__main__":
    main()
