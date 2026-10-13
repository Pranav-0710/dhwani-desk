"""Synthetic Kannada caller turns for fine-tuning Laya (#7).

Labels are decided here, not by the language model: every row starts as a spec (scenario from
data/generation/scenarios.json, ward and how it is mentioned, mood, flags), and the model only
writes the caller's sentence for that spec in spoken Kannada. A row is kept only if the text
passes checks that keep it consistent with its labels (Kannada script only, the ward phrase is
there, no other ward's name or landmark is, the ID number is there or no long number is), so
the labels stay right by construction.

Usage:
    python generate_data.py --dry-run --n 5                           # prompts only, no model
    python generate_data.py --n 40 --out out/pilot                    # Kaggle / Colab GPU
    python generate_data.py --n 2500 --out out/train --push-repo <hf-user>/<dataset-repo>

The token for --push-repo comes from the HF_TOKEN environment variable; the notebook sets it
from Kaggle / Colab secrets.
"""

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

SCHEMA: dict[str, Any] = json.loads((DATA / "decision_schema.json").read_text("utf-8"))["questions"]
WARDS: list[dict[str, Any]] = json.loads((DATA / "wards.json").read_text("utf-8"))["wards"]
TEMPLATES: dict[str, Any] = json.loads((DATA / "generation" / "scenarios.json").read_text("utf-8"))
SEED: list[dict[str, Any]] = [
    json.loads(line)
    for line in (DATA / "seed" / "seed.jsonl").read_text("utf-8").splitlines()
    if line.strip()
]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEPARTMENT_WEIGHTS = {
    "water": 0.16,
    "sewage-drainage": 0.16,
    "solid-waste": 0.15,
    "roads": 0.15,
    "streetlights": 0.13,
    "stray-animals": 0.12,
    "other": 0.13,
}
OUT_OF_SCOPE_SHARE = 0.6  # of department "other"
P_WARD, P_GENERIC_PLACE = 0.40, 0.20  # the rest give no location
WARD_FORM_WEIGHTS = {"name": 0.45, "landmark": 0.30, "number": 0.25}
P_REPEAT, P_WANTS_HUMAN, P_PERSONAL_ID = 0.15, 0.08, 0.06

MIN_WORDS, MAX_WORDS = 3, 40
MAX_NEW_TOKENS = 512  # Kannada costs several tokens per word; the reply also carries the gloss
LATIN = re.compile(r"[A-Za-z]")
LONG_NUMBER = re.compile(r"\d(?:[\d ]{8,})\d")
# Allowed in a transcript: the Kannada block, zero-width (non-)joiners, ASCII digits, spaces.
NOT_KANNADA = re.compile(r"[^ಀ-೿‌‍0-9\s]")
PUNCTUATION = re.compile(r"[.,!?;:\"'“”‘’।()\[\]-]+")
VIRAMA = "್"
# Street-level words: a row labelled "no location" must not contain them.
STREET_WORDS = [place["kn"] for place in TEMPLATES["generic_places"]] + ["ಕ್ರಾಸ್", "ಮೇನ್ ರೋಡ್"]


# --- specs -------------------------------------------------------------------------------


def _ward_phrase(rng: random.Random, ward: dict[str, Any], form: str) -> str:
    if form == "name":
        return ward["name"]
    if form == "landmark":
        return rng.choice(ward["landmarks"])["name"]
    number = str(ward["number"])
    word = TEMPLATES["english_number_words"][number]
    return rng.choice([f"ವಾರ್ಡ್ {number}", f"ವಾರ್ಡ್ ನಂಬರ್ {number}", f"ವಾರ್ಡ್ {word}"])


def _mood(rng: random.Random, urgency: str, hazard: bool, repeat: bool, human: bool) -> str:
    if urgency == "emergency":
        return rng.choices(["distressed", "calm"], [0.8, 0.2])[0]
    if repeat or human:
        return rng.choices(["frustrated", "abusive", "calm"], [0.65, 0.15, 0.20])[0]
    if hazard:
        return rng.choices(["distressed", "calm", "frustrated"], [0.45, 0.45, 0.10])[0]
    return rng.choices(["calm", "frustrated", "abusive"], [0.75, 0.21, 0.04])[0]


def sample_spec(rng: random.Random, index: int) -> dict[str, Any]:
    """One row's labels plus what the sentence has to say to earn them."""
    department = rng.choices(list(DEPARTMENT_WEIGHTS), list(DEPARTMENT_WEIGHTS.values()))[0]
    out_of_scope = department == "other" and rng.random() < OUT_OF_SCOPE_SHARE
    pool = TEMPLATES["out_of_scope"] if out_of_scope else TEMPLATES["scenarios"][department]
    scenario = rng.choice(pool)

    roll = rng.random()
    ward_id, location = "unknown", {"form": "none", "kn": None}
    if roll < P_WARD:
        ward = rng.choice(WARDS)
        form = rng.choices(list(WARD_FORM_WEIGHTS), list(WARD_FORM_WEIGHTS.values()))[0]
        ward_id, location = ward["id"], {"form": form, "kn": _ward_phrase(rng, ward, form)}
    elif roll < P_WARD + P_GENERIC_PLACE:
        location = {"form": "generic", "kn": rng.choice(TEMPLATES["generic_places"])["kn"]}

    repeat = not out_of_scope and rng.random() < P_REPEAT
    human = rng.random() < P_WANTS_HUMAN
    personal_id = rng.choice(TEMPLATES["fake_id_numbers"]) if rng.random() < P_PERSONAL_ID else None
    labels = {
        "department": department,
        "ward": ward_id,
        "urgency": scenario["urgency"],
        "safety_hazard": scenario["safety_hazard"],
        "has_location": location["form"] != "none",
        "repeat_complaint": repeat,
        "wants_human": human,
        "caller_state": _mood(rng, scenario["urgency"], scenario["safety_hazard"], repeat, human),
        "out_of_scope": out_of_scope,
        "contains_personal_id": personal_id is not None,
    }
    return {
        "id": f"gen-{index:05d}",
        "labels": labels,
        "scenario": scenario["en"],
        "location": location,
        "personal_id": personal_id,
    }


def sample_specs(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [sample_spec(rng, i) for i in range(1, n + 1)]


# --- prompt ------------------------------------------------------------------------------

SYSTEM = (
    "You write realistic transcripts of phone calls to the civic complaints helpline of "
    "Hosapura, a fictional city in Karnataka. You write exactly what one caller says, in "
    "Kannada, the way a speech-to-text system would transcribe it."
)


def build_messages(spec: dict[str, Any], rng: random.Random) -> list[dict[str, str]]:
    labels = spec["labels"]
    # Examples never show a ward or an ID number the spec does not ask for; the model copies them.
    usable = [
        row
        for row in SEED
        if (labels["ward"] != "unknown" or row["labels"]["ward"] == "unknown")
        and (labels["contains_personal_id"] or not row["labels"]["contains_personal_id"])
    ]
    same = [row for row in usable if row["labels"]["department"] == labels["department"]]
    examples = rng.sample(same, min(2, len(same)))
    examples += rng.sample([row for row in usable if row not in examples], 3 - len(examples))

    location = spec["location"]
    if location["form"] == "none":
        where = "Do not mention any ward, area, street or landmark."
    else:
        where = (
            f'Mention the place as "{location["kn"]}", written exactly like that. '
            "Do not mention any other ward, area or landmark."
        )
    mood = SCHEMA["caller_state"]["criteria"][labels["caller_state"]]
    lines = [
        f"- What happened: {spec['scenario']}.",
        f"- How serious: {labels['urgency']} ({TEMPLATES['urgency_meaning'][labels['urgency']]}).",
        f"- Location: {where}",
        f"- Mood: {mood}.",
        "- Say that you already complained before and nothing happened."
        if labels["repeat_complaint"]
        else "- Do not say that you complained before.",
        "- Ask to speak to an officer, a real person, not the automated system."
        if labels["wants_human"]
        else "- Do not ask for an officer or a person.",
    ]
    if spec["personal_id"]:
        number = spec["personal_id"]
        lines.append(
            f"- Read out your {number['kind']}: {number['digits']}, in digits exactly as written."
        )
    else:
        lines.append("- Do not say any phone, Aadhaar or account number.")

    user = "\n".join(
        [
            "Write what the caller says in one turn: one to three short sentences, "
            f"at most {MAX_WORDS - 10} words.",
            "",
            "The call:",
            *lines,
            "",
            "How to write it:",
            "- Spoken, colloquial Kannada as people really talk on the phone "
            "(ಬರ್ತಿದೆ, ಆಗ್ತಿಲ್ಲ, ಮಾಡ್ಬೇಕು), not formal written Kannada.",
            "- Kannada script only, never Latin letters. Everyday English words stay English "
            "but are written in Kannada script: ಪೈಪ್, ಲೈಟ್, ಕಂಪ್ಲೈಂಟ್, ಆಫೀಸರ್, ರೋಡ್.",
            "- Numbers in digits. Little or no punctuation, like a transcript.",
            "",
            "Style examples from other calls:",
            *[f"- {row['text']}" for row in examples],
            "",
            'Reply with JSON only: {"text": "<what the caller says>", '
            '"gloss": "<its meaning in English>"}',
        ]
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# --- checks ------------------------------------------------------------------------------


def parse_reply(reply: str) -> dict[str, str] | None:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    text, gloss = data.get("text"), data.get("gloss")
    if not isinstance(text, str) or not isinstance(gloss, str):
        return None
    text = " ".join(PUNCTUATION.sub(" ", text).split())
    gloss = " ".join(gloss.split())
    return {"text": text, "gloss": gloss} if text and gloss else None


def squash(text: str) -> str:
    """NFC, without spaces or zero-width joiners: "ಗಾಂಧಿ ನಗರ" and "ಗಾಂಧಿನಗರ" compare equal."""
    return re.sub(r"[\s‌‍]", "", unicodedata.normalize("NFC", text))


def stem(phrase: str) -> str:
    """A phrase without its final virama, so inflected forms match: ಚರ್ಚ್ → ಚರ್ಚಿನ, ಚರ್ಚ್ನಲ್ಲಿ."""
    squashed = squash(phrase)
    return squashed[:-1] if squashed.endswith(VIRAMA) else squashed


def mentions(text: str, phrase: str) -> bool:
    return stem(phrase) in squash(text)


WARD_PHRASES: dict[str, list[str]] = {
    ward["id"]: [ward["name"]]
    + [landmark["name"] for landmark in ward["landmarks"]]
    + TEMPLATES["ward_aliases"].get(ward["id"], [])
    for ward in WARDS
}
WARD_BY_NUMBER = {str(ward["number"]): ward["id"] for ward in WARDS}
NUMBER_BY_WORD = {word: number for number, word in TEMPLATES["english_number_words"].items()}
WARD_NUMBER = re.compile(
    r"ವಾರ್ಡ್\s*(?:ನಂಬರ್\s*)?(\d+(?!\d)|"
    + "|".join(sorted(NUMBER_BY_WORD, key=len, reverse=True))
    + ")"
)


def wards_mentioned(text: str) -> set[str]:
    """Every ward the text points to, by name, landmark, alias or number.

    A number with no ward (ward 11) comes back as "ward-11", so it never passes for a real one.
    """
    found = {
        ward_id
        for ward_id, phrases in WARD_PHRASES.items()
        if any(mentions(text, phrase) for phrase in phrases)
    }
    for match in WARD_NUMBER.finditer(unicodedata.normalize("NFC", text)):
        number = NUMBER_BY_WORD.get(match.group(1), match.group(1))
        found.add(WARD_BY_NUMBER.get(number, f"ward-{number}"))
    return found


def problems(text: str, spec: dict[str, Any]) -> list[str]:
    """Why `text` cannot carry the labels in `spec`; empty when it can."""
    reasons = []
    if LATIN.search(text):
        reasons.append("latin_letters")
    if NOT_KANNADA.search(text):
        reasons.append("not_kannada")
    if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
        reasons.append("length")

    labels, location = spec["labels"], spec["location"]
    mentioned = wards_mentioned(text)
    if labels["ward"] != "unknown":
        if labels["ward"] not in mentioned:
            reasons.append("ward_missing")
        if mentioned - {labels["ward"]}:
            reasons.append("other_ward_mentioned")
    else:
        if mentioned or "ವಾರ್ಡ್" in text:
            reasons.append("ward_mentioned_but_unknown")
        if location["form"] == "generic" and not mentions(text, location["kn"]):
            reasons.append("place_phrase_missing")
        if location["form"] == "none" and any(mentions(text, phrase) for phrase in STREET_WORDS):
            reasons.append("place_mentioned_but_none")

    if spec["personal_id"]:
        if spec["personal_id"]["digits"].replace(" ", "") not in text.replace(" ", ""):
            reasons.append("id_number_missing")
    elif LONG_NUMBER.search(text):
        reasons.append("unexpected_long_number")
    return reasons


def dedup_key(text: str) -> str:
    return re.sub(r"\s+", "", text)


# --- generation --------------------------------------------------------------------------


def load_model(name: str, four_bit: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gpu = torch.cuda.is_available()
    kwargs: dict[str, Any] = {
        # Spread over every GPU (T4 x2). On a CPU, "auto" offloads to disk when RAM is short.
        "device_map": "auto" if gpu else None,
        "dtype": torch.float16 if gpu else torch.float32,
    }
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


def complete(
    model: Any,
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    import torch

    prompts = [
        # enable_thinking=False turns off reasoning traces on models that have them (Qwen3);
        # other chat templates ignore it.
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for messages in conversations
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **batch,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(
        output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
    )


def chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate(
    specs: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    attempts: int,
    temperature: float,
    four_bit: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows that passed every check, and every failed attempt with its reasons."""
    model, tokenizer = load_model(model_name, four_bit)
    rng = random.Random(seed)
    seen = {dedup_key(row["text"]) for row in SEED}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending = specs
    for attempt in range(1, attempts + 1):
        retry = []
        for batch in chunks(pending, batch_size):
            replies = complete(
                model,
                tokenizer,
                [build_messages(spec, rng) for spec in batch],
                temperature,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            for spec, reply in zip(batch, replies, strict=True):
                parsed = parse_reply(reply)
                reasons = problems(parsed["text"], spec) if parsed else ["unparseable"]
                if parsed and not reasons and dedup_key(parsed["text"]) in seen:
                    reasons = ["duplicate"]
                if reasons:
                    rejected.append(
                        {"id": spec["id"], "attempt": attempt, "reasons": reasons, "reply": reply}
                    )
                    retry.append(spec)
                    continue
                seen.add(dedup_key(parsed["text"]))
                accepted.append(
                    {
                        "id": spec["id"],
                        **parsed,
                        "labels": spec["labels"],
                        "meta": {
                            "scenario": spec["scenario"],
                            "location_form": spec["location"]["form"],
                            "model": model_name,
                            "attempt": attempt,
                        },
                    }
                )
            print(f"attempt {attempt}: {len(accepted)} kept, {len(retry)} to retry", flush=True)
        pending = retry
        if not pending:
            break
    accepted.sort(key=lambda row: row["id"])
    return accepted, rejected


# --- report and output -------------------------------------------------------------------


def kept_share(requested: Counter, kept: Counter) -> dict[str, float]:
    return {value: round(kept[value] / count, 3) for value, count in requested.items()}


def report(
    specs: list[dict[str, Any]], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]
) -> dict[str, Any]:
    from laya_benchmark import keyword_rules

    agree = sum(
        keyword_rules(row["text"])["department"] == row["labels"]["department"] for row in accepted
    )
    return {
        "requested": len(specs),
        "kept": len(accepted),
        "first_attempt_pass_rate": (
            sum(row["meta"]["attempt"] == 1 for row in accepted) / len(specs) if specs else 0.0
        ),
        "rejection_reasons": dict(
            Counter(reason for row in rejected for reason in row["reasons"]).most_common()
        ),
        # Weak signal only: the keyword rules were fitted to the seed set, so low agreement can
        # mean new wording rather than a wrong sentence. Read the samples too.
        "keyword_rule_department_agreement": agree / len(accepted) if accepted else None,
        "labels": {
            name: dict(Counter(str(row["labels"][name]) for row in accepted).most_common())
            for name in SCHEMA
        },
        # Specs that failed every attempt are dropped, and some kinds fail more often than
        # others; kept / requested per value shows which ones the kept set is short of.
        "kept_share": {
            name: kept_share(
                Counter(str(spec["labels"][name]) for spec in specs),
                Counter(str(row["labels"][name]) for row in accepted),
            )
            for name in SCHEMA
        }
        | {
            "location_form": kept_share(
                Counter(spec["location"]["form"] for spec in specs),
                Counter(row["meta"]["location_form"] for row in accepted),
            )
        },
    }


def dataset_card(model_name: str, summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "---",
            "language: [kn]",
            "tags: [synthetic, civic-complaints, laya]",
            "---",
            "",
            "# Dhwani Desk synthetic caller turns",
            "",
            "Synthetic Kannada caller turns for fine-tuning Laya on the decisions in "
            "`data/decision_schema.json` of the dhwani-desk repo. Every sentence is "
            f"machine-written by `{model_name}` from a sampled spec; the labels come from the "
            "spec, and a row is kept only if its text passes consistency checks. The city "
            "(Hosapura), its wards and the authority are fictional; ID numbers are placeholders.",
            "",
            "Files: `train.jsonl` (kept rows), `rejected.jsonl` (failed attempts), "
            "`report.json` (pass rates and label distribution).",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=1),
            "```",
            "",
        ]
    )


def write_output(
    out: Path,
    model_name: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", accepted), ("rejected.jsonl", rejected)):
        lines = (json.dumps(row, ensure_ascii=False) for row in rows)
        (out / name).write_text("".join(f"{line}\n" for line in lines), "utf-8")
    (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    (out / "README.md").write_text(dataset_card(model_name, summary), "utf-8")


def push(out: Path, repo: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # create_repo leaves an existing repo as it is, so a public one would stay public.
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is public; make it private before uploading")
    api.upload_folder(folder_path=out, repo_id=repo, repo_type="dataset", commit_message=message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=40, help="rows to request")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("out/pilot"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=3, help="tries per spec")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--four-bit", action="store_true", help="load in 4-bit (one T4)")
    parser.add_argument("--dry-run", action="store_true", help="print prompts, load no model")
    parser.add_argument("--push-repo", help="private HF dataset repo to upload to")
    args = parser.parse_args()

    specs = sample_specs(args.n, args.seed)
    if args.dry_run:
        rng = random.Random(args.seed)
        for spec in specs:
            print(json.dumps(spec["labels"], ensure_ascii=False))
            print(build_messages(spec, rng)[1]["content"], end="\n\n")
        return

    accepted, rejected = generate(
        specs,
        args.model,
        args.batch_size,
        args.attempts,
        args.temperature,
        args.four_bit,
        args.seed,
    )
    summary = {"model": args.model, "seed": args.seed, **report(specs, accepted, rejected)}
    write_output(args.out, args.model, accepted, rejected, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.push_repo:
        push(args.out, args.push_repo, f"{len(accepted)} rows from {args.model}")


if __name__ == "__main__":
    main()
