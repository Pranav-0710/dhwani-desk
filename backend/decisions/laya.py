"""Laya as the call's decider: a Laya checkpoint exported to ONNX, run on the CPU.

Each caller turn is two Laya runs over everything the caller has said: the four safety
questions, then the other four (decisions/schema.py). The ward and personal numbers come from
code (decisions/extractors.py). Timings are in docs/decisions/laya-latency.md.

The checkpoint is pinned: a Hugging Face repo at a tag or commit, so a new upload cannot change
a running build, or a local directory for trying an export before it is uploaded. Either way it
holds Laya's own files (rl_agent_config.json, tokenizer/) and the ONNX graph in onnx/, as
training/finetune_laya.py publishes it.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from config import Settings
from decisions.extractors import find_ward, says_personal_number
from decisions.schema import OTHERS, SAFETY
from dialogue.decision import URGENCY_LEVELS, Decision, Screening

logger = logging.getLogger("dhwani.laya")

ONNX_FILE = "onnx/model.onnx"
CHECKPOINT_FILES = ["rl_agent_config.json", "tokenizer/*", "onnx/*"]
# Laya keeps the start of a transcript too long for it and silently drops the end, which is
# what the next reply most depends on. With the longest question (department) it has room for
# ~830 tokens of transcript, so the decider sends at most this many of the latest ones: about
# 1,000 Kannada characters, far more than a normal call. It also keeps an abandoned decision
# short.
MAX_TRANSCRIPT_TOKENS = 700


class Agent(Protocol):
    """The part of `laya.onnx_agent.ONNXAgent` the decider uses."""

    def system_one(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]: ...


class LayaDecider:
    def __init__(
        self, agent: Agent, revision: str, count_tokens: Callable[[str], int] = len
    ) -> None:
        self._agent = agent
        self.revision = revision
        self._count_tokens = count_tokens
        # One run at a time: a run already uses all the threads it is given, and the tokenizer
        # is not safe to share between threads. Calls queue here, so the timings in
        # docs/decisions/laya-latency.md hold for one call at a time: a second caller's safety
        # step waits for the first caller's run to finish.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings: Settings) -> "LayaDecider":
        """Load the checkpoint `settings` name; slow (seconds), so done once at startup."""
        source = settings.laya_checkpoint_repo
        if not source:
            raise ValueError("Set LAYA_CHECKPOINT_REPO for DECIDER=laya")
        if Path(source).is_dir():
            checkpoint = Path(source)
            revision = f"laya:{checkpoint.resolve().name}@local"
        else:
            token = settings.hf_token.get_secret_value() or None
            checkpoint = _download(source, settings.laya_checkpoint_revision, token)
            # Downloads are stored in a directory named after the commit.
            revision = f"laya:{source}@{checkpoint.name}"
        from laya.onnx_agent import ONNXAgent

        onnx_path = checkpoint / ONNX_FILE
        agent = ONNXAgent(str(checkpoint), onnx_path=str(onnx_path))
        agent.session = _cpu_session(onnx_path, settings.laya_threads)
        logger.info("Laya loaded: %s", revision)

        def count_tokens(text: str) -> int:
            return len(agent.tok(text, add_special_tokens=False)["input_ids"])

        return cls(agent, revision, count_tokens)

    def screen(self, transcript: str) -> Screening:
        answers = self._ask(transcript, SAFETY)
        urgency = answers["urgency"]["probabilities"]
        return Screening(
            safety_hazard=answers["safety_hazard"]["noul"],
            # The most likely level. Laya's `score` is the expected level, which pulls an unsure
            # answer to the middle of the scale.
            urgency=URGENCY_LEVELS[int(max(urgency, key=urgency.__getitem__))],
            wants_human=answers["wants_human"]["noul"],
            out_of_scope=answers["out_of_scope"]["noul"],
        )

    def decide(self, transcript: str, screening: Screening) -> Decision:
        answers = self._ask(transcript, OTHERS)
        department = answers["department"]
        return Decision(
            department=department["choice"],
            department_confidence=department["probabilities"][department["choice"]],
            ward=find_ward(transcript),
            urgency=screening.urgency,
            safety_hazard=screening.safety_hazard,
            has_location=answers["has_location"]["noul"],
            repeat_complaint=answers["repeat_complaint"]["noul"],
            wants_human=screening.wants_human,
            caller_state=answers["caller_state"]["choice"],
            out_of_scope=screening.out_of_scope,
            contains_personal_id=1.0 if says_personal_number(transcript) else 0.0,
        )

    def _ask(self, transcript: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            state = recent(transcript, self._count_tokens)
            result = self._agent.system_one(state, questions)
        if truncated := result["usage"].get("truncated_questions"):
            logger.warning("Laya read only part of the transcript for %s", truncated)
        return result["answers"]


def recent(transcript: str, count_tokens: Callable[[str], int]) -> str:
    """The transcript, or its latest whole words that fit in MAX_TRANSCRIPT_TOKENS."""
    if count_tokens(transcript) <= MAX_TRANSCRIPT_TOKENS:
        return transcript
    words = transcript.split(" ")
    # The fewest words dropped from the start that make the rest fit.
    low, high = 1, len(words) - 1
    while low < high:
        middle = (low + high) // 2
        if count_tokens(" ".join(words[middle:])) <= MAX_TRANSCRIPT_TOKENS:
            high = middle
        else:
            low = middle + 1
    return " ".join(words[low:]) or transcript  # one word too long: Laya cuts it itself


def _download(repo: str, revision: str, token: str | None) -> Path:
    if not revision or revision == "main":
        raise ValueError(
            "Set LAYA_CHECKPOINT_REVISION to a tag or commit of LAYA_CHECKPOINT_REPO, not a branch"
        )
    from huggingface_hub import HfApi, constants, snapshot_download

    # Real files in a directory of their own, not the Hub cache's links into its blob store:
    # ONNX Runtime refuses weights (onnx/model.onnx.data) that resolve outside the graph's
    # directory.
    commit = HfApi(token=token).model_info(repo, revision=revision).sha
    target = Path(constants.HF_HOME) / "laya-checkpoints" / repo.replace("/", "--") / commit
    snapshot_download(
        repo, revision=commit, token=token, allow_patterns=CHECKPOINT_FILES, local_dir=target
    )
    return target


def _cpu_session(onnx_path: Path, threads: int) -> Any:
    """Replaces ONNXAgent's own session, which uses ONNX Runtime's default thread count: on the
    laptop's mix of performance and efficiency cores that ran all 8 questions in 9.3 s, against
    4.5 s with 8 threads."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
