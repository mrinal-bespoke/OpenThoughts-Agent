#!/usr/bin/env python3
"""Fail fast when a policy eval would violate preserved Evalchemy resume state."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


class ResumeFingerprintError(RuntimeError):
    """Raised when a requested eval leg cannot safely resume."""


@dataclass(frozen=True)
class TaskProtocol:
    subdir: str
    task_name: str
    max_tokens: int | str
    num_fewshot: int
    apply_chat_template: bool


_MATH_TOKENS = "math_tokens"
TASK_PROTOCOLS = {
    "MATH500": TaskProtocol("math500", "MATH500", _MATH_TOKENS, 0, True),
    "AIME24": TaskProtocol("aime24", "AIME24", _MATH_TOKENS, 0, True),
    "HumanEvalPlus": TaskProtocol("humanevalplus", "HumanEvalPlus", 1024, 0, True),
    "MBPPPlus": TaskProtocol("mbppplus", "MBPPPlus", 1024, 0, True),
    "GPQADiamond": TaskProtocol("gpqa_diamond", "GPQADiamond", _MATH_TOKENS, 0, True),
    "gsm8k": TaskProtocol("gsm8k", "gsm8k", 512, 0, True),
    "mmlu": TaskProtocol("mmlu", "mmlu", 256, 5, False),
    "hellaswag": TaskProtocol("hellaswag", "hellaswag", 256, 10, False),
    "arc_challenge": TaskProtocol("arc_challenge", "arc_challenge", 256, 25, False),
    "arc_easy": TaskProtocol("arc_easy", "arc_easy", 256, 0, False),
    "piqa": TaskProtocol("piqa", "piqa", 256, 0, False),
    "winogrande": TaskProtocol("winogrande", "winogrande", 256, 5, False),
    "openbookqa": TaskProtocol("openbookqa", "openbookqa", 256, 0, False),
    "boolq": TaskProtocol("boolq", "boolq", 256, 0, False),
    "truthfulqa_mc2": TaskProtocol("truthfulqa_mc2", "truthfulqa_mc2", 256, 0, False),
    "lambada_openai": TaskProtocol("lambada_openai", "lambada_openai", 256, 0, False),
    "triviaqa": TaskProtocol("triviaqa", "triviaqa", 128, 5, True),
    "nq_open": TaskProtocol("nq_open", "nq_open", 128, 5, True),
    "drop": TaskProtocol("drop", "drop", 256, 3, True),
}


def _parse_legs(legs: str) -> list[tuple[str, int]]:
    parsed = []
    for leg in legs.split(";"):
        task, separator, seed_text = leg.partition("@")
        if (
            not separator
            or not seed_text.isdigit()
            or seed_text != str(int(seed_text))
            or task not in TASK_PROTOCOLS
        ):
            raise ResumeFingerprintError(f"malformed or unsupported eval leg: {leg}")
        parsed.append((task, int(seed_text)))
    return parsed


def _expected_payload(
    protocol: TaskProtocol,
    *,
    seed: int,
    max_model_len: int,
    limit: float | None,
) -> dict[str, object]:
    max_tokens = protocol.max_tokens
    if max_tokens == _MATH_TOKENS:
        # The wrapper reserves 64 tokens between vLLM and Evalchemy, while its
        # math-generation cap reserves 1024 tokens from the vLLM context.
        max_tokens = min(8192, max_model_len - 960)
    return {
        "apply_chat_template": protocol.apply_chat_template,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "num_fewshot": protocol.num_fewshot,
        "task_name": protocol.task_name,
        "temperature": 0.7,
        "top_p": 1.0,
        "seed": seed,
        "limit": limit,
    }


def _validate_payload(
    path: Path,
    payload: dict[str, object],
    expected: dict[str, object],
) -> None:
    mismatches = {}
    for field in (
        "apply_chat_template",
        "max_model_len",
        "max_tokens",
        "num_fewshot",
        "task_name",
        "temperature",
        "top_p",
    ):
        if payload.get(field) != expected[field]:
            mismatches[field] = {
                "this": expected[field],
                "preserved": payload.get(field),
            }

    seeds = payload.get("seed_set")
    if not isinstance(seeds, list) or not seeds or set(seeds) != {expected["seed"]}:
        mismatches["seed_set"] = {
            "this": [expected["seed"]],
            "preserved": seeds,
        }

    rendered = payload.get("rendered_config")
    if not isinstance(rendered, dict) or "limit" not in rendered:
        mismatches["rendered_config.limit"] = {
            "this": expected["limit"],
            "preserved": "missing",
        }
        preserved_limit = None
    else:
        preserved_limit = rendered["limit"]
    if preserved_limit != expected["limit"]:
        mismatches["limit"] = {
            "this": expected["limit"],
            "preserved": preserved_limit,
        }

    if mismatches:
        raise ResumeFingerprintError(
            f"resume fingerprint mismatch at {path}: {mismatches}"
        )


def validate_resume_fingerprints(
    *,
    eval_root: Path,
    legs: str,
    expected_max_model_len: int,
    resume_mode: str = "auto",
    limit: float | None = None,
    require_fingerprint: bool = False,
) -> list[Path]:
    """Validate material fields for existing resume fingerprints.

    A fresh leg has no fingerprint and is accepted. ``force-fresh`` explicitly
    opts out because Evalchemy will create a new run rather than resume.
    """
    if resume_mode == "force-fresh":
        return []
    if expected_max_model_len <= 0:
        raise ResumeFingerprintError("expected max model length must be positive")

    validated = []
    for task, seed in _parse_legs(legs):
        protocol = TASK_PROTOCOLS[task]
        resume_root = eval_root / protocol.subdir / f"seed-{seed}" / ".resume"
        fingerprints = sorted(resume_root.glob("**/fingerprint.json"))
        if require_fingerprint and not fingerprints:
            raise ResumeFingerprintError(
                f"required resume fingerprint not found for {task}@{seed} under {resume_root}"
            )
        expected = _expected_payload(
            protocol,
            seed=seed,
            max_model_len=expected_max_model_len,
            limit=limit,
        )
        for path in fingerprints:
            try:
                document = json.loads(path.read_text())
                payload = document["canonical_payload"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise ResumeFingerprintError(
                    f"cannot read resume fingerprint {path}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ResumeFingerprintError(
                    f"resume fingerprint has no canonical payload: {path}"
                )
            _validate_payload(path, payload, expected)
            validated.append(path)
    return validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--legs", required=True)
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    parser.add_argument("--resume-mode", default="auto")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--require-fingerprint", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    validated = validate_resume_fingerprints(
        eval_root=args.eval_root,
        legs=args.legs,
        expected_max_model_len=args.expected_max_model_len,
        resume_mode=args.resume_mode,
        limit=args.limit,
        require_fingerprint=args.require_fingerprint,
    )
    print(f"EVAL_RESUME_PREFLIGHT_OK fingerprints={len(validated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
