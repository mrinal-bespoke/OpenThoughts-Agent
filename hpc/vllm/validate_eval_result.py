#!/usr/bin/env python3
"""Validate that one Evalchemy attempt produced a complete policy artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class ResultValidationError(RuntimeError):
    """Raised when an Evalchemy artifact is stale, incomplete, or malformed."""


EXPECTED_SAMPLE_LENGTHS = {
    "MATH500": 500,
    "AIME24": 30,
    "HumanEvalPlus": 164,
    "MBPPPlus": 378,
    "GPQADiamond": 198,
    "gsm8k": 1319,
    "mmlu": 14042,
    "hellaswag": 10042,
    "arc_challenge": 1172,
    "arc_easy": 2376,
    "piqa": 1838,
    "winogrande": 1267,
    "openbookqa": 500,
    "boolq": 3270,
    "truthfulqa_mc2": 817,
    "lambada_openai": 5153,
    "triviaqa": 17944,
    "nq_open": 3610,
    "drop": 9536,
}


def _is_requested_task(result_key: str, task: str) -> bool:
    return result_key == task or result_key.startswith(f"{task}:")


def _has_numeric_score(metrics: dict[str, object]) -> bool:
    count_fields = {
        "num_repeat",
        "num_solved",
        "num_total",
        "sample_count",
        "sample_len",
        "solved_avg",
    }
    for key, value in metrics.items():
        if (
            key in {"alias", "examples", "name", "run_stats", "samples"}
            or key in count_fields
            or "stderr" in key
        ):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return True
    return False


def _parse_key_values(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        return {}
    parsed = {}
    for field in raw.split(","):
        key, separator, value = field.partition("=")
        if separator:
            parsed[key] = value
    return parsed


def _require_float(mapping: dict[str, str], key: str, expected: float) -> None:
    try:
        actual = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError(f"missing or invalid generation setting {key}") from error
    if actual != expected:
        raise ResultValidationError(
            f"generation setting mismatch for {key}: expected {expected}, got {actual}"
        )


def _validate_config(
    payload: dict[str, object],
    *,
    task: str,
    seed: int,
    model_repo: str,
    model_revision: str,
    model_adapter: str,
    max_length: int,
    max_tokens: int,
) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ResultValidationError(f"result has no config for {task}")
    if config.get("random_seed") != seed:
        raise ResultValidationError(
            f"result seed mismatch for {task}: expected {seed}, got {config.get('random_seed')}"
        )
    if config.get("model") != model_adapter:
        raise ResultValidationError(
            f"model adapter mismatch for {task}: expected {model_adapter}, got {config.get('model')}"
        )

    model_args = _parse_key_values(config.get("model_args"))
    if model_args.get("model") != model_repo:
        raise ResultValidationError(
            f"model repo mismatch for {task}: expected {model_repo}, got {model_args.get('model')}"
        )
    tokenizer = model_args.get("tokenizer", "")
    if f"/snapshots/{model_revision}" not in tokenizer:
        raise ResultValidationError(
            f"model revision mismatch for {task}: expected tokenizer snapshot {model_revision}"
        )
    try:
        recorded_max_length = int(
            config.get("max_length", model_args.get("max_length", ""))
        )
    except (TypeError, ValueError) as error:
        raise ResultValidationError(f"missing max_length for {task}") from error
    if recorded_max_length != max_length:
        raise ResultValidationError(
            f"max_length mismatch for {task}: expected {max_length}, got {recorded_max_length}"
        )

    gen_kwargs = _parse_key_values(config.get("gen_kwargs"))
    _require_float(gen_kwargs, "temperature", 0.7)
    _require_float(gen_kwargs, "top_p", 1.0)
    _require_float(gen_kwargs, "repetition_penalty", 1.1)
    try:
        recorded_max_tokens = int(
            config.get("max_tokens", gen_kwargs.get("max_gen_toks", ""))
        )
    except (TypeError, ValueError) as error:
        raise ResultValidationError(f"missing max_tokens for {task}") from error
    if recorded_max_tokens != max_tokens:
        raise ResultValidationError(
            f"max_tokens mismatch for {task}: expected {max_tokens}, got {recorded_max_tokens}"
        )

    chat_template = payload.get("chat_template")
    if model_adapter == "local-chat-completions":
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ResultValidationError(f"chat template missing for chat task {task}")
    elif chat_template is not None:
        raise ResultValidationError(
            f"unexpected chat template for completions task {task}"
        )


def _sample_file_count(artifact: Path, result_key: str) -> int | None:
    prefix = "results_"
    if not artifact.stem.startswith(prefix):
        return None
    timestamp = artifact.stem.removeprefix(prefix)
    sample_file = artifact.with_name(f"samples_{result_key}_{timestamp}.jsonl")
    if not sample_file.is_file():
        return None
    try:
        return sum(1 for line in sample_file.read_text().splitlines() if line.strip())
    except OSError as error:
        raise ResultValidationError(f"cannot read sample file {sample_file}: {error}") from error


def _entry_sample_count(
    artifact: Path, result_key: str, metrics: dict[str, object]
) -> int | None:
    for field in ("sample_len", "num_total"):
        value = metrics.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return _sample_file_count(artifact, result_key)


def _snapshot_paths(path: Path) -> set[Path]:
    try:
        return {
            Path(line).resolve()
            for line in path.read_text().splitlines()
            if line.strip()
        }
    except OSError as error:
        raise ResultValidationError(f"cannot read artifact snapshot {path}: {error}") from error


def validate_result(
    *,
    output_dir: Path,
    task: str,
    seed: int,
    artifact_snapshot: Path,
    model_repo: str,
    model_revision: str,
    model_adapter: str,
    max_length: int,
    max_tokens: int,
) -> Path:
    """Return the newest valid artifact absent from the pre-eval snapshot."""
    if task not in EXPECTED_SAMPLE_LENGTHS:
        raise ResultValidationError(f"no expected sample count configured for {task}")
    existing = _snapshot_paths(artifact_snapshot)
    artifacts = sorted(
        (
            path
            for path in output_dir.rglob("results_*.json")
            if path.resolve() not in existing
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not artifacts:
        raise ResultValidationError(
            f"no new results artifact for {task} under {output_dir}"
        )

    artifact = artifacts[-1]
    try:
        payload = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ResultValidationError(f"cannot read result {artifact}: {error}") from error
    if not isinstance(payload, dict):
        raise ResultValidationError(f"result is not a JSON object: {artifact}")

    _validate_config(
        payload,
        task=task,
        seed=seed,
        model_repo=model_repo,
        model_revision=model_revision,
        model_adapter=model_adapter,
        max_length=max_length,
        max_tokens=max_tokens,
    )

    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ResultValidationError(f"empty results map for {task}: {artifact}")
    task_results = [
        (key, metrics)
        for key, metrics in results.items()
        if _is_requested_task(key, task) and isinstance(metrics, dict)
    ]
    if not task_results:
        raise ResultValidationError(
            f"result does not contain requested task {task}: {artifact}"
        )

    expected_samples = EXPECTED_SAMPLE_LENGTHS[task]
    valid_entries = []
    observed_counts = []
    for result_key, metrics in task_results:
        sample_count = _entry_sample_count(artifact, result_key, metrics)
        observed_counts.append(sample_count)
        if sample_count == expected_samples and _has_numeric_score(metrics):
            valid_entries.append(result_key)
    if not valid_entries:
        raise ResultValidationError(
            f"no complete numeric result entry for {task}: expected {expected_samples} "
            f"samples, got {observed_counts}"
        )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-snapshot", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-adapter", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = validate_result(
        output_dir=args.output_dir,
        task=args.task,
        seed=args.seed,
        artifact_snapshot=args.artifact_snapshot,
        model_repo=args.model_repo,
        model_revision=args.model_revision,
        model_adapter=args.model_adapter,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
    )
    print(f"EVAL_RESULT_OK task={args.task} artifact={artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
