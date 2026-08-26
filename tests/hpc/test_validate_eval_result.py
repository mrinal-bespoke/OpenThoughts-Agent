import json
from pathlib import Path

import pytest

from hpc.vllm.validate_eval_result import ResultValidationError, validate_result


MODEL = "laion/sft-repro-thinking-step630-nemotron-terminal-step1888"
REVISION = "0ef753164bb3004f23b5dca904f27e40710ce48e"


def _write_result(
    root: Path,
    *,
    task: str = "hellaswag",
    sample_field: str | None = "sample_len",
    sample_count: int = 10042,
    metric=0.7,
    metric_key: str = "acc_norm,none",
    seed: int = 42,
    model: str = MODEL,
    revision: str = REVISION,
    adapter: str = "local-completions",
    max_length: int = 32704,
    max_tokens: int = 256,
    gen_kwargs: str | None = None,
    chat_template: str | None = None,
    write_samples: bool = False,
) -> Path:
    path = root / "model" / "results_2026.json"
    path.parent.mkdir(parents=True)
    metrics = {"alias": task, metric_key: metric}
    if sample_field is not None:
        metrics[sample_field] = sample_count
    if gen_kwargs is None:
        gen_kwargs = (
            "temperature=0.7,top_p=1.0,repetition_penalty=1.1,"
            f"max_gen_toks={max_tokens}"
        )
    path.write_text(
        json.dumps(
            {
                "results": {task: metrics},
                "config": {
                    "random_seed": seed,
                    "model": adapter,
                    "model_args": (
                        f"model={model},tokenizer=/cache/snapshots/{revision},"
                        f"max_length={max_length}"
                    ),
                    "max_length": max_length,
                    "max_tokens": max_tokens,
                    "gen_kwargs": gen_kwargs,
                },
                "chat_template": chat_template,
            }
        )
    )
    if write_samples:
        sample_path = path.with_name(f"samples_{task}_2026.jsonl")
        sample_path.write_text("\n".join("{}" for _ in range(sample_count)) + "\n")
    return path


def _snapshot(root: Path, *artifacts: Path) -> Path:
    path = root / "before.txt"
    path.write_text("".join(f"{artifact.resolve()}\n" for artifact in artifacts))
    return path


def _validate(
    root: Path,
    snapshot: Path,
    *,
    task: str = "hellaswag",
    adapter: str = "local-completions",
    max_tokens: int = 256,
) -> Path:
    return validate_result(
        output_dir=root,
        task=task,
        seed=42,
        artifact_snapshot=snapshot,
        model_repo=MODEL,
        model_revision=REVISION,
        model_adapter=adapter,
        max_length=32704,
        max_tokens=max_tokens,
    )


def test_accepts_new_numeric_full_sample_artifact(tmp_path):
    artifact = _write_result(tmp_path)
    assert _validate(tmp_path, _snapshot(tmp_path)) == artifact


def test_accepts_custom_benchmark_num_total(tmp_path):
    artifact = _write_result(
        tmp_path,
        task="AIME24",
        sample_field="num_total",
        sample_count=30,
        metric=0.4,
        metric_key="accuracy_avg",
        adapter="local-chat-completions",
        max_tokens=8192,
        chat_template="{{ messages }}",
    )
    assert (
        _validate(
            tmp_path,
            _snapshot(tmp_path),
            task="AIME24",
            adapter="local-chat-completions",
            max_tokens=8192,
        )
        == artifact
    )


def test_accepts_count_from_matching_samples_file(tmp_path):
    artifact = _write_result(
        tmp_path,
        task="HumanEvalPlus",
        sample_field=None,
        sample_count=164,
        metric=0.3,
        adapter="local-chat-completions",
        max_tokens=1024,
        chat_template="{{ messages }}",
        write_samples=True,
    )
    assert (
        _validate(
            tmp_path,
            _snapshot(tmp_path),
            task="HumanEvalPlus",
            adapter="local-chat-completions",
            max_tokens=1024,
        )
        == artifact
    )


def test_rejects_an_artifact_present_in_the_pre_eval_snapshot(tmp_path):
    artifact = _write_result(tmp_path)
    with pytest.raises(ResultValidationError, match="no new results artifact"):
        _validate(tmp_path, _snapshot(tmp_path, artifact))


@pytest.mark.parametrize("metric", [None, "0.7", True])
def test_rejects_non_numeric_scores(tmp_path, metric):
    _write_result(tmp_path, metric=metric)
    with pytest.raises(ResultValidationError, match="complete numeric"):
        _validate(tmp_path, _snapshot(tmp_path))


def test_count_and_score_must_be_in_the_same_result_entry(tmp_path):
    artifact = _write_result(tmp_path, metric=None)
    document = json.loads(artifact.read_text())
    document["results"]["hellaswag:score-only"] = {"acc_norm,none": 0.7}
    artifact.write_text(json.dumps(document))
    with pytest.raises(ResultValidationError, match="complete numeric"):
        _validate(tmp_path, _snapshot(tmp_path))


def test_rejects_an_artifact_for_a_different_task(tmp_path):
    _write_result(tmp_path, task="piqa", sample_count=1838)
    with pytest.raises(ResultValidationError, match="requested task"):
        _validate(tmp_path, _snapshot(tmp_path))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": 123}, "seed"),
        ({"model": "wrong/model"}, "model repo"),
        ({"revision": "bad"}, "model revision"),
        ({"adapter": "local-chat-completions", "chat_template": "x"}, "adapter"),
        ({"max_length": 32640}, "max_length"),
        ({"max_tokens": 128}, "max_tokens"),
        (
            {
                "gen_kwargs": "temperature=0.7,top_p=1.0,repetition_penalty=1.0,max_gen_toks=256"
            },
            "repetition_penalty",
        ),
    ],
)
def test_rejects_protocol_mismatch(tmp_path, change, message):
    _write_result(tmp_path, **change)
    with pytest.raises(ResultValidationError, match=message):
        _validate(tmp_path, _snapshot(tmp_path))


def test_rejects_incomplete_sample_count(tmp_path):
    _write_result(tmp_path, sample_count=3000)
    with pytest.raises(ResultValidationError, match="expected 10042"):
        _validate(tmp_path, _snapshot(tmp_path))


def test_rejects_missing_chat_template_for_chat_task(tmp_path):
    _write_result(
        tmp_path,
        task="AIME24",
        sample_field="num_total",
        sample_count=30,
        adapter="local-chat-completions",
        max_tokens=8192,
    )
    with pytest.raises(ResultValidationError, match="chat template missing"):
        _validate(
            tmp_path,
            _snapshot(tmp_path),
            task="AIME24",
            adapter="local-chat-completions",
            max_tokens=8192,
        )
