import json
from pathlib import Path

import pytest

from hpc.vllm.validate_eval_resume_fingerprint import (
    ResumeFingerprintError,
    validate_resume_fingerprints,
)


def _write_fingerprint(
    root: Path,
    *,
    subdir: str = "hellaswag",
    task_name: str = "hellaswag",
    max_model_len: int = 32704,
    max_tokens: int = 256,
    num_fewshot: int = 10,
    apply_chat_template: bool = False,
) -> Path:
    path = (
        root
        / subdir
        / "seed-42"
        / ".resume"
        / "model"
        / task_name
        / "resume"
        / "fingerprint.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "canonical_payload": {
                    "apply_chat_template": apply_chat_template,
                    "max_model_len": max_model_len,
                    "max_tokens": max_tokens,
                    "num_fewshot": num_fewshot,
                    "seed_set": [42, 42, 42, 42],
                    "task_name": task_name,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "rendered_config": {"limit": None},
                },
                "schema_version": 1,
                "value": "sha256:test",
            }
        )
    )
    return path


def test_missing_resume_state_is_safe_for_a_fresh_leg(tmp_path):
    assert validate_resume_fingerprints(
        eval_root=tmp_path,
        legs="hellaswag@42",
        expected_max_model_len=32704,
    ) == []


def test_required_resume_state_fails_closed_on_a_wrong_root(tmp_path):
    with pytest.raises(ResumeFingerprintError, match="required resume fingerprint not found"):
        validate_resume_fingerprints(
            eval_root=tmp_path,
            legs="hellaswag@42",
            expected_max_model_len=32704,
            require_fingerprint=True,
        )


def test_matching_resume_fingerprint_is_accepted(tmp_path):
    fingerprint = _write_fingerprint(tmp_path)

    assert validate_resume_fingerprints(
        eval_root=tmp_path,
        legs="hellaswag@42",
        expected_max_model_len=32704,
    ) == [fingerprint]


def test_context_mismatch_fails_before_expensive_serving(tmp_path):
    _write_fingerprint(tmp_path, max_model_len=32704)

    with pytest.raises(ResumeFingerprintError, match="max_model_len.*32640.*32704"):
        validate_resume_fingerprints(
            eval_root=tmp_path,
            legs="hellaswag@42",
            expected_max_model_len=32640,
        )


def test_task_protocol_mismatch_fails_before_expensive_serving(tmp_path):
    _write_fingerprint(tmp_path, apply_chat_template=True)

    with pytest.raises(ResumeFingerprintError, match="apply_chat_template"):
        validate_resume_fingerprints(
            eval_root=tmp_path,
            legs="hellaswag@42",
            expected_max_model_len=32704,
        )


def test_force_fresh_deliberately_bypasses_resume_validation(tmp_path):
    _write_fingerprint(tmp_path, max_model_len=32704)

    assert validate_resume_fingerprints(
        eval_root=tmp_path,
        legs="hellaswag@42",
        expected_max_model_len=32640,
        resume_mode="force-fresh",
    ) == []


def test_fingerprint_layout_does_not_depend_on_an_intermediate_resume_dir(tmp_path):
    fingerprint = _write_fingerprint(tmp_path)
    alternate = fingerprint.parent.parent / "fingerprint.json"
    alternate.write_text(fingerprint.read_text())
    fingerprint.unlink()

    assert validate_resume_fingerprints(
        eval_root=tmp_path,
        legs="hellaswag@42",
        expected_max_model_len=32704,
        require_fingerprint=True,
    ) == [alternate]


def test_missing_rendered_limit_fails_closed(tmp_path):
    fingerprint = _write_fingerprint(tmp_path)
    document = json.loads(fingerprint.read_text())
    del document["canonical_payload"]["rendered_config"]
    fingerprint.write_text(json.dumps(document))

    with pytest.raises(ResumeFingerprintError, match="rendered_config.limit"):
        validate_resume_fingerprints(
            eval_root=tmp_path,
            legs="hellaswag@42",
            expected_max_model_len=32704,
        )


def test_noncanonical_seed_spelling_is_rejected(tmp_path):
    with pytest.raises(ResumeFingerprintError, match="malformed"):
        validate_resume_fingerprints(
            eval_root=tmp_path,
            legs="hellaswag@042",
            expected_max_model_len=32704,
        )


@pytest.mark.parametrize(
    ("leg", "subdir", "task_name", "max_tokens", "num_fewshot", "chat_template"),
    [
        ("AIME24@42", "aime24", "AIME24", 8192, 0, True),
        ("triviaqa@42", "triviaqa", "triviaqa", 128, 5, True),
        ("drop@42", "drop", "drop", 256, 3, True),
    ],
)
def test_remaining_policy_legs_use_their_exact_resume_protocol(
    tmp_path,
    leg,
    subdir,
    task_name,
    max_tokens,
    num_fewshot,
    chat_template,
):
    fingerprint = _write_fingerprint(
        tmp_path,
        subdir=subdir,
        task_name=task_name,
        max_tokens=max_tokens,
        num_fewshot=num_fewshot,
        apply_chat_template=chat_template,
    )

    assert validate_resume_fingerprints(
        eval_root=tmp_path,
        legs=leg,
        expected_max_model_len=32704,
    ) == [fingerprint]
