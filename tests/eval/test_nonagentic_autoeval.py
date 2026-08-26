"""Focused tests for the fixed nine-benchmark non-agentic eval wrapper.

Covers command construction, protocol identity, dry-run safety, and
resume/skip. No scheduler, retry or DB behaviour is asserted because none is
implemented.
"""
import json
import re
from pathlib import Path

import pytest

from eval.tacc.campaign.nonagentic_autoeval import (
    BENCHMARKS,
    DEFAULT_SUITE_ID,
    POLICY_DEFERRED_BENCHMARKS,
    POLICY_SUITE_ID,
    SUITES,
    SUITE_ID,
    SUITE_SEED,
    build_sbatch_command,
    load_allowlist,
    load_state,
    main,
    pending_tasks,
    pending_legs,
    plan_target,
    save_state,
    target_slug,
)


ENTRY = {
    "model": "laion/sft-repro-thinking-step630-nemotron-terminal-step1888",
    "revision": "0ef753164bb3004f23b5dca904f27e40710ce48e",
    "max_model_len": 32768,
    "enable_thinking": True,
}


def _write_result(root, subdir, payload):
    """Write an Evalchemy-shaped results file under a task directory."""
    target = root / subdir / "nested" / "results_2026.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    return target


def _valid_payload(task="MATH500", value=0.62, seed=SUITE_SEED):
    return {
        "results": {task: {"exact_match,none": value, "alias": task}},
        "config": {"random_seed": seed},
    }


def test_command_pins_model_revision_result_root_and_tasks(tmp_path):
    command = build_sbatch_command(
        ENTRY, sbatch_path="hpc/vllm/x.sbatch", result_root=tmp_path / "r", tasks=["ifeval"]
    )
    joined = " ".join(command)

    assert command[0] == "sbatch"
    assert "--time=47:59:00" in command
    assert command[-1] == "hpc/vllm/x.sbatch"
    assert "RUN_EVAL=1" in joined and "RUN_AGENTIC=0" in joined
    assert "MODEL_REPO=laion/sft-repro-thinking-step630-nemotron-terminal-step1888" in joined
    assert "MODEL_REVISION=0ef753164bb3004f23b5dca904f27e40710ce48e" in joined
    assert "RUN_DIR={}".format(tmp_path / "r") in joined
    assert "EVAL_ROOT_OVERRIDE={}".format(tmp_path / "r") in joined
    assert "REPO_ROOT={}".format(Path.cwd()) in joined
    assert "NONAGENTIC_TASKS_SEMICOLON=ifeval" in joined
    assert "EVAL_SEED=42" in joined
    assert "NONAGENTIC_SUITE={}".format(SUITE_ID) in joined
    assert "MAX_MODEL_LEN=32768" in joined
    assert "ENABLE_THINKING=1" in joined
    assert "EVAL_RESUME_MODE=force-fresh" in joined


def test_command_allows_an_explicit_resume_mode_override(tmp_path):
    entry = dict(ENTRY, env={"EVAL_RESUME_MODE": "auto"})
    command = build_sbatch_command(
        entry, sbatch_path="hpc/vllm/x.sbatch", result_root=tmp_path / "r", tasks=["AIME24@42"]
    )
    export_arg = next(arg for arg in command if arg.startswith("--export="))
    assert export_arg.count("EVAL_RESUME_MODE=auto") == 1


def test_command_transports_multiple_full9_tasks_as_one_export_field(tmp_path):
    tasks = ["ifeval", "gsm8k", "mmlu"]
    command = build_sbatch_command(
        ENTRY,
        sbatch_path="hpc/vllm/x.sbatch",
        result_root=tmp_path / "r",
        tasks=tasks,
    )

    export_arg = next(arg for arg in command if arg.startswith("--export="))
    export_fields = export_arg.removeprefix("--export=").split(",")

    assert "NONAGENTIC_TASKS_SEMICOLON=ifeval;gsm8k;mmlu" in export_fields
    assert "NONAGENTIC_TASKS=ifeval" not in export_fields
    assert not {"gsm8k", "mmlu"}.intersection(export_fields)


def test_slug_separates_revisions_of_one_repo():
    other = dict(ENTRY, revision="ffffffffffffffffffffffffffffffffffffffff")
    assert target_slug(ENTRY) != target_slug(other)


def test_pending_tasks_lists_everything_when_no_results(tmp_path):
    assert pending_tasks(tmp_path) == [benchmark.task for benchmark in BENCHMARKS]


def test_manifest_is_the_fixed_nine_benchmark_handoff_suite():
    assert [benchmark.task for benchmark in BENCHMARKS] == [
        "MATH500",
        "ifeval",
        "gsm8k",
        "mmlu",
        "hellaswag",
        "HumanEvalPlus",
        "MBPPPlus",
        "GPQADiamond",
        "AIME24",
    ]
    assert len({benchmark.subdir for benchmark in BENCHMARKS}) == 9


def test_manifest_carries_task_specific_protocol_settings():
    specs = {benchmark.task: benchmark for benchmark in BENCHMARKS}

    assert specs["gsm8k"].num_fewshot == 5
    assert specs["mmlu"].num_fewshot == 5
    assert specs["hellaswag"].num_fewshot == 10
    assert specs["mmlu"].model_adapter == "local-completions"
    assert specs["hellaswag"].model_adapter == "local-completions"
    assert specs["HumanEvalPlus"].unsafe_code is True
    assert specs["MBPPPlus"].unsafe_code is True


def test_policy_profile_contains_the_19_runnable_policy_benchmarks():
    policy = SUITES[POLICY_SUITE_ID]

    assert [benchmark.task for benchmark in policy.benchmarks] == [
        "MATH500",
        "AIME24",
        "HumanEvalPlus",
        "MBPPPlus",
        "GPQADiamond",
        "gsm8k",
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "piqa",
        "winogrande",
        "openbookqa",
        "boolq",
        "truthfulqa_mc2",
        "lambada_openai",
        "triviaqa",
        "nq_open",
        "drop",
    ]
    assert "ifeval" not in {benchmark.task for benchmark in policy.benchmarks}
    assert DEFAULT_SUITE_ID == SUITE_ID


def test_policy_profile_uses_one_external_seed_per_benchmark():
    specs = {benchmark.task: benchmark for benchmark in SUITES[POLICY_SUITE_ID].benchmarks}

    assert {benchmark.seeds for benchmark in specs.values()} == {(42,)}
    assert specs["AIME24"].seeds == (42,)
    assert specs["GPQADiamond"].seeds == (42,)
    assert len(pending_legs(Path("/definitely/missing"), SUITES[POLICY_SUITE_ID])) == 19


def test_development_and_known_issue_benchmarks_are_coverage_only():
    runnable = {benchmark.task for benchmark in SUITES[POLICY_SUITE_ID].benchmarks}
    deferred = {benchmark.name: benchmark.status for benchmark in POLICY_DEFERRED_BENCHMARKS}

    assert deferred == {
        "OlympiadBench": "known-issue",
        "MMLU-Pro": "known-issue",
        "CruxEval": "known-issue",
        "FinanceBench": "development",
        "IFBench": "development",
        "MRCR": "development",
    }
    assert runnable.isdisjoint(deferred)


def test_policy_resume_is_task_and_seed_specific(tmp_path):
    policy = SUITES[POLICY_SUITE_ID]
    math = next(benchmark for benchmark in policy.benchmarks if benchmark.task == "MATH500")
    _write_result(tmp_path, "math500/seed-42", _valid_payload("MATH500", seed=42))

    outstanding = pending_legs(tmp_path, policy)

    assert "MATH500@42" not in outstanding
    assert [leg for leg in outstanding if leg.startswith("AIME24@")] == ["AIME24@42"]
    assert [leg for leg in outstanding if leg.startswith("GPQADiamond@")] == ["GPQADiamond@42"]
    assert math.seeds == (42,)


def test_policy_command_passes_only_explicit_missing_legs(tmp_path):
    policy = SUITES[POLICY_SUITE_ID]
    command = build_sbatch_command(
        ENTRY,
        sbatch_path="hpc/vllm/x.sbatch",
        result_root=tmp_path / "r",
        tasks=["MATH500@42", "AIME24@42"],
        suite=policy,
    )
    joined = " ".join(command)

    assert "NONAGENTIC_LEGS=MATH500@42;AIME24@42" in joined
    assert "NONAGENTIC_SUITE={}".format(POLICY_SUITE_ID) in joined
    assert "NONAGENTIC_TASKS=" not in joined
    assert "EVAL_SEED=" not in joined


def test_policy_dry_run_reports_deferred_coverage_and_quotes_legs(tmp_path, capsys):
    allowlist = tmp_path / "allow.json"
    allowlist.write_text(json.dumps({"models": [ENTRY]}))

    exit_code = main([
        "--allowlist", str(allowlist),
        "--results-base", str(tmp_path / "results"),
        "--state-file", str(tmp_path / "state.json"),
        "--suite", POLICY_SUITE_ID,
    ])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "suite={} benchmarks=19 legs=19".format(POLICY_SUITE_ID) in out
    assert "IFBench (development)" in out
    assert "NONAGENTIC_LEGS=" in out
    assert "'" in out  # shlex quoting protects the semicolon-delimited value.
    assert not (tmp_path / "state.json").exists()


def test_sbatch_contains_safe_task_routing_and_all_nine_tasks():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'env -u PYTHONPATH HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0' in source
    assert "--confirm_run_unsafe_code" in source
    assert "local-completions" in source
    assert "/v1/completions" in source
    for benchmark in BENCHMARKS:
        assert re.search(r"wants_task\s+{}\b".format(re.escape(benchmark.task)), source)


def test_sbatch_default_walltime_covers_aime_internal_repetitions():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert "#SBATCH --time=12:00:00" in source


def test_sbatch_routes_every_policy_task_and_isolates_seed_outputs():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'NONAGENTIC_LEGS' in source
    assert 'VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"' in source
    assert 'MOE_BACKEND="${MOE_BACKEND:-auto}"' in source
    assert '--moe-backend "$MOE_BACKEND"' in source
    assert 'seed-${seed}' in source
    assert 'run_named_eval_leg "$task" "$seed"' in source
    for benchmark in SUITES[POLICY_SUITE_ID].benchmarks:
        assert re.search(
            r"\b{}\)".format(re.escape(benchmark.task)),
            source,
        ), benchmark.task


def test_sbatch_supports_tp2_dp1_without_expert_parallel():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"' in source
    assert 'DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-$NODE_COUNT}"' in source
    assert 'ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"' in source
    assert '--tensor-parallel-size "$TENSOR_PARALLEL_SIZE"' in source
    assert '--data-parallel-size "$DATA_PARALLEL_SIZE"' in source
    assert 'if [ "$DATA_PARALLEL_SIZE" -gt 1 ]; then' in source
    assert 'if [ "$ENABLE_EXPERT_PARALLEL" = "1" ]; then' in source
    assert '"${DP_ARGS[@]}"' in source
    assert '"${EP_ARGS[@]}"' in source


def test_sbatch_supports_pp2_dp1_without_expert_parallel():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"' in source
    assert 'tensor/pipeline/data parallel sizes must be positive integers' in source
    assert 'TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE' in source
    assert '--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"' in source
    assert '--setting "PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE"' in source
    assert 'pp${PIPELINE_PARALLEL_SIZE}' in source


def test_sbatch_supports_dp_free_single_gpu_preflight():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'if [ "$NODE_COUNT" -lt 1 ]; then' in source
    assert 'echo "FATAL: expected at least 1 node' in source
    assert 'if [ "$DATA_PARALLEL_SIZE" -gt 1 ]; then' in source
    assert 'if [ "$ENABLE_EXPERT_PARALLEL" = "1" ]; then' in source


def test_sbatch_can_disable_vllm_async_scheduling_for_dp_moe_recovery():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-auto}"' in source
    assert '0) ASYNC_SCHEDULING_ARGS=(--no-async-scheduling) ;;' in source
    assert '1) ASYNC_SCHEDULING_ARGS=(--async-scheduling) ;;' in source
    assert '"${ASYNC_SCHEDULING_ARGS[@]}"' in source


def test_sbatch_can_serialize_evalchemy_requests_for_dp_wave_recovery():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'EVAL_NUM_CONCURRENT="${EVAL_NUM_CONCURRENT:-4}"' in source
    assert 'EVAL_NUM_CONCURRENT must be a positive integer' in source
    assert 'num_concurrent=${EVAL_NUM_CONCURRENT}' in source


def test_sbatch_supports_isolated_vllm_overlay_and_bounded_eval_preflight():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'VLLM_PYTHON_OVERLAY="${VLLM_PYTHON_OVERLAY:-}"' in source
    assert 'export PYTHONPATH="$VLLM_PYTHON_OVERLAY:$REPO_ROOT:' in source
    assert 'EVAL_LIMIT="${EVAL_LIMIT:-}"' in source
    assert 'limit_args+=(--limit "$EVAL_LIMIT")' in source


def test_sbatch_validates_resume_fingerprint_before_starting_ray():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    validator = 'validate_eval_resume_fingerprint.py'
    assert validator in source
    assert '--expected-max-model-len "$EVAL_MAX_LENGTH"' in source
    assert 'EVAL_RESUME_MODE="${EVAL_RESUME_MODE:-force-fresh}"' in source
    assert 'auto) RESUME_PREFLIGHT_ARGS+=(--require-fingerprint)' in source
    assert source.index(validator) < source.index(
        '"$PYTHON" -m ray.scripts.scripts start --head'
    )


def test_sbatch_revalidates_overlay_and_requires_every_dp_rank_marker():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'build_vllm_dp_coordination_overlay.py' in source
    assert '--source-package "$OVERLAY_SOURCE_PACKAGE"' in source
    assert 'verify_vllm_dp_patch_markers.py' in source
    assert '--expected-size "$DATA_PARALLEL_SIZE"' in source
    assert '--vllm-pid "$VLLM_PID"' in source
    assert "export RAY_DEDUP_LOGS=0" in source


def test_sbatch_requires_a_fresh_complete_numeric_artifact_per_leg():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert "validate_eval_result.py" in source
    assert "find \"$task_output_dir\" -type f -name 'results_*.json'" in source
    assert '--artifact-snapshot "$artifact_snapshot"' in source
    assert '--task "$task"' in source
    assert '--seed "${EVAL_SEED:-42}"' in source
    assert '--model-repo "$MODEL_REPO"' in source
    assert '--model-revision "$MODEL_REVISION"' in source


def test_sbatch_requires_explicit_model_pins_and_writes_a_run_manifest():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert "must explicitly pin MODEL_REPO and MODEL_REVISION" in source
    assert "write_eval_run_manifest.py" in source
    assert '--setting "ENABLE_THINKING=$ENABLE_THINKING"' in source
    assert '--setting "EVAL_RESUME_MODE=$EVAL_RESUME_MODE"' in source
    assert '--file "result_validator=$REPO_ROOT/hpc/vllm/validate_eval_result.py"' in source
    assert '--setting "NUM_FEWSHOT=$num_fewshot"' in source
    assert '--setting "APPLY_CHAT_TEMPLATE=$apply_chat_template"' in source
    assert 'GEN_KWARGS=temperature=0.7,top_p=1.0,repetition_penalty=1.1' in source


def test_completed_task_is_skipped_and_only_the_rest_requested(tmp_path):
    for benchmark in BENCHMARKS[:-1]:
        _write_result(
            tmp_path,
            benchmark.subdir,
            _valid_payload(benchmark.task),
        )
    assert pending_tasks(tmp_path) == ["AIME24"]


def test_result_from_old_seed_does_not_count_as_complete(tmp_path):
    _write_result(tmp_path, "math500", _valid_payload(seed=1234))
    assert "MATH500" in pending_tasks(tmp_path)


def test_empty_results_block_does_not_count_as_complete(tmp_path):
    _write_result(tmp_path, "math500", {"results": {}})
    assert "MATH500" in pending_tasks(tmp_path)


def test_malformed_artifact_does_not_count_as_complete(tmp_path):
    target = tmp_path / "math500" / "results_bad.json"
    target.parent.mkdir(parents=True)
    target.write_text("{not json")
    assert "MATH500" in pending_tasks(tmp_path)


def test_non_numeric_metric_does_not_count_as_complete(tmp_path):
    _write_result(tmp_path, "math500", {"results": {"MATH500": {"alias": "MATH500"}}})
    assert "MATH500" in pending_tasks(tmp_path)


def test_all_complete_plans_a_skip(tmp_path):
    root = tmp_path / "results" / target_slug(ENTRY)
    for benchmark in BENCHMARKS:
        _write_result(root, benchmark.subdir, _valid_payload(benchmark.task))

    plan = plan_target(ENTRY, results_base=tmp_path / "results", state={"targets": {}})
    assert plan["action"] == "skip"
    assert "valid results" in plan["reason"]


def test_partial_results_plan_a_submit_naming_what_is_missing(tmp_path):
    root = tmp_path / "results" / target_slug(ENTRY)
    for benchmark in BENCHMARKS[:-1]:
        _write_result(root, benchmark.subdir, _valid_payload(benchmark.task))

    plan = plan_target(ENTRY, results_base=tmp_path / "results", state={"targets": {}})
    assert plan["action"] == "submit"
    assert plan["pending"] == ["AIME24"]
    assert "AIME24" in plan["reason"] and "MATH500" in plan["reason"]


def test_already_submitted_target_is_not_resubmitted(tmp_path):
    state = {"targets": {target_slug(ENTRY): {"submitted_job_id": "912345", "closed": False}}}
    plan = plan_target(ENTRY, results_base=tmp_path, state=state)
    assert plan["action"] == "skip"
    assert "912345" in plan["reason"]


def test_state_roundtrip_is_atomic_and_survives_corruption(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, {"version": 1, "targets": {"a": {"closed": True}}})
    assert load_state(path)["targets"]["a"]["closed"] is True
    assert not path.with_suffix(".json.tmp").exists()

    path.write_text("{corrupt")
    assert load_state(path)["targets"] == {}


def test_dry_run_is_the_default_and_submits_nothing(tmp_path, monkeypatch, capsys):
    allowlist = tmp_path / "allow.json"
    allowlist.write_text(
        json.dumps({"models": [dict(ENTRY, job_name="snowball-nonagentic-full9-r1")]})
    )

    def _explode(*args, **kwargs):
        raise AssertionError("dry run must not shell out to sbatch")

    monkeypatch.setattr(
        "eval.tacc.campaign.nonagentic_autoeval.subprocess.run", _explode
    )

    exit_code = main([
        "--allowlist", str(allowlist),
        "--results-base", str(tmp_path / "results"),
        "--state-file", str(tmp_path / "state.json"),
    ])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "DRY RUN" in out
    assert "sbatch" in out
    assert "--job-name=snowball-nonagentic-full9-r1" in out
    assert not (tmp_path / "state.json").exists()


def test_allowlist_rejects_entries_without_a_model(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"models": [{"revision": "abc"}]}))
    with pytest.raises(ValueError):
        load_allowlist(bad)
