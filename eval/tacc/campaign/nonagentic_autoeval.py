#!/usr/bin/env python3
"""Minimal wrapper: submit versioned non-agentic evaluation suites on Vista.

Scope is deliberately small. This is a thin wrapper around the already-proven
hpc/vllm/smoke_grug_vllm_tacc.sbatch (Marin-vLLM + Evalchemy), not a scheduler.
It decides WHICH models still need suite tasks and builds the sbatch command;
Slurm, vLLM and Evalchemy do everything else.

What it does:
  * reads an explicit model allowlist (no discovery, no queue flooding)
  * preserves the fixed nine-benchmark midtrain handoff suite at seed 42
  * offers an opt-in production profile derived from Marin policy issue #7958
  * detects per-task or per-task+seed completion and skips finished legs
  * keeps a tiny atomic JSON state file for dedupe across runs
  * prints the exact command by default and submits only with --submit

What it deliberately does NOT do: capacity management, retry policy, daemon
mode, DB writes, or benchmark plugins. Those are separate decisions, and V1
exists to be run and debugged end to end first.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUITE_ID = "midtrain9-v1-seed42"
SUITE_SEED = 42


@dataclass(frozen=True)
class BenchmarkSpec:
    task: str
    subdir: str
    max_tokens: int
    num_fewshot: int = 0
    model_adapter: str = "local-chat-completions"
    unsafe_code: bool = False
    seeds: Tuple[int, ...] = (SUITE_SEED,)


@dataclass(frozen=True)
class DeferredBenchmark:
    name: str
    status: str


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    benchmarks: Tuple[BenchmarkSpec, ...]
    seed_subdirs: bool = False


BENCHMARKS: Tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("MATH500", "math500", 8192),
    BenchmarkSpec("ifeval", "ifeval", 2048),
    BenchmarkSpec("gsm8k", "gsm8k", 512, num_fewshot=5),
    BenchmarkSpec("mmlu", "mmlu", 256, num_fewshot=5, model_adapter="local-completions"),
    BenchmarkSpec("hellaswag", "hellaswag", 256, num_fewshot=10, model_adapter="local-completions"),
    BenchmarkSpec("HumanEvalPlus", "humanevalplus", 1024, unsafe_code=True),
    BenchmarkSpec("MBPPPlus", "mbppplus", 1024, unsafe_code=True),
    BenchmarkSpec("GPQADiamond", "gpqa_diamond", 8192),
    BenchmarkSpec("AIME24", "aime24", 8192),
)

# The policy's benchmark tables and lm-eval section specify one external run at
# seed 42. Evalchemy itself performs the required repetitions inside AIME24
# (ten) and GPQA Diamond (three), so expanding either into external reruns would
# multiply the policy sample count.
POLICY_SUITE_ID = "marin-policy-production-v1"
DEFAULT_SUITE_ID = SUITE_ID

POLICY_BENCHMARKS: Tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("MATH500", "math500", 8192),
    BenchmarkSpec("AIME24", "aime24", 8192),
    BenchmarkSpec("HumanEvalPlus", "humanevalplus", 1024, unsafe_code=True),
    BenchmarkSpec("MBPPPlus", "mbppplus", 1024, unsafe_code=True),
    BenchmarkSpec("GPQADiamond", "gpqa_diamond", 8192),
    BenchmarkSpec("gsm8k", "gsm8k", 512),
    BenchmarkSpec("mmlu", "mmlu", 256, num_fewshot=5, model_adapter="local-completions"),
    BenchmarkSpec(
        "hellaswag", "hellaswag", 256, num_fewshot=10, model_adapter="local-completions"
    ),
    BenchmarkSpec(
        "arc_challenge", "arc_challenge", 256, num_fewshot=25, model_adapter="local-completions"
    ),
    BenchmarkSpec("arc_easy", "arc_easy", 256, model_adapter="local-completions"),
    BenchmarkSpec("piqa", "piqa", 256, model_adapter="local-completions"),
    BenchmarkSpec(
        "winogrande", "winogrande", 256, num_fewshot=5, model_adapter="local-completions"
    ),
    BenchmarkSpec("openbookqa", "openbookqa", 256, model_adapter="local-completions"),
    BenchmarkSpec("boolq", "boolq", 256, model_adapter="local-completions"),
    BenchmarkSpec(
        "truthfulqa_mc2", "truthfulqa_mc2", 256, model_adapter="local-completions"
    ),
    BenchmarkSpec(
        "lambada_openai", "lambada_openai", 256, model_adapter="local-completions"
    ),
    BenchmarkSpec("triviaqa", "triviaqa", 128, num_fewshot=5),
    BenchmarkSpec("nq_open", "nq_open", 128, num_fewshot=5),
    BenchmarkSpec("drop", "drop", 256, num_fewshot=3),
)

POLICY_DEFERRED_BENCHMARKS: Tuple[DeferredBenchmark, ...] = (
    DeferredBenchmark("OlympiadBench", "known-issue"),
    DeferredBenchmark("MMLU-Pro", "known-issue"),
    DeferredBenchmark("CruxEval", "known-issue"),
    DeferredBenchmark("FinanceBench", "development"),
    DeferredBenchmark("IFBench", "development"),
    DeferredBenchmark("MRCR", "development"),
)

SUITES: Dict[str, SuiteSpec] = {
    SUITE_ID: SuiteSpec(SUITE_ID, BENCHMARKS),
    POLICY_SUITE_ID: SuiteSpec(POLICY_SUITE_ID, POLICY_BENCHMARKS, seed_subdirs=True),
}

DEFAULT_SBATCH = "hpc/vllm/smoke_grug_vllm_tacc.sbatch"
STATE_VERSION = 1


# --------------------------------------------------------------------------
# result artifacts
# --------------------------------------------------------------------------

def _has_real_scores(payload: Any) -> bool:
    """True when an Evalchemy results payload carries at least one numeric score.

    Evalchemy exits 0 and writes `{"results": {}}` when the engine dies, so a
    file existing -- even a well-formed one -- is not evidence of a result.
    Requiring a numeric metric is what separates "evaluated" from "crashed
    quietly", which is the failure this check exists for.
    """
    if not isinstance(payload, dict):
        return False
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        return False
    for task_metrics in results.values():
        if not isinstance(task_metrics, dict):
            continue
        for key, value in task_metrics.items():
            if key in ("alias", "samples"):
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return True
    return False


def _matches_suite_protocol(payload: Any, seed: int) -> bool:
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        return False
    return config.get("random_seed") == seed


def task_is_complete(
    result_root: Path,
    benchmark: BenchmarkSpec,
    *,
    seed: int = SUITE_SEED,
    seed_subdir: bool = False,
) -> bool:
    """Has this benchmark already produced a usable result under result_root?

    Scans for any results_*.json beneath the task directory, because Evalchemy
    nests output under a model-derived path that we do not want to predict.
    """
    task_dir = Path(result_root) / benchmark.subdir
    if seed_subdir:
        task_dir /= "seed-{}".format(seed)
    if not task_dir.is_dir():
        return False
    for candidate in sorted(task_dir.rglob("results_*.json")):
        try:
            payload = json.loads(candidate.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # Malformed or unreadable: treat as absent so it gets rerun rather
            # than silently accepted.
            continue
        if _has_real_scores(payload) and _matches_suite_protocol(payload, seed):
            return True
    return False


def pending_tasks(result_root: Path) -> List[str]:
    """Which of the fixed benchmarks still need running."""
    return [benchmark.task for benchmark in BENCHMARKS if not task_is_complete(result_root, benchmark)]


def pending_legs(result_root: Path, suite: SuiteSpec) -> List[str]:
    """Return the incomplete work units for a suite.

    Legacy full9 work units are task names because it has one global seed and
    the established artifact layout. Policy work units are explicit
    ``task@seed`` legs so retries cannot accidentally accept or overwrite a
    different replicate.
    """
    if not suite.seed_subdirs:
        return [
            benchmark.task
            for benchmark in suite.benchmarks
            if not task_is_complete(result_root, benchmark)
        ]

    outstanding: List[str] = []
    for benchmark in suite.benchmarks:
        for seed in benchmark.seeds:
            if not task_is_complete(
                result_root,
                benchmark,
                seed=seed,
                seed_subdir=True,
            ):
                outstanding.append("{}@{}".format(benchmark.task, seed))
    return outstanding


# --------------------------------------------------------------------------
# allowlist + state
# --------------------------------------------------------------------------

def load_allowlist(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    entries = payload["models"] if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of models (or {{'models': [...]}})")
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("model"):
            raise ValueError(f"{path}: every entry needs a 'model' key; got {entry!r}")
    return entries


def target_slug(entry: Dict[str, Any], suite: SuiteSpec = SUITES[DEFAULT_SUITE_ID]) -> str:
    """Stable identity for a model+revision.

    The revision is part of the key: two revisions of one repo are different
    subjects, and sharing a result root between them would let one resume into
    the other's outputs.
    """
    slug = str(entry["model"]).replace("/", "__")
    revision = entry.get("revision")
    if revision:
        slug = "{}@{}".format(slug, str(revision)[:12])
    return "{}__{}".format(slug, suite.suite_id)


def load_state(path: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {"version": STATE_VERSION, "targets": {}}
    path = Path(path)
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt state file must not wedge the pipeline: dedupe also
            # consults the artifacts on disk, which are the stronger signal.
            return state
        if isinstance(loaded, dict):
            state.update(loaded)
            state.setdefault("targets", {})
    return state


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """Atomic write: a torn state file is worse than none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------

def build_sbatch_command(
    entry: Dict[str, Any],
    *,
    sbatch_path: str,
    result_root: Path,
    tasks: Sequence[str],
    job_name: Optional[str] = None,
    suite: SuiteSpec = SUITES[DEFAULT_SUITE_ID],
) -> List[str]:
    """Build the sbatch argv for one model.

    RUN_DIR is passed explicitly so results land in a STABLE root across
    retries. The sbatch otherwise derives it from $SLURM_JOB_ID, which gives
    every attempt a fresh directory and makes resume impossible.

    NONAGENTIC_TASKS_SEMICOLON carries only the outstanding benchmarks, so a
    rerun does not redo work that already produced a valid result. A semicolon
    is used because commas delimit fields inside Slurm's --export argument.
    """
    repo_root = Path(sbatch_path).resolve().parents[2]
    exports = [
        "ALL",
        "RUN_EVAL=1",
        "RUN_AGENTIC=0",
        "EVAL_RESUME_MODE={}".format(
            (entry.get("env") or {}).get("EVAL_RESUME_MODE", "force-fresh")
        ),
        "MODEL_REPO={}".format(entry["model"]),
        "RUN_DIR={}".format(result_root),
        "EVAL_ROOT_OVERRIDE={}".format(result_root),
        "REPO_ROOT={}".format(repo_root),
    ]
    if suite.seed_subdirs:
        exports.extend(
            [
                "NONAGENTIC_SUITE={}".format(suite.suite_id),
                "NONAGENTIC_LEGS={}".format(";".join(tasks)),
            ]
        )
    else:
        exports.extend(
            [
                "NONAGENTIC_TASKS_SEMICOLON={}".format(";".join(tasks)),
                "NONAGENTIC_SUITE={}".format(suite.suite_id),
                "EVAL_SEED={}".format(SUITE_SEED),
            ]
        )
    if entry.get("revision"):
        exports.append("MODEL_REVISION={}".format(entry["revision"]))
    if entry.get("max_model_len"):
        exports.append("MAX_MODEL_LEN={}".format(entry["max_model_len"]))
    if entry.get("enable_thinking"):
        exports.append("ENABLE_THINKING=1")
    for key, value in sorted((entry.get("env") or {}).items()):
        if key == "EVAL_RESUME_MODE":
            continue
        exports.append("{}={}".format(key, value))

    return [
        "sbatch",
        "--time=47:59:00",
        "--job-name={}".format(job_name or "nonagentic-{}".format(target_slug(entry, suite))),
        "--export={}".format(",".join(exports)),
        sbatch_path,
    ]


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def plan_target(
    entry: Dict[str, Any],
    *,
    results_base: Path,
    state: Dict[str, Any],
    suite: SuiteSpec = SUITES[DEFAULT_SUITE_ID],
) -> Dict[str, Any]:
    """Decide what should happen to one model, and say why."""
    slug = target_slug(entry, suite)
    result_root = Path(results_base) / slug
    outstanding = pending_legs(result_root, suite)
    record = state.get("targets", {}).get(slug, {})

    if not outstanding:
        reason = "all benchmarks have valid results"
        action = "skip"
    elif record.get("submitted_job_id") and not record.get("closed"):
        reason = "already submitted as job {}".format(record["submitted_job_id"])
        action = "skip"
    else:
        all_legs = [
            benchmark.task if not suite.seed_subdirs else "{}@{}".format(benchmark.task, seed)
            for benchmark in suite.benchmarks
            for seed in (benchmark.seeds if suite.seed_subdirs else (SUITE_SEED,))
        ]
        done = [leg for leg in all_legs if leg not in outstanding]
        reason = "needs {}".format(", ".join(outstanding))
        if done:
            reason += " (already done: {})".format(", ".join(done))
        action = "submit"

    return {
        "slug": slug,
        "model": entry["model"],
        "revision": entry.get("revision"),
        "result_root": str(result_root),
        "pending": outstanding,
        "action": action,
        "reason": reason,
    }


def run_once(args: argparse.Namespace) -> int:
    entries = load_allowlist(Path(args.allowlist))
    state = load_state(Path(args.state_file))
    results_base = Path(args.results_base)
    suite = SUITES[args.suite]

    print("suite={} benchmarks={} legs={}".format(
        suite.suite_id,
        len(suite.benchmarks),
        sum(len(benchmark.seeds) for benchmark in suite.benchmarks),
    ))
    if suite.suite_id == POLICY_SUITE_ID:
        deferred = ", ".join(
            "{} ({})".format(benchmark.name, benchmark.status)
            for benchmark in POLICY_DEFERRED_BENCHMARKS
        )
        print("coverage deferred: {}".format(deferred))

    submitted = 0
    for entry in entries:
        plan = plan_target(entry, results_base=results_base, state=state, suite=suite)
        print("[{}] {}".format(plan["action"].upper(), plan["slug"]))
        print("    why : {}".format(plan["reason"]))
        print("    root: {}".format(plan["result_root"]))

        if plan["action"] != "submit":
            continue

        command = build_sbatch_command(
            entry,
            sbatch_path=args.sbatch,
            result_root=Path(plan["result_root"]),
            tasks=plan["pending"],
            job_name=(
                entry.get("job_name")
                if suite.suite_id == DEFAULT_SUITE_ID
                else (entry.get("job_names") or {}).get(suite.suite_id)
            ),
            suite=suite,
        )
        # The policy leg list contains semicolons. shlex.join keeps the printed
        # dry-run command safe to copy into a shell while subprocess still
        # receives the original argv directly.
        print("    cmd : {}".format(shlex.join(command)))

        if not args.submit:
            continue

        completed = subprocess.run(command, capture_output=True, text=True)
        output = (completed.stdout or "") + (completed.stderr or "")
        job_id = None
        for token in output.split():
            if token.isdigit():
                job_id = token
                break
        print("    sbatch: {}".format(output.strip()[:200]))
        state.setdefault("targets", {})[plan["slug"]] = {
            "submitted_job_id": job_id,
            "submitted_at": int(time.time()),
            "tasks": plan["pending"],
            "result_root": plan["result_root"],
            "closed": False,
        }
        save_state(Path(args.state_file), state)
        submitted += 1

    if args.submit:
        print("\nsubmitted {} job(s)".format(submitted))
    else:
        print("\nDRY RUN: nothing submitted. Re-run with --submit to launch.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nonagentic_autoeval",
        description="Submit a versioned non-agentic suite for an explicit model allowlist.",
    )
    parser.add_argument("--allowlist", required=True, help="JSON file listing models to evaluate")
    parser.add_argument("--results-base", required=True, help="stable root holding per-model result dirs")
    parser.add_argument("--state-file", required=True, help="tiny JSON dedupe/resume state")
    parser.add_argument("--sbatch", default=DEFAULT_SBATCH, help="proven eval sbatch to wrap")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default=DEFAULT_SUITE_ID,
        help="versioned suite profile (default preserves the full9 bring-up suite)",
    )
    # Dry-run is the default on purpose: this submits real jobs to a shared
    # allocation, so launching must be an explicit act.
    parser.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_once(args)


if __name__ == "__main__":
    sys.exit(main())
