#!/usr/bin/env python3
"""Run a synthetic sustained-decode liveness canary against a vLLM endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class CanaryFailure(RuntimeError):
    """A fail-closed canary verdict."""


@dataclass(frozen=True)
class MetricSnapshot:
    timestamp: float
    generated_tokens: float
    running_requests: float
    waiting_requests: float


@dataclass(frozen=True)
class StagePlan:
    sustained_request_count: int
    sustained_request_timeout: float


_REQUIRED_METRICS = {
    "generated_tokens": "vllm:generation_tokens_total",
    "running_requests": "vllm:num_requests_running",
    "waiting_requests": "vllm:num_requests_waiting",
}


def parse_prometheus_metrics(text: str, *, timestamp: float) -> MetricSnapshot:
    """Extract and sum the required vLLM metrics across labeled series."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)", line)
        if not match:
            continue
        name, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value

    missing = [name for name in _REQUIRED_METRICS.values() if name not in totals]
    if missing:
        raise CanaryFailure(f"missing required Prometheus metrics: {', '.join(missing)}")
    return MetricSnapshot(
        timestamp=timestamp,
        generated_tokens=totals[_REQUIRED_METRICS["generated_tokens"]],
        running_requests=totals[_REQUIRED_METRICS["running_requests"]],
        waiting_requests=totals[_REQUIRED_METRICS["waiting_requests"]],
    )


class ProgressWatchdog:
    """Detect an in-flight engine that has stopped producing tokens."""

    def __init__(self, *, stall_timeout_seconds: float) -> None:
        self.stall_timeout_seconds = stall_timeout_seconds
        self._last_generated_tokens: float | None = None
        self._last_progress_at: float | None = None

    def observe(self, snapshot: MetricSnapshot) -> None:
        progressed = (
            self._last_generated_tokens is None
            or snapshot.generated_tokens != self._last_generated_tokens
        )
        pending_requests = snapshot.running_requests + snapshot.waiting_requests
        if pending_requests <= 0:
            self._last_progress_at = None
        elif progressed or self._last_progress_at is None:
            self._last_progress_at = snapshot.timestamp
        elif snapshot.timestamp - self._last_progress_at > self.stall_timeout_seconds:
            raise CanaryFailure(
                "generation stalled for "
                f"{snapshot.timestamp - self._last_progress_at:.1f}s with "
                f"{snapshot.running_requests:g} requests running and "
                f"{snapshot.waiting_requests:g} waiting"
            )
        self._last_generated_tokens = snapshot.generated_tokens


_FATAL_PATTERNS = (
    re.compile(r"sample_tokens.*(?:timeout|timed out)", re.IGNORECASE),
    re.compile(r"step_with_batch_queue.*(?:timeout|timed out)", re.IGNORECASE),
)

_COORDINATION_WARNING = re.compile(
    r"No available shared memory broadcast block found in \d+ seconds",
    re.IGNORECASE,
)


def scan_fatal_signature(text: str) -> str | None:
    """Return the first known DP/collective wedge signature, if any."""
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in _FATAL_PATTERNS):
            return line.strip()
    return None


def scan_coordination_warning(text: str) -> str | None:
    """Return the latest informational coordination warning, if present."""
    match = None
    for line in text.splitlines():
        if _COORDINATION_WARNING.search(line):
            match = line.strip()
    return match


def compute_sustained_request_count(
    *,
    aggregate_tokens_per_second: float,
    target_seconds: float,
    tokens_per_request: int,
    concurrency: int,
) -> int:
    """Size a stage from measured throughput and round to full request waves."""
    if aggregate_tokens_per_second <= 0 or target_seconds <= 0:
        raise CanaryFailure("calibration produced no usable token throughput")
    if tokens_per_request <= 0 or concurrency <= 0:
        raise ValueError("tokens_per_request and concurrency must be positive")
    raw_count = math.ceil(aggregate_tokens_per_second * target_seconds / tokens_per_request)
    return max(concurrency, math.ceil(raw_count / concurrency) * concurrency)


def plan_stages(
    *,
    per_stream_tokens_per_second: float,
    elapsed_seconds: float,
    hard_timeout_seconds: float,
    cleanup_margin_seconds: float,
    sustained_target_seconds: float,
    sustained_tokens: int,
    concurrency: int,
    base_request_timeout_seconds: float,
    timeout_multiplier: float,
) -> StagePlan:
    """Fit full sustained waves into the budget remaining after the depth stage."""
    if per_stream_tokens_per_second <= 0 or timeout_multiplier < 1:
        raise CanaryFailure("invalid measured throughput or timeout multiplier")
    sustained_estimate = sustained_tokens / per_stream_tokens_per_second
    sustained_timeout = max(
        base_request_timeout_seconds,
        sustained_estimate * timeout_multiplier + 60,
    )
    available = hard_timeout_seconds - elapsed_seconds - cleanup_margin_seconds
    max_waves = math.floor(available / sustained_timeout)
    if max_waves < 1:
        raise CanaryFailure(
            "insufficient hard-timeout budget for one sustained wave after depth: "
            f"{available:.1f}s available"
        )
    desired_waves = max(1, math.ceil(sustained_target_seconds / sustained_estimate))
    waves = min(desired_waves, max_waves)
    return StagePlan(
        sustained_request_count=waves * concurrency,
        sustained_request_timeout=sustained_timeout,
    )


def compute_depth_request_timeout(
    *,
    per_stream_tokens_per_second: float,
    elapsed_seconds: float,
    hard_timeout_seconds: float,
    cleanup_margin_seconds: float,
    depth_tokens: int,
) -> float:
    """Give the first depth request the remaining wall budget, after a feasibility gate."""
    if per_stream_tokens_per_second <= 0 or depth_tokens <= 0:
        raise CanaryFailure("invalid measured throughput or depth token count")
    available = hard_timeout_seconds - elapsed_seconds - cleanup_margin_seconds
    depth_estimate = depth_tokens / per_stream_tokens_per_second
    if depth_estimate + 60 > available:
        raise CanaryFailure(
            "insufficient hard-timeout budget for the depth request: "
            f"estimate={depth_estimate:.1f}s available={available:.1f}s"
        )
    return available


def validate_success(
    records: Iterable[dict[str, Any]],
    *,
    elapsed_seconds: float,
    min_duration_seconds: float,
    min_depth_tokens: int,
    required_buckets: tuple[int, ...],
    concurrency: int,
) -> dict[str, Any]:
    """Apply the independently reviewed depth, duration, and shape gates."""
    records = list(records)
    if elapsed_seconds < min_duration_seconds:
        raise CanaryFailure(
            f"duration gate failed: {elapsed_seconds:.1f}s < {min_duration_seconds:.1f}s"
        )
    completed_buckets = sorted(
        {
            int(record["requested_tokens"])
            for record in records
            if int(record.get("completion_tokens", 0)) > 0
        }
    )
    missing_buckets = sorted(set(required_buckets) - set(completed_buckets))
    if missing_buckets:
        raise CanaryFailure(f"shape coverage gate failed; missing buckets {missing_buckets}")
    depth_streams = sum(
        1
        for record in records
        if record.get("stage") == "depth"
        and int(record.get("completion_tokens", 0)) >= min_depth_tokens
    )
    if depth_streams < concurrency:
        raise CanaryFailure(
            f"depth streams gate failed: {depth_streams} < required {concurrency}"
        )
    return {
        "elapsed_seconds": elapsed_seconds,
        "request_count": len(records),
        "completion_tokens": sum(int(record["completion_tokens"]) for record in records),
        "completed_buckets": completed_buckets,
        "depth_streams": depth_streams,
    }


_STATUS_PRIORITY = {
    "CANARY_RUNNING": 0,
    "CANARY_OK": 1,
    "CANARY_ABORTED": 2,
    "CANARY_FAIL": 3,
}


def write_status(
    run_dir: Path,
    status: str,
    details: dict[str, Any],
    *,
    lock: threading.RLock | None = None,
) -> Path:
    """Atomically persist the verdict before the wrapper tears the server down."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "canary-status.json"
    guard = lock if lock is not None else contextlib.nullcontext()
    with guard:
        if output.exists():
            try:
                existing = json.loads(output.read_text()).get("status")
            except (json.JSONDecodeError, OSError):
                existing = None
            if _STATUS_PRIORITY.get(existing, -1) >= _STATUS_PRIORITY.get(status, -1):
                return output
        temporary = run_dir / (
            f"canary-status.json.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        )
        with temporary.open("w") as handle:
            handle.write(json.dumps({"status": status, **details}, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    return output


def _append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock, path.open("a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _fetch_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise CanaryFailure(f"GET {url} returned HTTP {response.status}")
        return response.read().decode()


class RuntimeMonitor(threading.Thread):
    """Scrape metrics and logs; hard-exit after durably writing a failure."""

    def __init__(
        self,
        *,
        base_url: str,
        run_dir: Path,
        vllm_log: Path,
        interval_seconds: float,
        stall_timeout_seconds: float,
        unhealthy_window_seconds: float,
        max_failure_rate: float,
        min_health_failures: int,
        hard_timeout_seconds: float,
        started_at: float,
        io_lock: threading.Lock,
        status_lock: threading.RLock,
    ) -> None:
        super().__init__(name="canary-runtime-monitor", daemon=True)
        self.base_url = base_url.rstrip("/")
        self.run_dir = run_dir
        self.vllm_log = vllm_log
        self.interval_seconds = interval_seconds
        self.unhealthy_window_seconds = unhealthy_window_seconds
        self.max_failure_rate = max_failure_rate
        self.min_health_failures = min_health_failures
        self.hard_timeout_seconds = hard_timeout_seconds
        self.started_at = started_at
        self.io_lock = io_lock
        self.status_lock = status_lock
        self.watchdog = ProgressWatchdog(stall_timeout_seconds=stall_timeout_seconds)
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.finished_cleanly = False
        self._health_samples: deque[tuple[float, bool]] = deque()
        self._log_offset = self.vllm_log.stat().st_size if self.vllm_log.exists() else 0
        self._log_fragment = ""
        self._last_coordination_warning: str | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def _fail(self, reason: str) -> None:
        self.error = reason
        try:
            write_status(
                self.run_dir,
                "CANARY_FAIL",
                {"reason": reason, "time": time.time()},
                lock=self.status_lock,
            )
        finally:
            # Threads blocked in a long HTTP generation cannot be cancelled safely.
            # Exit the client process after the verdict is durable; the wrapper's
            # EXIT trap then performs the normal Ray/vLLM cleanup.
            os._exit(2)

    def _scan_log(self) -> None:
        if not self.vllm_log.exists():
            return
        with self.vllm_log.open(errors="replace") as log:
            log.seek(self._log_offset)
            chunk = log.read()
            self._log_offset = log.tell()
        combined = self._log_fragment + chunk
        if combined and not combined.endswith("\n"):
            complete, _, self._log_fragment = combined.rpartition("\n")
        else:
            complete, self._log_fragment = combined, ""
        signature = scan_fatal_signature(complete)
        if signature:
            raise CanaryFailure(f"fatal vLLM coordination signature: {signature}")
        warning = scan_coordination_warning(complete)
        if warning:
            self._last_coordination_warning = warning

    def _record_health(self, now: float, healthy: bool) -> None:
        self._health_samples.append((now, healthy))
        cutoff = now - self.unhealthy_window_seconds
        while len(self._health_samples) >= 2 and self._health_samples[1][0] <= cutoff:
            self._health_samples.popleft()
        if (
            self._health_samples
            and self._health_samples[-1][0] - self._health_samples[0][0]
            >= self.unhealthy_window_seconds
        ):
            failures = sum(not sample[1] for sample in self._health_samples)
            if (
                failures >= self.min_health_failures
                and failures / len(self._health_samples) > self.max_failure_rate
            ):
                raise CanaryFailure(
                    f"metrics non-200/error rate exceeded {self.max_failure_rate:.1%} "
                    f"for {self.unhealthy_window_seconds:.0f}s"
                )

    def run(self) -> None:
        metrics_path = self.run_dir / "canary-metrics.jsonl"
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now - self.started_at > self.hard_timeout_seconds:
                    reason = f"hard timeout exceeded: {self.hard_timeout_seconds:.0f}s"
                    if self._last_coordination_warning:
                        reason += (
                            "; latest coordination warning: "
                            f"{self._last_coordination_warning}"
                        )
                    self._fail(reason)
                log_error = None
                try:
                    self._scan_log()
                except CanaryFailure as exc:
                    self._fail(str(exc))
                except OSError as exc:
                    log_error = str(exc)
                try:
                    text = _fetch_text(self.base_url + "/metrics", timeout=15)
                    snapshot = parse_prometheus_metrics(text, timestamp=now)
                except Exception as exc:
                    try:
                        self._record_health(now, False)
                        _append_jsonl(
                            metrics_path,
                            {
                                "timestamp": time.time(),
                                "error": str(exc),
                                "log_error": log_error,
                                "coordination_warning": self._last_coordination_warning,
                            },
                            self.io_lock,
                        )
                        if isinstance(exc, CanaryFailure) and not str(exc).startswith(
                            "GET "
                        ):
                            raise
                    except Exception as fatal:
                        self._fail(str(fatal))
                else:
                    _append_jsonl(
                        metrics_path,
                        {
                            "timestamp": time.time(),
                            "generated_tokens": snapshot.generated_tokens,
                            "running_requests": snapshot.running_requests,
                            "waiting_requests": snapshot.waiting_requests,
                            "log_error": log_error,
                            "coordination_warning": self._last_coordination_warning,
                        },
                        self.io_lock,
                    )
                    try:
                        self._record_health(now, log_error is None)
                        self.watchdog.observe(snapshot)
                    except CanaryFailure as exc:
                        reason = str(exc)
                        if self._last_coordination_warning:
                            reason += (
                                "; latest coordination warning: "
                                f"{self._last_coordination_warning}"
                            )
                        self._fail(reason)
                self.stop_event.wait(self.interval_seconds)
            self.finished_cleanly = True
        except BaseException as exc:
            self._fail(f"runtime monitor crashed: {type(exc).__name__}: {exc}")


def _request_completion(
    *,
    base_url: str,
    model: str,
    stage: str,
    request_index: int,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = (
        "Count upward from one, spelling every integer as English words on its own line. "
        "Continue without commentary until the response limit is reached."
        if request_index % 2
        else "Write an exhaustive technical explanation of pipelined processor hazards, "
        "including many concrete examples. Continue until the response limit is reached."
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "seed": 42 + request_index,
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise CanaryFailure(f"completion request returned HTTP {response.status}")
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:2_000]
        raise CanaryFailure(
            f"completion request returned HTTP {exc.code}: {body}"
        ) from exc
    completion_tokens = result.get("usage", {}).get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise CanaryFailure(f"completion response lacks numeric token usage: {result.keys()}")
    return {
        "stage": stage,
        "request_index": request_index,
        "requested_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
        "elapsed_seconds": time.monotonic() - started,
        "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
    }


def _run_stage(
    *,
    base_url: str,
    model: str,
    stage: str,
    request_count: int,
    max_tokens: int,
    concurrency: int,
    timeout_seconds: float,
    records_path: Path,
    io_lock: threading.Lock,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    try:
        futures = [
            executor.submit(
                _request_completion,
                base_url=base_url,
                model=model,
                stage=stage,
                request_index=index,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            for index in range(request_count)
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            _append_jsonl(records_path, record, io_lock)
    except Exception:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vllm-log", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--metrics-interval", type=float, default=30)
    parser.add_argument("--stall-timeout", type=float, default=600)
    parser.add_argument("--unhealthy-window", type=float, default=300)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--min-health-failures", type=int, default=2)
    parser.add_argument("--hard-timeout", type=float, default=6_000)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--min-duration", type=float, default=2_700)
    parser.add_argument("--sustained-seconds", type=float, default=2_400)
    parser.add_argument("--cleanup-margin", type=float, default=300)
    parser.add_argument("--calibration-tokens", type=int, default=512)
    parser.add_argument("--sustained-tokens", type=int, default=4_096)
    parser.add_argument("--depth-tokens", type=int, default=16_384)
    parser.add_argument("--min-depth-tokens", type=int, default=8_192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency <= 0:
        raise SystemExit("concurrency must be positive")
    if args.min_health_failures <= 0:
        raise SystemExit("min-health-failures must be positive")
    if args.timeout_multiplier < 1:
        raise SystemExit("timeout-multiplier must be at least 1")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "canary-requests.jsonl"
    if records_path.exists() or (run_dir / "canary-status.json").exists():
        raise SystemExit("refusing to reuse an existing canary artifact root")

    started_at = time.monotonic()
    io_lock = threading.Lock()
    status_lock = threading.RLock()
    write_status(
        run_dir,
        "CANARY_RUNNING",
        {"started_at": time.time(), "model": args.model},
        lock=status_lock,
    )

    def handle_signal(signum: int, _frame: Any) -> None:
        write_status(
            run_dir,
            "CANARY_ABORTED",
            {"signal": signum, "time": time.time()},
            lock=status_lock,
        )
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    monitor = RuntimeMonitor(
        base_url=args.base_url,
        run_dir=run_dir,
        vllm_log=args.vllm_log,
        interval_seconds=args.metrics_interval,
        stall_timeout_seconds=args.stall_timeout,
        unhealthy_window_seconds=args.unhealthy_window,
        max_failure_rate=args.max_failure_rate,
        min_health_failures=args.min_health_failures,
        hard_timeout_seconds=args.hard_timeout,
        started_at=started_at,
        io_lock=io_lock,
        status_lock=status_lock,
    )
    monitor.start()
    records: list[dict[str, Any]] = []
    try:
        calibration_started = time.monotonic()
        records.extend(
            _run_stage(
                base_url=args.base_url,
                model=args.model,
                stage="calibrate",
                request_count=args.concurrency * 2,
                max_tokens=args.calibration_tokens,
                concurrency=args.concurrency,
                timeout_seconds=args.request_timeout,
                records_path=records_path,
                io_lock=io_lock,
            )
        )
        calibration_elapsed = time.monotonic() - calibration_started
        calibration_tokens = sum(record["completion_tokens"] for record in records)
        aggregate_tps = calibration_tokens / calibration_elapsed
        per_stream_tps = aggregate_tps / args.concurrency
        elapsed = time.monotonic() - started_at
        depth_timeout = compute_depth_request_timeout(
            per_stream_tokens_per_second=per_stream_tps,
            elapsed_seconds=elapsed,
            hard_timeout_seconds=args.hard_timeout,
            cleanup_margin_seconds=args.cleanup_margin,
            depth_tokens=args.depth_tokens,
        )
        records.extend(
            _run_stage(
                base_url=args.base_url,
                model=args.model,
                stage="depth",
                request_count=args.concurrency,
                max_tokens=args.depth_tokens,
                concurrency=args.concurrency,
                timeout_seconds=depth_timeout,
                records_path=records_path,
                io_lock=io_lock,
            )
        )
        stage_plan = plan_stages(
            per_stream_tokens_per_second=per_stream_tps,
            elapsed_seconds=time.monotonic() - started_at,
            hard_timeout_seconds=args.hard_timeout,
            cleanup_margin_seconds=args.cleanup_margin,
            sustained_target_seconds=args.sustained_seconds,
            sustained_tokens=args.sustained_tokens,
            concurrency=args.concurrency,
            base_request_timeout_seconds=args.request_timeout,
            timeout_multiplier=args.timeout_multiplier,
        )
        sustained_count = stage_plan.sustained_request_count
        records.extend(
            _run_stage(
                base_url=args.base_url,
                model=args.model,
                stage="sustained",
                request_count=sustained_count,
                max_tokens=args.sustained_tokens,
                concurrency=args.concurrency,
                timeout_seconds=stage_plan.sustained_request_timeout,
                records_path=records_path,
                io_lock=io_lock,
            )
        )
        while time.monotonic() - started_at < args.min_duration:
            remaining_budget = (
                args.hard_timeout
                - (time.monotonic() - started_at)
                - args.cleanup_margin
            )
            if remaining_budget < stage_plan.sustained_request_timeout:
                raise CanaryFailure(
                    "duration top-up cannot fit before the hard timeout"
                )
            records.extend(
                _run_stage(
                    base_url=args.base_url,
                    model=args.model,
                    stage="sustained",
                    request_count=args.concurrency,
                    max_tokens=args.sustained_tokens,
                    concurrency=args.concurrency,
                    timeout_seconds=stage_plan.sustained_request_timeout,
                    records_path=records_path,
                    io_lock=io_lock,
                )
            )
        elapsed = time.monotonic() - started_at
        summary = validate_success(
            records,
            elapsed_seconds=elapsed,
            min_duration_seconds=args.min_duration,
            min_depth_tokens=args.min_depth_tokens,
            required_buckets=(
                args.calibration_tokens,
                args.sustained_tokens,
                args.depth_tokens,
            ),
            concurrency=args.concurrency,
        )
        summary.update(
            {
                "aggregate_calibration_tokens_per_second": aggregate_tps,
                "depth_request_timeout_seconds": depth_timeout,
                "sustained_request_timeout_seconds": stage_plan.sustained_request_timeout,
                "sustained_request_count": sum(
                    record.get("stage") == "sustained" for record in records
                ),
                "thresholds": {
                    "concurrency": args.concurrency,
                    "hard_timeout_seconds": args.hard_timeout,
                    "min_duration_seconds": args.min_duration,
                    "stall_timeout_seconds": args.stall_timeout,
                    "unhealthy_window_seconds": args.unhealthy_window,
                    "max_failure_rate": args.max_failure_rate,
                    "min_health_failures": args.min_health_failures,
                    "timeout_multiplier": args.timeout_multiplier,
                    "calibration_tokens": args.calibration_tokens,
                    "sustained_tokens": args.sustained_tokens,
                    "depth_tokens": args.depth_tokens,
                    "min_depth_tokens": args.min_depth_tokens,
                },
                "finished_at": time.time(),
            }
        )
        monitor.stop()
        monitor.join(timeout=max(60, args.metrics_interval + 35))
        if monitor.error:
            raise CanaryFailure(f"runtime monitor failed: {monitor.error}")
        if monitor.is_alive():
            raise CanaryFailure("runtime monitor did not stop before success")
        if not monitor.finished_cleanly:
            raise CanaryFailure("runtime monitor exited without a clean completion")
        (run_dir / "canary-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_status(run_dir, "CANARY_OK", summary, lock=status_lock)
        print(f"SUSTAINED_DECODE_CANARY_OK summary={run_dir / 'canary-summary.json'}")
        return 0
    except Exception as exc:
        write_status(
            run_dir,
            "CANARY_FAIL",
            {"reason": str(exc), "time": time.time()},
            lock=status_lock,
        )
        print(f"SUSTAINED_DECODE_CANARY_FAIL reason={exc}", file=sys.stderr, flush=True)
        os._exit(2)
    finally:
        monitor.stop()
        monitor.join(timeout=max(60, args.metrics_interval + 35))


if __name__ == "__main__":
    raise SystemExit(main())
