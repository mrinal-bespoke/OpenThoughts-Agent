import json
import os
import threading
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest

from hpc.vllm.sustained_decode_canary import (
    CanaryFailure,
    MetricSnapshot,
    ProgressWatchdog,
    RuntimeMonitor,
    _request_completion,
    compute_depth_request_timeout,
    compute_sustained_request_count,
    parse_prometheus_metrics,
    plan_stages,
    scan_coordination_warning,
    scan_fatal_signature,
    validate_success,
    write_status,
)


def test_parse_prometheus_metrics_sums_labeled_engine_series():
    snapshot = parse_prometheus_metrics(
        """
# HELP vllm:generation_tokens_total Total generated tokens.
vllm:generation_tokens_total{engine="0"} 123
vllm:generation_tokens_total{engine="1"} 456
vllm:num_requests_running{engine="0"} 2
vllm:num_requests_running{engine="1"} 1
vllm:num_requests_waiting{engine="0"} 4
vllm:num_requests_waiting{engine="1"} 0
""",
        timestamp=10.0,
    )

    assert snapshot == MetricSnapshot(
        timestamp=10.0,
        generated_tokens=579.0,
        running_requests=3.0,
        waiting_requests=4.0,
    )


def test_parse_prometheus_metrics_fails_closed_when_required_series_are_missing():
    with pytest.raises(CanaryFailure, match="missing required Prometheus metrics"):
        parse_prometheus_metrics("vllm:generation_tokens_total 10\n", timestamp=1.0)


def test_progress_watchdog_ignores_idle_time_and_resets_on_token_progress():
    watchdog = ProgressWatchdog(stall_timeout_seconds=600)

    watchdog.observe(MetricSnapshot(0, 100, 0, 0))
    watchdog.observe(MetricSnapshot(700, 100, 0, 0))
    watchdog.observe(MetricSnapshot(701, 100, 1, 0))
    watchdog.observe(MetricSnapshot(1200, 101, 1, 0))
    watchdog.observe(MetricSnapshot(1799, 101, 1, 0))


def test_progress_watchdog_fails_after_ten_minutes_with_an_inflight_request():
    watchdog = ProgressWatchdog(stall_timeout_seconds=600)
    watchdog.observe(MetricSnapshot(10, 100, 1, 0))

    with pytest.raises(CanaryFailure, match="generation stalled"):
        watchdog.observe(MetricSnapshot(611, 100, 1, 0))


def test_progress_watchdog_fails_with_only_a_waiting_request():
    watchdog = ProgressWatchdog(stall_timeout_seconds=600)
    watchdog.observe(MetricSnapshot(10, 100, 0, 1))

    with pytest.raises(CanaryFailure, match="0 requests running and 1 waiting"):
        watchdog.observe(MetricSnapshot(611, 100, 0, 1))


@pytest.mark.parametrize(
    "line",
    [
        "sample_tokens RPC timeout after 7200 seconds",
        "step_with_batch_queue timed out after 7200 seconds",
    ],
)
def test_scan_fatal_signature_recognizes_coordination_wedge_precursors(line):
    assert scan_fatal_signature(f"ordinary line\n{line}\n") == line


def test_scan_fatal_signature_ignores_unrelated_warnings():
    benign = (
        "WARNING [shm_broadcast.py:404] No available shared memory broadcast block "
        "found in 600 seconds."
    )
    assert scan_fatal_signature(benign) is None
    assert scan_coordination_warning(benign) == benign


def test_sustained_request_count_rounds_up_to_complete_concurrency_waves():
    assert compute_sustained_request_count(
        aggregate_tokens_per_second=10,
        target_seconds=1_000,
        tokens_per_request=4_096,
        concurrency=4,
    ) == 4
    assert compute_sustained_request_count(
        aggregate_tokens_per_second=100,
        target_seconds=1_000,
        tokens_per_request=4_096,
        concurrency=4,
    ) == 28


@pytest.mark.parametrize("per_stream_tps", [4, 5, 6, 7, 8, 20, 50])
def test_stage_plan_never_spends_reserved_depth_or_cleanup_budget(per_stream_tps):
    calibration_elapsed = 2 * 512 / per_stream_tps
    depth_timeout = compute_depth_request_timeout(
        per_stream_tokens_per_second=per_stream_tps,
        elapsed_seconds=calibration_elapsed,
        hard_timeout_seconds=6_000,
        cleanup_margin_seconds=300,
        depth_tokens=16_384,
    )
    depth_elapsed = 16_384 / per_stream_tps
    plan = plan_stages(
        per_stream_tokens_per_second=per_stream_tps,
        elapsed_seconds=calibration_elapsed + depth_elapsed,
        hard_timeout_seconds=6_000,
        cleanup_margin_seconds=300,
        sustained_target_seconds=2_400,
        sustained_tokens=4_096,
        concurrency=4,
        base_request_timeout_seconds=900,
        timeout_multiplier=1.0,
    )

    sustained_waves = plan.sustained_request_count / 4
    assert depth_timeout == 6_000 - calibration_elapsed - 300
    assert (
        calibration_elapsed
        + depth_elapsed
        + sustained_waves * plan.sustained_request_timeout
        + 300
        <= 6_000
    )


def test_stage_plan_fails_early_when_depth_cannot_fit_safely():
    with pytest.raises(CanaryFailure, match="insufficient hard-timeout budget"):
        compute_depth_request_timeout(
            per_stream_tokens_per_second=3,
            elapsed_seconds=2 * 512 / 3,
            hard_timeout_seconds=6_000,
            cleanup_margin_seconds=300,
            depth_tokens=16_384,
        )


def test_validate_success_requires_duration_depth_and_all_shape_buckets():
    records = [
        {"stage": "calibrate", "requested_tokens": 512, "completion_tokens": 512},
        {"stage": "sustained", "requested_tokens": 4_096, "completion_tokens": 4_096},
        *[
            {"stage": "depth", "requested_tokens": 16_384, "completion_tokens": 16_384}
            for _ in range(4)
        ],
    ]

    summary = validate_success(
        records,
        elapsed_seconds=2_701,
        min_duration_seconds=2_700,
        min_depth_tokens=8_192,
        required_buckets=(512, 4_096, 16_384),
        concurrency=4,
    )

    assert summary["depth_streams"] == 4
    assert summary["completed_buckets"] == [512, 4_096, 16_384]


def test_validate_success_fails_when_depth_or_duration_is_too_small():
    records = [
        {"stage": "calibrate", "requested_tokens": 512, "completion_tokens": 512},
        {"stage": "sustained", "requested_tokens": 4_096, "completion_tokens": 4_096},
        *[
            {"stage": "depth", "requested_tokens": 16_384, "completion_tokens": 8_191}
            for _ in range(4)
        ],
    ]

    with pytest.raises(CanaryFailure, match="duration"):
        validate_success(
            records,
            elapsed_seconds=2_699,
            min_duration_seconds=2_700,
            min_depth_tokens=8_192,
            required_buckets=(512, 4_096, 16_384),
            concurrency=4,
        )

    with pytest.raises(CanaryFailure, match="depth streams"):
        validate_success(
            records,
            elapsed_seconds=2_701,
            min_duration_seconds=2_700,
            min_depth_tokens=8_192,
            required_buckets=(512, 4_096, 16_384),
            concurrency=4,
        )


def test_write_status_atomically_replaces_running_with_terminal_state(tmp_path):
    status_path = write_status(tmp_path, "CANARY_RUNNING", {"job": 1})
    write_status(tmp_path, "CANARY_OK", {"requests": 64})

    payload = json.loads(status_path.read_text())
    assert payload == {"status": "CANARY_OK", "requests": 64}
    assert not list(tmp_path.glob("canary-status.json.tmp.*"))


def test_write_status_never_replaces_a_failure_with_success(tmp_path):
    lock = threading.RLock()
    write_status(tmp_path, "CANARY_RUNNING", {}, lock=lock)
    write_status(tmp_path, "CANARY_FAIL", {"reason": "stall"}, lock=lock)
    write_status(tmp_path, "CANARY_OK", {"requests": 64}, lock=lock)

    assert json.loads((tmp_path / "canary-status.json").read_text()) == {
        "status": "CANARY_FAIL",
        "reason": "stall",
    }
    assert not list(tmp_path.glob("canary-status.json.tmp.*"))


def test_health_window_fails_after_five_minutes_of_errors(tmp_path):
    monitor = RuntimeMonitor(
        base_url="http://unused",
        run_dir=tmp_path,
        vllm_log=tmp_path / "vllm.log",
        interval_seconds=30,
        stall_timeout_seconds=600,
        unhealthy_window_seconds=300,
        max_failure_rate=0.05,
        min_health_failures=2,
        hard_timeout_seconds=5_400,
        started_at=0,
        io_lock=threading.Lock(),
        status_lock=threading.RLock(),
    )

    for index in range(10):
        monitor._record_health(index * 30.4, False)
    with pytest.raises(CanaryFailure, match="error rate exceeded"):
        monitor._record_health(10 * 30.4, False)


def test_health_window_tolerates_one_transient_scrape_error(tmp_path):
    monitor = RuntimeMonitor(
        base_url="http://unused",
        run_dir=tmp_path,
        vllm_log=tmp_path / "vllm.log",
        interval_seconds=30,
        stall_timeout_seconds=600,
        unhealthy_window_seconds=300,
        max_failure_rate=0.05,
        min_health_failures=2,
        hard_timeout_seconds=5_400,
        started_at=0,
        io_lock=threading.Lock(),
        status_lock=threading.RLock(),
    )

    for index in range(200):
        monitor._record_health(index * 30.4, index != 60)


def test_monitor_starts_log_scan_at_canary_boundary(tmp_path):
    log_path = tmp_path / "vllm.log"
    log_path.write_text("old startup warning\n")
    monitor = RuntimeMonitor(
        base_url="http://unused",
        run_dir=tmp_path,
        vllm_log=log_path,
        interval_seconds=30,
        stall_timeout_seconds=600,
        unhealthy_window_seconds=300,
        max_failure_rate=0.05,
        min_health_failures=2,
        hard_timeout_seconds=5_400,
        started_at=0,
        io_lock=threading.Lock(),
        status_lock=threading.RLock(),
    )

    monitor._scan_log()
    assert monitor._last_coordination_warning is None


def test_monitor_preserves_partial_log_lines_across_scans(tmp_path):
    log_path = tmp_path / "vllm.log"
    log_path.write_text("")
    monitor = RuntimeMonitor(
        base_url="http://unused",
        run_dir=tmp_path,
        vllm_log=log_path,
        interval_seconds=30,
        stall_timeout_seconds=600,
        unhealthy_window_seconds=300,
        max_failure_rate=0.05,
        min_health_failures=2,
        hard_timeout_seconds=5_400,
        started_at=0,
        io_lock=threading.Lock(),
        status_lock=threading.RLock(),
    )
    with log_path.open("a") as log:
        log.write("sample_tokens RPC tim")
    monitor._scan_log()
    with log_path.open("a") as log:
        log.write("ed out after 7200 seconds\n")

    with pytest.raises(CanaryFailure, match="sample_tokens"):
        monitor._scan_log()


def test_monitor_fails_closed_when_evidence_append_crashes(tmp_path, monkeypatch):
    monitor = RuntimeMonitor(
        base_url="http://unused",
        run_dir=tmp_path,
        vllm_log=tmp_path / "vllm.log",
        interval_seconds=30,
        stall_timeout_seconds=600,
        unhealthy_window_seconds=300,
        max_failure_rate=0.05,
        min_health_failures=2,
        hard_timeout_seconds=5_400,
        started_at=1e18,
        io_lock=threading.Lock(),
        status_lock=threading.RLock(),
    )
    metrics = """\
vllm:generation_tokens_total 10
vllm:num_requests_running 1
vllm:num_requests_waiting 0
"""
    monkeypatch.setattr(
        "hpc.vllm.sustained_decode_canary._fetch_text",
        lambda *_args, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        "hpc.vllm.sustained_decode_canary._append_jsonl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(5, "I/O error")),
    )
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        monitor.run()

    assert exc_info.value.code == 2
    assert monitor.finished_cleanly is False
    assert monitor.error == "runtime monitor crashed: OSError: [Errno 5] I/O error"
    assert json.loads((tmp_path / "canary-status.json").read_text())["status"] == "CANARY_FAIL"


def test_request_completion_uses_chat_sampling_policy(monkeypatch):
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"usage": {"completion_tokens": 32}, "choices": [{"finish_reason": "length"}]}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    record = _request_completion(
        base_url="http://server",
        model="model",
        stage="depth",
        request_index=1,
        max_tokens=32,
        timeout_seconds=60,
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["messages"][0]["role"] == "user"
    assert captured["payload"]["temperature"] == 0.7
    assert captured["payload"]["top_p"] == 1.0
    assert captured["payload"]["repetition_penalty"] == 1.1
    assert record["completion_tokens"] == 32


def test_request_completion_surfaces_http_error_body(monkeypatch):
    def fake_urlopen(_request, timeout):
        assert timeout == 60
        raise urllib.error.HTTPError(
            "http://server/v1/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":"maximum context length exceeded"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(CanaryFailure, match="maximum context length exceeded"):
        _request_completion(
            base_url="http://server",
            model="model",
            stage="depth",
            request_index=1,
            max_tokens=32,
            timeout_seconds=60,
        )


def test_wrapper_canary_mode_is_default_off_and_isolated_from_policy_eval():
    source = (Path(__file__).parents[2] / "hpc/vllm/smoke_grug_vllm_tacc.sbatch").read_text()

    assert 'RUN_CANARY="${RUN_CANARY:-0}"' in source
    assert 'if [ "$RUN_CANARY" = "1" ]; then' in source
    assert 'RUN_KIND=canary' in source
    assert 'RUN_EVAL=0 and RUN_AGENTIC=0' in source
    assert 'sustained_decode_canary.py' in source
    assert "must explicitly pin MODEL_REPO and MODEL_REVISION" in source
    assert 'MAX_MODEL_LEN must fit CANARY_DEPTH_TOKENS' in source
    assert 'CANARY_CONTEXT_MARGIN="${CANARY_CONTEXT_MARGIN:-512}"' in source
    assert 'CANARY_HARD_TIMEOUT="${CANARY_HARD_TIMEOUT:-6000}"' in source
    assert '--hard-timeout "$CANARY_HARD_TIMEOUT"' in source
    assert '--depth-tokens "$CANARY_DEPTH_TOKENS"' in source
    assert 'CANARY_MAX_FAILURE_RATE must be in (0, 1]' in source
    assert 'CANARY_TIMEOUT_MULTIPLIER must be at least 1' in source
    assert 'CANARY_MIN_HEALTH_FAILURES must be a positive integer' in source
    assert '"min_health_failures": int(min_health_failures)' in source
    assert '"vllm_python_overlay": vllm_python_overlay or None' in source
    canary_invocation = source.rsplit('if [ "$RUN_CANARY" = "1" ]; then', 1)[1].split(
        'if [ "$RUN_EVAL" = "1" ]; then', 1
    )[0]
    assert "sustained_decode_canary.py" in canary_invocation
    assert "evalchemy" not in canary_invocation
