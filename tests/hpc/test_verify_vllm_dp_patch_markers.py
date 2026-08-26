from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hpc.vllm.verify_vllm_dp_patch_markers import wait_for_patch_markers


def test_waits_for_delayed_remote_rank_marker(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    log_path.write_text("VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=0\n")

    def append_remote_marker() -> None:
        time.sleep(0.05)
        with log_path.open("a") as log:
            log.write("VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=1\n")

    writer = threading.Thread(target=append_remote_marker)
    writer.start()
    try:
        assert wait_for_patch_markers(
            log_path,
            expected_size=2,
            vllm_pid=os.getpid(),
            timeout_sec=1,
            poll_interval_sec=0.01,
        ) == {0, 1}
    finally:
        writer.join()


def test_fails_closed_if_marker_never_arrives(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    log_path.write_text("VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=0\n")

    with pytest.raises(TimeoutError, match=r"expected \[0, 1\], got \[0\]"):
        wait_for_patch_markers(
            log_path,
            expected_size=2,
            vllm_pid=os.getpid(),
            timeout_sec=0.02,
            poll_interval_sec=0.005,
        )


def test_rejects_unexpected_rank_immediately(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    log_path.write_text("VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=2\n")

    with pytest.raises(RuntimeError, match=r"unexpected DP patch ranks \[2\]"):
        wait_for_patch_markers(
            log_path,
            expected_size=2,
            vllm_pid=os.getpid(),
            timeout_sec=1,
        )


def test_fails_immediately_if_vllm_process_has_exited(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    log_path.write_text("VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=0\n")
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=5)

    with pytest.raises(RuntimeError, match=r"exited while waiting"):
        wait_for_patch_markers(
            log_path,
            expected_size=2,
            vllm_pid=process.pid,
            timeout_sec=1,
            poll_interval_sec=0.01,
        )
