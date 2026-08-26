#!/usr/bin/env python3
"""Wait for every vLLM DP rank to report the coordination patch marker."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path


MARKER_RE = re.compile(r"VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=(\d+)")


def _process_is_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return state != "Z"


def wait_for_patch_markers(
    log_path: Path,
    expected_size: int,
    vllm_pid: int,
    *,
    timeout_sec: float = 60.0,
    poll_interval_sec: float = 0.25,
) -> set[int]:
    """Return observed DP ranks after all expected markers reach the log.

    Ray forwards remote actor logs asynchronously, so a healthy rank can emit
    its marker before the line becomes visible in the controller's log file.
    Keep the integrity check fail-closed, but allow that bounded forwarding
    delay while ensuring the serving process remains alive.
    """

    expected = set(range(expected_size))
    deadline = time.monotonic() + timeout_sec
    observed: set[int] = set()

    while True:
        try:
            text = log_path.read_text()
        except FileNotFoundError:
            text = ""
        observed = {int(value) for value in MARKER_RE.findall(text)}
        unexpected = observed - expected
        if unexpected:
            raise RuntimeError(
                f"unexpected DP patch ranks {sorted(unexpected)}; "
                f"expected {sorted(expected)}"
            )
        if observed == expected:
            return observed
        if not _process_is_alive(vllm_pid):
            raise RuntimeError(
                f"vLLM process {vllm_pid} exited while waiting for DP patch "
                f"markers; expected {sorted(expected)}, got {sorted(observed)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for DP patch markers; "
                f"expected {sorted(expected)}, got {sorted(observed)}"
            )
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--vllm-pid", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    args = parser.parse_args()

    if args.expected_size < 1:
        raise SystemExit("FATAL: expected-size must be positive")

    try:
        ranks = wait_for_patch_markers(
            args.log_path,
            args.expected_size,
            args.vllm_pid,
            timeout_sec=args.timeout_sec,
        )
    except (RuntimeError, TimeoutError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(f"VLLM_DP_COORDINATION_PATCH_VERIFIED ranks={sorted(ranks)}")


if __name__ == "__main__":
    main()
