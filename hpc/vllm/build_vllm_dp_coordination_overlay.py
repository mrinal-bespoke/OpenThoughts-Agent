#!/usr/bin/env python3
"""Build an isolated vLLM package overlay with the DP coordination fix.

The shared vLLM environment is left untouched. The overlay symlinks every
package entry except ``vllm/v1/engine/core.py``, which is copied and patched.
Prepending the overlay root to ``PYTHONPATH`` makes Ray workers use the fixed
engine loop while retaining the exact installed binaries and Python modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
from pathlib import Path


OLD_IDLE_BLOCK = """            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                with self.log_iteration_details(None):
                    self.execute_dummy_batch()
"""

NEW_IDLE_BLOCK = """            if not executed:
                # Every idle DP rank must participate in the dummy forward and
                # coordinate_batch_across_dp. Skipping a fully idle rank can
                # strand its peer in the collective indefinitely.
                with self.log_iteration_details(None):
                    self.execute_dummy_batch()
"""

OLD_SYNC_BLOCK = """        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
"""

NEW_SYNC_BLOCK = """        # Synchronize every step so engines_running cannot remain stale while
        # another rank becomes idle between the old 32-step checkpoints.
        self.step_counter += 1

        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
"""

OLD_LOOP_HEADER = """    def run_busy_loop(self):
        \"\"\"Core busy loop of the EngineCore for data parallel case.\"\"\"

        # Loop until process is sent a SIGINT or SIGTERM
"""

NEW_LOOP_HEADER = """    def run_busy_loop(self):
        \"\"\"Core busy loop of the EngineCore for data parallel case.\"\"\"

        logger.warning(\"VLLM_DP_COORDINATION_PATCH_ACTIVE dp_rank=%s\", self.dp_rank)
        # Loop until process is sent a SIGINT or SIGTERM
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_core_text(source: str) -> str:
    """Apply the two upstream DP coordination changes, failing closed."""
    if (
        NEW_IDLE_BLOCK in source
        and NEW_SYNC_BLOCK in source
        and NEW_LOOP_HEADER in source
    ):
        return source
    if source.count(OLD_IDLE_BLOCK) != 1:
        raise ValueError("expected exactly one unpatched DP idle block")
    if source.count(OLD_SYNC_BLOCK) != 1:
        raise ValueError("expected exactly one unpatched 32-step DP sync block")
    if source.count(OLD_LOOP_HEADER) != 1:
        raise ValueError("expected exactly one unpatched DP busy-loop header")
    return (
        source.replace(OLD_IDLE_BLOCK, NEW_IDLE_BLOCK)
        .replace(OLD_SYNC_BLOCK, NEW_SYNC_BLOCK)
        .replace(OLD_LOOP_HEADER, NEW_LOOP_HEADER)
    )


def _symlink_children(source: Path, destination: Path, excluded: set[str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        if child.name not in excluded:
            (destination / child.name).symlink_to(child)


def build_overlay(source_package: Path, overlay_root: Path) -> Path:
    source_package = source_package.resolve()
    if source_package.name != "vllm":
        raise ValueError(f"source package must be named vllm: {source_package}")
    source_core = source_package / "v1" / "engine" / "core.py"
    if not source_core.is_file():
        raise FileNotFoundError(source_core)

    destination_package = overlay_root / "vllm"
    destination_core = destination_package / "v1" / "engine" / "core.py"
    manifest_path = overlay_root / "manifest.json"
    patched = patch_core_text(source_core.read_text())

    if overlay_root.exists():
        if not manifest_path.is_file() or not destination_core.is_file():
            raise FileExistsError(f"refusing to reuse incomplete overlay: {overlay_root}")
        cache_paths = [
            destination_package / "__pycache__",
            destination_package / "v1" / "__pycache__",
            destination_package / "v1" / "engine" / "__pycache__",
        ]
        if any(path.is_symlink() for path in cache_paths):
            raise RuntimeError("overlay contains a shared __pycache__ symlink")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_core_sha256") != _sha256(source_core.read_bytes()):
            raise RuntimeError("installed vLLM core changed since overlay creation")
        if destination_core.read_text() != patched:
            raise RuntimeError("existing overlay does not contain the expected patch")
        return destination_core

    overlay_root.mkdir(parents=True)
    _symlink_children(source_package, destination_package, {"v1", "__pycache__"})
    _symlink_children(
        source_package / "v1",
        destination_package / "v1",
        {"engine", "__pycache__"},
    )
    _symlink_children(
        source_package / "v1" / "engine",
        destination_package / "v1" / "engine",
        {"core.py", "__pycache__"},
    )
    destination_core.write_text(patched)
    compile_probe = overlay_root / ".core.pyc.check"
    py_compile.compile(str(destination_core), cfile=str(compile_probe), doraise=True)
    compile_probe.unlink()
    manifest_path.write_text(
        json.dumps(
            {
                "patch": "vllm-dp-coordination-deadlock-pr12",
                "source_package": str(source_package),
                "source_core_sha256": _sha256(source_core.read_bytes()),
                "patched_core_sha256": _sha256(destination_core.read_bytes()),
            },
            indent=2,
        )
        + "\n"
    )
    return destination_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    args = parser.parse_args()
    core = build_overlay(args.source_package, args.overlay_root)
    print(f"VLLM_DP_COORDINATION_OVERLAY_OK core={core}")


if __name__ == "__main__":
    main()
