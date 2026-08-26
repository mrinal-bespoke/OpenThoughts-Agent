import pytest

from hpc.vllm.build_vllm_dp_coordination_overlay import (
    NEW_IDLE_BLOCK,
    NEW_LOOP_HEADER,
    NEW_SYNC_BLOCK,
    OLD_IDLE_BLOCK,
    OLD_LOOP_HEADER,
    OLD_SYNC_BLOCK,
    build_overlay,
    patch_core_text,
)


def _core_source() -> str:
    return (
        "class FakeEngine:\n"
        + OLD_LOOP_HEADER
        + "        while True:\n"
        + "            executed = True\n"
        + "            local_unfinished_reqs = True\n"
        + OLD_IDLE_BLOCK
        + "            break\n\n"
        + "    def _has_global_unfinished_reqs(self, local_unfinished):\n"
        + OLD_SYNC_BLOCK
        + "            self.dp_group, has_unfinished=local_unfinished, pending_pause=False\n"
        + "        )\n"
        + "        return has_unfinished\n"
    )


def test_patch_removes_idle_skip_and_32_step_stale_state():
    patched = patch_core_text(_core_source())

    assert OLD_IDLE_BLOCK not in patched
    assert OLD_SYNC_BLOCK not in patched
    assert NEW_IDLE_BLOCK in patched
    assert NEW_SYNC_BLOCK in patched
    assert NEW_LOOP_HEADER in patched
    assert "VLLM_DP_COORDINATION_PATCH_ACTIVE" in patched
    assert "step_counter % 32" not in patched


def test_patch_fails_closed_on_an_unknown_engine_layout():
    with pytest.raises(ValueError, match="idle block"):
        patch_core_text("class EngineCore: pass\n")


def test_overlay_only_copies_the_patched_core(tmp_path):
    source_package = tmp_path / "installed" / "vllm"
    engine = source_package / "v1" / "engine"
    engine.mkdir(parents=True)
    (source_package / "__init__.py").write_text("")
    (source_package / "binary.so").write_bytes(b"binary")
    (source_package / "__pycache__").mkdir()
    (source_package / "v1" / "__init__.py").write_text("")
    (source_package / "v1" / "__pycache__").mkdir()
    (engine / "__init__.py").write_text("")
    (engine / "__pycache__").mkdir()
    (engine / "helper.py").write_text("VALUE = 1\n")
    (engine / "core.py").write_text(_core_source())

    overlay = tmp_path / "overlay"
    core = build_overlay(source_package, overlay)

    assert core.is_file() and not core.is_symlink()
    assert (overlay / "vllm" / "binary.so").is_symlink()
    assert (overlay / "vllm" / "v1" / "engine" / "helper.py").is_symlink()
    assert not (overlay / "vllm" / "__pycache__").exists()
    assert not (overlay / "vllm" / "v1" / "__pycache__").exists()
    assert not (overlay / "vllm" / "v1" / "engine" / "__pycache__").exists()
    assert NEW_IDLE_BLOCK in core.read_text()
    assert build_overlay(source_package, overlay) == core


def test_overlay_rejects_source_drift(tmp_path):
    source_package = tmp_path / "installed" / "vllm"
    engine = source_package / "v1" / "engine"
    engine.mkdir(parents=True)
    (source_package / "__init__.py").write_text("")
    (source_package / "v1" / "__init__.py").write_text("")
    (engine / "__init__.py").write_text("")
    source_core = engine / "core.py"
    source_core.write_text(_core_source())
    overlay = tmp_path / "overlay"
    build_overlay(source_package, overlay)

    source_core.write_text(_core_source() + "# changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        build_overlay(source_package, overlay)


def test_overlay_allows_local_pycache_but_rejects_a_shared_symlink(tmp_path):
    source_package = tmp_path / "installed" / "vllm"
    engine = source_package / "v1" / "engine"
    engine.mkdir(parents=True)
    (source_package / "__init__.py").write_text("")
    (source_package / "v1" / "__init__.py").write_text("")
    (engine / "__init__.py").write_text("")
    (engine / "core.py").write_text(_core_source())
    overlay = tmp_path / "overlay"
    build_overlay(source_package, overlay)

    cache = overlay / "vllm" / "v1" / "engine" / "__pycache__"
    cache.mkdir()
    assert build_overlay(source_package, overlay).is_file()

    cache.rmdir()
    shared_cache = engine / "__pycache__"
    shared_cache.mkdir()
    cache.symlink_to(shared_cache)
    with pytest.raises(RuntimeError, match="shared __pycache__"):
        build_overlay(source_package, overlay)
