"""Cached bytes must agree with published revision metadata before reuse."""

import hashlib
from types import SimpleNamespace

import pytest

from eval.autoeval32k.checkpoint import verified_cache


def cache_inventory(tmp_path):
    revision = "a" * 40
    cache = tmp_path / "cache"
    source = cache / "models--Qwen--test" / "snapshots" / revision
    source.mkdir(parents=True)
    files = {
        "config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": b"weights",
    }
    entries = []
    for name, content in files.items():
        (source / name).write_bytes(content)
        lfs = name.endswith(".safetensors")
        sha = (
            hashlib.sha256(content).hexdigest()
            if lfs
            else hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        )
        entries.append(
            SimpleNamespace(
                rfilename=name,
                size=len(content),
                lfs=SimpleNamespace(sha256=sha) if lfs else None,
                blob_id=None if lfs else sha,
            )
        )
    return cache, source, SimpleNamespace(sha=revision, siblings=entries)


def test_verified_cache_reuses_exact_bytes_without_changing_source(tmp_path):
    cache, source, info = cache_inventory(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    dest, records = verified_cache(
        "Qwen/test", info.sha, cache, tmp_path / "new", info=info
    )
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert len(records) == 3


@pytest.mark.parametrize("name", ["config.json", "model.safetensors"])
def test_corrupt_cached_byte_is_rejected_before_staging(tmp_path, name):
    cache, source, info = cache_inventory(tmp_path)
    content = (source / name).read_bytes()
    (source / name).write_bytes(b"X" + content[1:])
    with pytest.raises(ValueError, match="content mismatch"):
        verified_cache("Qwen/test", info.sha, cache, tmp_path / "new", info=info)
    assert not (tmp_path / "new").exists()


def test_wrong_published_revision_rejects_cache(tmp_path):
    cache, _, info = cache_inventory(tmp_path)
    with pytest.raises(ValueError, match="revision mismatch"):
        verified_cache("Qwen/test", "b" * 40, cache, tmp_path / "new", info=info)
