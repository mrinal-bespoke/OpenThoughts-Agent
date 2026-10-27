"""Read-only reuse of checkpoint bytes verified against published HF metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_file(path, *, size, sha, lfs):
    if path.stat().st_size != size:
        raise ValueError("Checkpoint file size mismatch")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha:
        raise ValueError("Checkpoint file content mismatch")


def verified_cache(repo, revision, cache, destination, *, info):
    """Never trust cache names alone, and never write to the source cache."""
    if info.sha != revision:
        raise ValueError("Published checkpoint revision mismatch")
    source = (
        Path(cache) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    )
    selected = [
        f
        for f in info.siblings
        if f.rfilename.endswith((".safetensors", ".json", ".jinja"))
        or f.rfilename in {"merges.txt", "vocab.txt", "tokenizer.model"}
    ]
    names = {f.rfilename for f in selected}
    if not {"config.json", "tokenizer_config.json"} <= names or not any(
        name.endswith(".safetensors") for name in names
    ):
        raise ValueError("Incomplete published checkpoint inventory")
    records = []
    for entry in selected:
        relative = Path(entry.rfilename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid published file path")
        path = source / relative
        lfs = entry.lfs
        sha = lfs.sha256 if lfs else entry.blob_id
        verify_file(path, size=entry.size, sha=sha, lfs=bool(lfs))
        records.append({"file": entry.rfilename, "sha": sha, "lfs": bool(lfs)})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for record in records:
        target = destination / record["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((source / record["file"]).resolve())
    return destination, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    model = json.loads(args.manifest.read_text())["model"]
    private_cache = args.run_dir / "hf-cache"
    records = []
    if args.cache:
        info = HfApi().model_info(
            model["repo"], revision=model["revision"], files_metadata=True
        )
        destination = (
            private_cache
            / ("models--" + model["repo"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
        )
        snapshot, records = verified_cache(
            model["repo"], model["revision"], args.cache, destination, info=info
        )
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=model["repo"],
                revision=model["revision"],
                cache_dir=private_cache,
            )
        )
    if json.loads((snapshot / "config.json").read_text()).get("model_type") != "qwen3":
        raise ValueError("Unsupported checkpoint architecture")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    template = tokenizer.chat_template
    if (
        not isinstance(template, str)
        or hashlib.sha256(template.encode()).hexdigest() != model["template_sha256"]
    ):
        raise ValueError("Tokenizer template differs from manifest")
    result = {
        "snapshot": str(snapshot),
        "repo": model["repo"],
        "revision": model["revision"],
        "verified_files": records,
    }
    with (args.run_dir / "checkpoint.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"snapshot": str(snapshot)}))


if __name__ == "__main__":
    main()
