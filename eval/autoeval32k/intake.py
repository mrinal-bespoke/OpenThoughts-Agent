"""Freeze an explicit published checkpoint; no model discovery or evaluation launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .protocol import Manifest, token
from .runtime import write_new


def read_template(path):
    """Freeze explicit Qwen3 template bytes without importing a GPU stack."""
    path = Path(path)
    if (path / "chat_template.jinja").is_file():
        template = (path / "chat_template.jinja").read_text()
    else:
        template = json.loads((path / "tokenizer_config.json").read_text()).get(
            "chat_template"
        )
    if not isinstance(template, str) or not template:
        raise ValueError("A single explicit tokenizer template is required")
    return template


def freeze_checkpoint(spec, repo, revision, thinking, cache_dir, *, resolve=None):
    """Resolve metadata into a fresh private cache and return an immutable manifest.

    The suite specification supplies the reviewed task/snapshot inventory and
    budgets. Only metadata/tokenizer files are downloaded here, not weights.
    """
    token(revision, r"[0-9a-f]{40}", "immutable checkpoint revision")
    token(repo, r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "checkpoint repo")
    if "model" in spec:
        raise ValueError("Suite specification must not contain a model override")
    model = {
        "repo": repo,
        "revision": revision,
        "family": "qwen3",
        "thinking": thinking,
        "template_sha256": "0" * 64,
    }
    Manifest.parse(
        dict(spec, model=model)
    )  # reject invalid input before network/cache writes
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)
    if resolve is None:
        from huggingface_hub import HfApi, snapshot_download

        def resolve(repo, revision, cache_dir):
            info = HfApi().model_info(repo, revision=revision)
            if info.sha != revision:
                raise ValueError(
                    "Published checkpoint resolved to a different revision"
                )
            path = snapshot_download(
                repo_id=repo,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[
                    "config.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                ],
            )
            config = json.loads((Path(path) / "config.json").read_text())
            return info.sha, config, read_template(path)

    sha, config, template = resolve(repo, revision, cache_dir)
    if sha != revision or config.get("model_type") != "qwen3":
        raise ValueError("Checkpoint revision/architecture is not supported")
    if not isinstance(template, str) or not template:
        raise ValueError("An explicit tokenizer chat template is required")
    model["template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    return Manifest.parse(dict(spec, model=model))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--thinking", choices=("true", "false"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing manifest")
    manifest = freeze_checkpoint(
        json.loads(args.spec.read_text()),
        args.checkpoint,
        args.revision,
        args.thinking == "true",
        args.cache_dir,
    )
    write_new(args.output, manifest.as_dict())
    print(manifest.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
