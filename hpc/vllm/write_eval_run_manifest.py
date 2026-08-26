#!/usr/bin/env python3
"""Write a hash-pinned manifest for one Vista evaluation allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class ManifestError(RuntimeError):
    """Raised when manifest inputs are incomplete or ambiguous."""


def _split(value: str, kind: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or not item:
        raise ManifestError(f"malformed {kind}: {value}")
    return key, item


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ManifestError(f"cannot hash {path}: {error}") from error


def write_manifest(
    *,
    output: Path,
    files: list[str],
    settings: list[str],
    overlay_manifest: Path | None = None,
) -> Path:
    file_hashes = {}
    for raw in files:
        label, value = _split(raw, "file")
        if label in file_hashes:
            raise ManifestError(f"duplicate file label: {label}")
        path = Path(value).resolve()
        file_hashes[label] = {"path": str(path), "sha256": _sha256(path)}

    resolved_settings = {}
    for raw in settings:
        key, value = _split(raw, "setting")
        if key in resolved_settings:
            raise ManifestError(f"duplicate setting: {key}")
        resolved_settings[key] = value

    document: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": file_hashes,
        "settings": resolved_settings,
    }
    if overlay_manifest is not None:
        try:
            overlay = json.loads(overlay_manifest.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(
                f"cannot read overlay manifest {overlay_manifest}: {error}"
            ) from error
        if not isinstance(overlay, dict):
            raise ManifestError("overlay manifest must be a JSON object")
        document["overlay_manifest"] = overlay

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--overlay-manifest", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = write_manifest(
        output=args.output,
        files=args.file,
        settings=args.setting,
        overlay_manifest=args.overlay_manifest,
    )
    print(f"EVAL_RUN_MANIFEST_OK path={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
