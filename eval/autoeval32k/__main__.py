"""Explicit checkpoint-manifest entry point. Planning is read-only by default."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from .protocol import Manifest, render_harbor, token
from .queue import submit_once
from .runtime import write_new


def validate_settings(settings):
    required = {
        "cluster",
        "account",
        "partition",
        "python",
        "serve_python",
        "repo_root",
        "harbor_root",
        "run_root",
        "state_root",
        "walltime_seconds",
        "concurrency",
    }
    optional = {"credential_file"}
    if (
        not isinstance(settings, dict)
        or not required <= settings.keys()
        or settings.keys() - required - optional
    ):
        raise ValueError("Missing or unknown deployment settings")
    if settings["cluster"] != "tacc":
        raise ValueError("Only TACC is implemented; Jupiter awaits the TACC gate")
    for field in ("account", "partition"):
        token(settings[field], r"[A-Za-z0-9][A-Za-z0-9_.-]*", field)
    for field in required - {
        "cluster",
        "account",
        "partition",
        "walltime_seconds",
        "concurrency",
    } | (optional & settings.keys()):
        value = settings[field]
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or ".." in Path(value).parts
            or any(c in value for c in "\n\r\x00")
        ):
            raise ValueError(f"{field} must be an absolute normalized path")
    if (
        type(settings["walltime_seconds"]) is not int
        or not 60 <= settings["walltime_seconds"] <= 172800
    ):
        raise ValueError("Allocation must be bounded to at most 48 hours")
    if type(settings["concurrency"]) is not int or settings["concurrency"] != 1:
        raise ValueError("Concurrency >1 awaits measured TACC qualification")
    return dict(settings)


def plan(manifest, settings):
    settings = validate_settings(settings)
    root = Path(settings["run_root"])
    run_dir = root / manifest.run_id
    # A separate immutable control bundle allows runtime mkdir to refuse resumes.
    bundle = Path(settings["state_root"]) / "bundles" / manifest.run_id
    argv = [
        settings["python"],
        "-m",
        "eval.autoeval32k.runtime",
        "--manifest",
        str(bundle / "manifest.json"),
        "--settings",
        str(bundle / "settings.json"),
        "--run-dir",
        str(run_dir),
    ]
    seconds = settings["walltime_seconds"]
    walltime = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    command = [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--export=NONE",
        "--account=" + settings["account"],
        "--partition=" + settings["partition"],
        "--nodes=1",
        "--ntasks-per-node=1",
        "--time=" + walltime,
        "--job-name=" + manifest.run_id,
        "--comment=" + manifest.run_id,
        "--signal=B:TERM@60",
        "--chdir=" + str(bundle / "code"),
        "--output=" + str(bundle / "slurm-%j.log"),
        "--wrap=exec " + shlex.join(argv),
    ]
    return {
        "run_id": manifest.run_id,
        "manifest": manifest.as_dict(),
        "settings": settings,
        "bundle": str(bundle),
        "run_dir": str(run_dir),
        "command": command,
        "code_sha256": source_hashes(),
        "harbor_config": render_harbor(manifest, run_dir),
    }


def source_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def materialize(planned):
    bundle = Path(planned["bundle"])
    payloads = {
        "manifest.json": planned["manifest"],
        "settings.json": planned["settings"],
        "code-lock.json": planned["code_sha256"],
    }
    if bundle.exists():
        if bundle.is_symlink() or any(
            not (bundle / name).is_file()
            or (bundle / name).is_symlink()
            or json.loads((bundle / name).read_text()) != value
            for name, value in payloads.items()
        ):
            raise ValueError(
                "Existing bundle is incomplete or differs; reconcile, do not overwrite"
            )
        code = bundle / "code" / "eval" / "autoeval32k"
        if {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code.glob("*.py")
        } != planned["code_sha256"]:
            raise ValueError("Frozen execution source changed")
        return
    bundle.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        write_new(bundle / name, value)
    code = bundle / "code" / "eval" / "autoeval32k"
    code.mkdir(parents=True)
    (code.parent / "__init__.py").touch(exist_ok=False)
    for name, expected in planned["code_sha256"].items():
        source = Path(__file__).parent / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Source changed while freezing execution bundle")
        shutil.copyfile(source, code / name)
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Execution source copy differs")


def query_queue():
    result = subprocess.run(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%r"],
        check=True,
        capture_output=True,
        text=True,
        timeout=40,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            raise ValueError("Malformed all-user scheduler response")
        rows.append(dict(zip(("job_id", "state", "reason"), parts)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Explicitly reserve and submit once (default: dry-run)",
    )
    args = parser.parse_args(argv)
    manifest = Manifest.parse(json.loads(args.manifest.read_text()))
    planned = plan(manifest, json.loads(args.settings.read_text()))
    if not args.submit:
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 0
    # This command is run on the target login host, never through implicit SSH.
    if Path.cwd().resolve() != Path(planned["settings"]["repo_root"]).resolve():
        raise ValueError("Submission must run from configured deployed checkout")
    materialize(planned)
    receipt = submit_once(
        Path(planned["settings"]["state_root"]),
        manifest.run_id,
        planned["command"],
        query_queue=query_queue,
        submit=lambda command: (
            subprocess.run(
                command, timeout=40, check=True, capture_output=True, text=True
            ).stdout
        ),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
