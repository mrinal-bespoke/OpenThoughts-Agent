import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from lifecycle import validate_lifetime

root = Path(__file__).resolve().parent
os.umask(0o077)
assert json.loads((root / "proxy-login-check.json").read_text())["status"] == 200
validate_lifetime(json.loads((root / "proxy-ready.json").read_text()), time.time())
assert (root / "proxychains.conf").is_file()
with (root / "submission-intent.json").open("x") as f:
    json.dump(
        {
            "epoch": time.time(),
            "hashes": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "smoke.sbatch",
                    "smoke.py",
                    "connect_proxy.py",
                    "prepare.py",
                    "lifecycle.py",
                    "scope.json",
                )
            },
        },
        f,
        indent=2,
    )
result = subprocess.run(
    ["sbatch", "--parsable", "smoke.sbatch"],
    cwd=root,
    capture_output=True,
    text=True,
    timeout=20,
)
receipt = {
    "exit_code": result.returncode,
    "stdout": result.stdout.strip(),
    "stderr": result.stderr.strip(),
}
with (root / "submission.json").open("x") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt), flush=True)
raise SystemExit(result.returncode)
