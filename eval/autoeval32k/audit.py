"""Read-only normalized outcome audit; never changes or repairs a raw score."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def audit_results(manifest, paths):
    expected = {t["id"] for t in manifest.as_dict()["tasks"]}
    outcomes, hashes, errors = {}, {}, Counter()
    for path in map(Path, paths):
        raw = path.read_bytes()
        record = json.loads(raw)
        if (
            record.get("run_id") != manifest.run_id
            or type(record.get("attempt")) is not int
            or record["attempt"] != 0
        ):
            raise ValueError("Run/attempt identity mismatch")
        task = record.get("task_id")
        if task not in expected:
            raise ValueError("Unknown task")
        if task in outcomes:
            raise ValueError("Duplicate task outcome")
        reward = record.get("reward")
        if "reward" not in record or (
            reward is not None
            and (
                type(reward) not in (int, float)
                or not math.isfinite(reward)
                or reward not in (0, 1)
            )
        ):
            raise ValueError("Expected finite binary reward or explicit null")
        error = record.get("error")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("Malformed error")
        outcomes[task] = reward
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        if error:
            errors[error] += 1
    numeric = [r for r in outcomes.values() if r is not None]
    missing = sorted(expected - outcomes.keys())
    return {
        "run_id": manifest.run_id,
        "planned": len(expected),
        "terminal": len(outcomes),
        "numeric": len(numeric),
        "zero": numeric.count(0),
        "successes": numeric.count(1),
        "unscored": len(outcomes) - len(numeric),
        "missing": missing,
        "complete": not missing and len(numeric) == len(expected),
        "mean_numeric": sum(numeric) / len(numeric) if numeric else None,
        "successes_over_planned": numeric.count(1) / len(expected),
        "errors": dict(errors),
        "raw_sha256": hashes,
    }
