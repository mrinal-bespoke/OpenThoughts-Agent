import hashlib
import json

import pytest

from hpc.vllm.write_eval_run_manifest import ManifestError, write_manifest


def test_writes_hashes_settings_and_overlay_manifest(tmp_path):
    wrapper = tmp_path / "wrapper.sbatch"
    wrapper.write_text("#!/bin/bash\n")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"patched_core_sha256": "abc"}))
    output = tmp_path / "run" / "manifest.json"

    assert (
        write_manifest(
            output=output,
            files=[f"wrapper={wrapper}"],
            settings=["MODEL_REPO=laion/model", "ENABLE_THINKING=1"],
            overlay_manifest=overlay,
        )
        == output
    )
    document = json.loads(output.read_text())
    assert document["files"]["wrapper"]["sha256"] == hashlib.sha256(
        wrapper.read_bytes()
    ).hexdigest()
    assert document["settings"] == {
        "ENABLE_THINKING": "1",
        "MODEL_REPO": "laion/model",
    }
    assert document["overlay_manifest"]["patched_core_sha256"] == "abc"
    assert document["created_at"]


@pytest.mark.parametrize(
    "files,settings,message",
    [
        (["bad"], [], "malformed file"),
        ([], ["bad"], "malformed setting"),
        (["x=/missing"], [], "cannot hash"),
    ],
)
def test_rejects_malformed_or_missing_inputs(tmp_path, files, settings, message):
    with pytest.raises(ManifestError, match=message):
        write_manifest(output=tmp_path / "manifest.json", files=files, settings=settings)


def test_rejects_duplicate_labels_and_settings(tmp_path):
    source = tmp_path / "source"
    source.write_text("x")
    with pytest.raises(ManifestError, match="duplicate file"):
        write_manifest(
            output=tmp_path / "manifest.json",
            files=[f"x={source}", f"x={source}"],
            settings=[],
        )
    with pytest.raises(ManifestError, match="duplicate setting"):
        write_manifest(
            output=tmp_path / "manifest.json",
            files=[],
            settings=["X=1", "X=2"],
        )
