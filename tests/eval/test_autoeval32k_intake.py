"""Published checkpoint resolution is the only simulated service boundary."""

import hashlib
import json

import pytest

from eval.autoeval32k.intake import freeze_checkpoint, read_template
from tests.eval import test_autoeval32k


@pytest.fixture
def spec():
    data = test_autoeval32k.example_manifest()
    del data["model"]
    return data


def test_explicit_template_file_takes_precedence(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "old"})
    )
    assert read_template(tmp_path) == "old"
    (tmp_path / "chat_template.jinja").write_text("published template\n")
    assert read_template(tmp_path) == "published template\n"


def test_ambiguous_template_mapping_is_rejected(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": {"one": "x", "two": "y"}})
    )
    with pytest.raises(ValueError, match="single explicit"):
        read_template(tmp_path)


def test_published_pin_and_template_are_frozen(spec, tmp_path):
    manifest = freeze_checkpoint(
        spec,
        "Qwen/Qwen3-8B",
        "a" * 40,
        False,
        tmp_path / "new-cache",
        resolve=lambda *args: ("a" * 40, {"model_type": "qwen3"}, "template bytes"),
    )
    assert (
        manifest.as_dict()["model"]["template_sha256"]
        == hashlib.sha256(b"template bytes").hexdigest()
    )
    assert manifest.as_dict()["model"]["revision"] == "a" * 40


@pytest.mark.parametrize(
    "resolved",
    [
        ("b" * 40, {"model_type": "qwen3"}, "template"),
        ("a" * 40, {"model_type": "qwen3_moe"}, "template"),
        ("a" * 40, {"model_type": "qwen3"}, None),
    ],
)
def test_unsupported_checkpoint_is_rejected(spec, tmp_path, resolved):
    with pytest.raises(ValueError):
        freeze_checkpoint(
            spec,
            "org/model",
            "a" * 40,
            False,
            tmp_path / "new-cache",
            resolve=lambda *args: resolved,
        )


def test_floating_revision_rejected_without_cache_or_network(spec, tmp_path):
    with pytest.raises(ValueError):
        freeze_checkpoint(
            spec,
            "org/model",
            "main",
            False,
            tmp_path / "new-cache",
            resolve=lambda *args: pytest.fail("must not resolve"),
        )
    assert not (tmp_path / "new-cache").exists()


def test_existing_cache_is_untouched(spec, tmp_path):
    sentinel = tmp_path / "cached-result"
    sentinel.write_text("protected")
    with pytest.raises(FileExistsError):
        freeze_checkpoint(
            spec,
            "org/model",
            "a" * 40,
            False,
            tmp_path,
            resolve=lambda *args: pytest.fail("must not resolve"),
        )
    assert sentinel.read_text() == "protected"
