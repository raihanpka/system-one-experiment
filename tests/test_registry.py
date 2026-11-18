# prompt templates: file overrides, double-brace variables, and the MLflow URI adapter

import json

import pytest

from column_mapping.knowledge import Knowledge
from column_mapping.registry import ChainRegistry, FileRegistry
from column_mapping.registry_mlflow import MlflowRegistry


def test_file_registry_formats_double_brace_variables_and_overlays():
    base = FileRegistry({"bind": "Field {{field}} needs {{role}}"})
    assert base.get("bind").format(field="total", role="fee") == "Field total needs fee"
    over = base.overlay({"bind": "{{role}} for {{field}}"})
    assert over.get("bind").format(field="total", role="fee") == "fee for total"
    assert base.get("bind").format(field="x", role="y") == "Field x needs y"
    with pytest.raises(KeyError):
        base.get("missing")


def test_a_knowledge_folder_can_override_prompts(tmp_path):
    (tmp_path / "prompts.json").write_text(
        json.dumps({"bind": "Which column is the {{role}} of {{field}}?"}),
        encoding="utf-8",
    )
    reg = FileRegistry.load(tmp_path)
    assert reg.get("bind").format(field="f", role="r") == "Which column is the r of f?"
    assert FileRegistry.load(tmp_path / "nothing").names == []


class FakePrompt:
    name = "column_mapping/presence"
    template = "Does {{field}} exist?"
    version = 3


def test_mlflow_registry_builds_version_and_alias_uris_and_keeps_the_version():
    seen = []

    def loader(uri):
        seen.append(uri)
        return FakePrompt()

    reg = MlflowRegistry(prefix="column_mapping", version="production", loader=loader)
    prompt = reg.get("presence")
    assert seen[-1] == "prompts:/column_mapping/presence@production"
    assert prompt.template == "Does {{field}} exist?" and prompt.version == "3"

    reg.get("presence", version="7")
    assert seen[-1] == "prompts:/column_mapping/presence/7"
    reg.get("presence", version="staging")
    assert seen[-1] == "prompts:/column_mapping/presence@staging"

    latest = MlflowRegistry(prefix="column_mapping", loader=loader)
    latest.get("presence")
    assert seen[-1] == "prompts:/column_mapping/presence@latest"


def test_a_missing_mlflow_prompt_falls_back_and_a_real_error_does_not():
    class Missing(Exception):
        pass

    def missing(uri):
        raise Missing("Prompt with name column_mapping/bind not found")

    reg = ChainRegistry(
        MlflowRegistry(loader=missing), FileRegistry({"bind": "fallback"})
    )
    assert reg.get("bind").template == "fallback"

    def broken(uri):
        raise RuntimeError("connection refused")

    with pytest.raises(RuntimeError):
        MlflowRegistry(loader=broken).get("bind")


def test_mlflow_chat_prompts_are_rejected_with_a_clear_message():
    class Chat:
        template = [{"role": "user", "content": "x"}]
        version = 1

    reg = MlflowRegistry(loader=lambda uri: Chat())
    with pytest.raises(ValueError, match="chat prompt"):
        reg.get("bind")


def test_a_knowledge_folder_carries_its_prompt_overrides_with_the_dataset(tmp_path):
    (tmp_path / "outputs.json").write_text(
        json.dumps([{"name": "x", "type": "string", "description": "X"}]),
        encoding="utf-8",
    )
    (tmp_path / "prompts.json").write_text(
        json.dumps({"bind": "{{role}} of {{field}}"}), encoding="utf-8"
    )
    knowledge = Knowledge.load(tmp_path)
    assert dict(knowledge.prompts)["bind"] == "{{role}} of {{field}}"
    reg = knowledge.registry()
    assert reg.get("bind").format(field="f", role="r") == "r of f"
    assert reg.get("presence").template  # the built-in default still fills the rest

    out = tmp_path / "saved"
    knowledge.save(out)
    assert Knowledge.load(out) == knowledge
    assert out.joinpath("prompts.json").exists()
