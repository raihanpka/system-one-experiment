import json

import pytest

from columnmap.core.paths import SETTINGS
from columnmap.core.settings import load_dotenv, load_settings, model_options, openrouter_settings, parse

BASE = {
    "Backend": {"Type": "openrouter"},
    "OpenRouter": {"Model": "~liquid/d1-ultra-latest", "BaseUrl": "https://openrouter.ai/api/alpha/decisions", "ApiKeyEnv": "OPENROUTER_API_KEY"},
    "Local": {"Quant": "Q8_0", "Ngl": 999, "FlashAttn": "default"},
    "Device": {"Mode": "auto"},
    "Run": {"KeepD": 0.0, "TargetPrecision": 1.0, "PromptVersion": "v1", "Mode": "options", "Seed": 1234, "Conds": ["name", "samples"]},
    "Splits": {"Dev": [11], "Calibration": [12, 13], "Test": [21, 22]},
}


def with_(**sections):
    d = json.loads(json.dumps(BASE))
    for k, v in sections.items():
        d[k] = v
    return d


def test_committed_defaults_load_and_hosted_openrouter_is_the_default():
    cfg = load_settings(env={})
    assert cfg["backend"]["type"] == "openrouter"
    assert cfg["open_router"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert cfg["local"]["quant"] == "Q8_0" and cfg["splits"]["dev"] == [11]


def test_env_override_wins_over_the_file_and_is_parsed_as_json():
    cfg = parse(BASE, env={"COLUMNMAP__Backend__Type": "local", "COLUMNMAP__Run__Seed": "7", "COLUMNMAP__Run__Conds": '["name"]'})
    assert cfg["backend"]["type"] == "local" and cfg["run"]["seed"] == 7 and cfg["run"]["conds"] == ["name"]


def test_local_file_overrides_defaults_without_touching_the_committed_file(tmp_path):
    local = tmp_path / "appsettings.local.json"
    local.write_text(json.dumps({"Backend": {"Type": "local"}}))
    cfg = load_settings(defaults=SETTINGS, local=local, env={})
    assert cfg["backend"]["type"] == "local"
    assert json.loads(SETTINGS.read_text())["Backend"]["Type"] == "openrouter"


def test_unknown_backend_type_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="Backend:Type must be one of openrouter, local, got 'openai'"):
        parse(with_(Backend={"Type": "openai"}), env={})


def test_openrouter_needs_model_and_base_url_and_key_env_name():
    d = with_(OpenRouter={"Model": "", "BaseUrl": "https://x", "ApiKeyEnv": "K"})
    with pytest.raises(ValueError, match="OpenRouter:Model is required"):
        parse(d, env={})


def test_openrouter_settings_read_the_key_from_env_and_never_from_the_config():
    cfg = parse(BASE, env={})
    s = openrouter_settings(cfg, env={"OPENROUTER_API_KEY": "sk-or-v1-test"})
    assert s == {"model": "~liquid/d1-ultra-latest", "base_url": "https://openrouter.ai/api/alpha/decisions", "api_key": "sk-or-v1-test"}
    assert "sk-or-v1" not in json.dumps(cfg)


def test_missing_key_fails_and_points_to_env_example():
    cfg = parse(BASE, env={})
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set.*\\.env\\.example"):
        openrouter_settings(cfg, env={})
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        openrouter_settings(cfg, env={"OPENROUTER_API_KEY": "   "})


def test_cpu_device_turns_gpu_layers_off_and_metal_keeps_them():
    cfg = parse(with_(Device={"Mode": "cpu"}), env={})
    assert model_options(cfg)["ngl"] == 0
    cfg = parse(with_(Device={"Mode": "metal"}), env={})
    assert model_options(cfg)["ngl"] == 999


def test_dotenv_loads_values_without_overriding_the_shell(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nOPENROUTER_API_KEY=sk-or-v1-file\nexport OTHER="quoted value"\nBARE=value # note\nSET_IN_SHELL=from-file\n')
    environ = {"SET_IN_SHELL": "from-shell"}
    loaded = load_dotenv(env_file, environ)
    assert environ["OPENROUTER_API_KEY"] == "sk-or-v1-file"
    assert environ["OTHER"] == "quoted value" and environ["BARE"] == "value"
    assert environ["SET_IN_SHELL"] == "from-shell"
    assert "SET_IN_SHELL" not in loaded


def test_dotenv_missing_file_is_a_no_op(tmp_path):
    assert load_dotenv(tmp_path / "nope.env", {}) == {}
