import pytest

from columnmap.core.config import load_config, model_options, openrouter_settings

BASE = """
[model]
quant = "Q8_0"
ngl = 999
flash_attn = "default"

[run]
keep_d = 0.0
target_precision = 1.0
prompt_version = "v1"
mode = "options"
seed = 1234
conds = ["name", "samples"]

[splits]
dev = [11]
calibration = [12, 13]
test = [21, 22]
"""


def test_valid_config_loads_and_missing_backend_and_device_get_defaults():
    cfg = load_config(BASE)
    assert cfg["backend"]["type"] == "local_gguf" and cfg["device"]["mode"] == "auto"


def test_unknown_backend_type_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="backend type must be one of local_gguf, openrouter, got 'openai'"):
        load_config(BASE + '\n[backend]\ntype = "openai"\n')


def test_missing_required_model_quant_is_refused():
    with pytest.raises(ValueError, match="model.quant is required"):
        load_config(BASE.replace('quant = "Q8_0"\n', ""))


def test_target_precision_out_of_range_is_refused():
    with pytest.raises(ValueError, match="target_precision must be in \\(0, 1\\]"):
        load_config(BASE.replace("target_precision = 1.0", "target_precision = 1.5"))


def test_cpu_device_turns_gpu_layers_off_and_auto_on_apple_silicon_keeps_them():
    cfg = load_config(BASE + '\n[device]\nmode = "cpu"\n')
    assert model_options(cfg)["ngl"] == 0
    cfg = load_config(BASE + '\n[device]\nmode = "metal"\n')
    assert model_options(cfg)["ngl"] == 999


OPENROUTER = BASE + '''
[backend]
type = "openrouter"

[backend.openrouter]
model = "~liquid/d1-ultra-latest"
base_url = "https://openrouter.ai/api/alpha/decisions"
api_key_env = "OPENROUTER_API_KEY"
'''


def test_openrouter_settings_read_the_key_from_env_and_never_from_the_file():
    cfg = load_config(OPENROUTER)
    s = openrouter_settings(cfg, env={"OPENROUTER_API_KEY": "sk-or-v1-test"})
    assert s == {"model": "~liquid/d1-ultra-latest", "base_url": "https://openrouter.ai/api/alpha/decisions", "api_key": "sk-or-v1-test"}
    assert "sk-or-v1" not in OPENROUTER and "sk-or-v1" not in str(cfg)


def test_openrouter_without_the_env_key_fails_with_a_pointer_to_env_example():
    cfg = load_config(OPENROUTER)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set.*\\.env\\.example"):
        openrouter_settings(cfg, env={})
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        openrouter_settings(cfg, env={"OPENROUTER_API_KEY": "   "})


def test_openrouter_needs_model_and_base_url():
    with pytest.raises(ValueError, match="backend.openrouter.model is required"):
        load_config(OPENROUTER.replace('model = "~liquid/d1-ultra-latest"\n', ""))
