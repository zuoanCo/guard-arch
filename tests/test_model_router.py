import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from guard_arch.core.model import ModelConfigError, ModelRouter

CONFIG = """
models:
  default:
    provider: openai
    model: deepseek-chat
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
  test:
    provider: test
    output_text: hello
  test-demo:
    provider: test
    script:
      - tool: read_file
        args: {path: README.md}
      - text: done
"""


@pytest.fixture
def router(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return ModelRouter.from_file(path)


def test_role_names(router):
    assert router.role_names() == ["default", "test", "test-demo"]


def test_unknown_role_raises(router):
    with pytest.raises(ModelConfigError, match="unknown model role"):
        router.select("nope")


def test_missing_api_key_friendly_error(router, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ModelConfigError, match="DEEPSEEK_API_KEY"):
        router.select("default")


def test_openai_compatible_model_builds(router, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    model = router.select("default")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "deepseek-chat"


def test_test_role_needs_no_key(router):
    model = router.select("test")
    assert isinstance(model, TestModel)


def test_scripted_test_role(router):
    model = router.select("test-demo")
    assert isinstance(model, FunctionModel)


def test_empty_config_raises():
    with pytest.raises(ModelConfigError, match="empty"):
        ModelRouter({})


def test_missing_config_file(tmp_path):
    with pytest.raises(ModelConfigError, match="not found"):
        ModelRouter.from_file(tmp_path / "nope.yaml")
