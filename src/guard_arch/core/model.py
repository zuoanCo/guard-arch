"""Model router: role -> pydantic-ai model, driven by config/models.yaml."""

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai.models import Model


class ModelConfigError(Exception):
    """Raised for missing config / missing API keys, with a user-friendly message."""


class ModelRouter:
    def __init__(self, config: dict[str, Any]):
        self.roles: dict[str, dict[str, Any]] = dict(config.get("models") or {})
        if not self.roles:
            raise ModelConfigError("models config is empty: expected a top-level `models:` mapping")

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRouter":
        path = Path(path)
        if not path.exists():
            raise ModelConfigError(f"models config not found: {path}")
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def role_names(self) -> list[str]:
        return sorted(self.roles)

    def select(self, role: str = "default") -> Model:
        try:
            cfg = self.roles[role]
        except KeyError:
            raise ModelConfigError(
                f"unknown model role {role!r} (configured: {sorted(self.roles)})"
            ) from None
        provider = cfg.get("provider", "openai")
        if provider == "test":
            return self._build_test_model(cfg)
        if provider in ("openai", "openai-compatible"):
            return self._build_openai(cfg)
        if provider == "anthropic":
            return self._build_anthropic(cfg)
        if provider == "google":
            return self._build_google(cfg)
        raise ModelConfigError(f"unsupported provider {provider!r} for role {role!r}")

    # -- providers -----------------------------------------------------------

    def _api_key(self, cfg: dict[str, Any], default_env: str) -> str:
        env = cfg.get("api_key_env") or default_env
        key = os.environ.get(env)
        if not key:
            raise ModelConfigError(
                f"missing API key: set the {env} environment variable "
                f"(see .env.example), or use --model test for a keyless run"
            )
        return key

    def _build_openai(self, cfg: dict[str, Any]) -> Model:
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
        from pydantic_ai.providers.openai import OpenAIProvider

        api_key = self._api_key(cfg, "OPENAI_API_KEY")
        provider = OpenAIProvider(base_url=cfg.get("base_url"), api_key=api_key)
        # 可选 extra_body：透传 provider 专有请求参数（如 MiMo 的 thinking 开关）
        settings = None
        if cfg.get("extra_body"):
            settings = OpenAIChatModelSettings(extra_body=cfg["extra_body"])
        return OpenAIChatModel(cfg["model"], provider=provider, settings=settings)

    def _build_anthropic(self, cfg: dict[str, Any]) -> Model:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        api_key = self._api_key(cfg, "ANTHROPIC_API_KEY")
        provider = AnthropicProvider(api_key=api_key, base_url=cfg.get("base_url"))
        return AnthropicModel(cfg["model"], provider=provider)

    def _build_google(self, cfg: dict[str, Any]) -> Model:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        api_key = self._api_key(cfg, "GOOGLE_API_KEY")
        provider = GoogleProvider(api_key=api_key, base_url=cfg.get("base_url"))
        return GoogleModel(cfg["model"], provider=provider)

    def _build_test_model(self, cfg: dict[str, Any]) -> Model:
        """Keyless model for tests and smoke runs.

        With a `script` entry the model first calls the listed tools with the
        given arguments, then answers with `text`; otherwise a plain TestModel.
        """
        script = cfg.get("script")
        if not script:
            from pydantic_ai.models.test import TestModel

            return TestModel(
                call_tools=cfg.get("call_tools", []),
                custom_output_text=cfg.get("output_text") or "Test model response.",
                seed=int(cfg.get("seed", 0)),
            )

        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import DeltaToolCall, FunctionModel

        tool_steps = [s for s in script if "tool" in s]
        text_steps = [s for s in script if "text" in s]
        final_text = text_steps[-1]["text"] if text_steps else "Done."

        def function(messages, info) -> ModelResponse:
            called = _called_tools(messages)
            for step in tool_steps:
                if step["tool"] not in called:
                    return ModelResponse(
                        parts=[ToolCallPart(step["tool"], dict(step.get("args") or {}))]
                    )
            return ModelResponse(parts=[TextPart(final_text)])

        async def stream_function(messages, info):
            called = _called_tools(messages)
            for index, step in enumerate(tool_steps):
                if step["tool"] not in called:
                    yield {
                        index: DeltaToolCall(
                            name=step["tool"],
                            json_args=json.dumps(step.get("args") or {}, ensure_ascii=False),
                        )
                    }
                    return
            yield final_text

        return FunctionModel(function, stream_function=stream_function, model_name="test")


def _called_tools(messages) -> set[str]:
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    called: set[str] = set()
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    called.add(part.tool_name)
    return called
