"""Tests for the generic web tools (web_search / web_fetch)."""

import pytest

from guard_arch.tools.web import make_web_tools


def _get_tool(name: str):
    tools = make_web_tools()
    return next(t for t in tools if t.name == name)


def _get_web_fetch():
    return _get_tool("web_fetch")


async def test_web_fetch_returns_response_text(monkeypatch: pytest.MonkeyPatch):
    """web_fetch fetches the given URL and returns the body text for the model to reason over."""

    class FakeResponse:
        text = '{"weather": "sunny", "temp": 25}'

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            assert url.startswith("https://")
            return FakeResponse()

    monkeypatch.setattr("guard_arch.tools.web.httpx.AsyncClient", FakeClient)

    result = await _get_web_fetch().handler("https://api.example.com/weather?city=beijing")
    assert "sunny" in result
    assert not result.startswith("Error:")


async def test_web_fetch_network_error_returns_error_text(monkeypatch: pytest.MonkeyPatch):
    """Network failure returns an 'Error: ...' string (feedback to the model, not a crash)."""
    import httpx

    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("unreachable")

    monkeypatch.setattr("guard_arch.tools.web.httpx.AsyncClient", FailingClient)

    result = await _get_web_fetch().handler("https://unreachable.example.com")
    assert result.startswith("Error:")


async def test_web_fetch_truncates_long_responses(monkeypatch: pytest.MonkeyPatch):
    """Over-long bodies are truncated so they don't flood the model context."""

    class FakeResponse:
        text = "x" * 10000

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr("guard_arch.tools.web.httpx.AsyncClient", FakeClient)

    result = await _get_web_fetch().handler("https://example.com/big")
    assert len(result) < 10000
    assert "truncated" in result


# ---------- web_search ----------

_DDG_HTML = """
<html><body>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fweather-beijing">北京天气 - 中国天气网</a>
<a class="result__a" href="https://example.org/forecast">Beijing Weather Forecast</a>
</body></html>
"""


async def test_web_search_returns_titles_and_urls(monkeypatch: pytest.MonkeyPatch):
    """web_search queries DuckDuckGo and returns a numbered title—url list (redirects decoded)."""

    class FakeResponse:
        text = _DDG_HTML

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr("guard_arch.tools.web.httpx.AsyncClient", FakeClient)

    result = await _get_tool("web_search").handler("北京天气")
    # 结果包含编号列表：标题 + 解码后的真实 URL（uddg 重定向包装已被解开）
    assert "1." in result
    assert "北京天气 - 中国天气网" in result
    assert "https://example.com/weather-beijing" in result
    assert "Beijing Weather Forecast" in result
    assert "https://example.org/forecast" in result


async def test_web_search_failure_returns_error_text(monkeypatch: pytest.MonkeyPatch):
    """Search network failure returns 'Error: ...' text (feedback, not crash)."""
    import httpx

    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("blocked")

    monkeypatch.setattr("guard_arch.tools.web.httpx.AsyncClient", FailingClient)

    result = await _get_tool("web_search").handler("anything")
    assert result.startswith("Error:")
