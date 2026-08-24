"""Generic web access tool: lets the agent fetch a URL and read the response text.

This is a general-purpose capability (not tied to any specific site/API): the
model decides on its own when it needs external/realtime information and which
URL to fetch (public web pages or public HTTP APIs), then reasons over the
returned content.
"""

import httpx

from guard_arch.core.tool import Tool

# 响应体截断上限：避免超长页面灌爆模型上下文
MAX_RESPONSE_CHARS = 4000

_TIMEOUT = 20.0


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGo result links are redirect-wrapped (//duckduckgo.com/l/?uddg=<urlencoded>);
    extract the real target URL from the uddg param, else return href as-is."""
    from urllib.parse import parse_qs, urlparse

    if "uddg=" in href:
        query = parse_qs(urlparse(href if "://" in href else f"https:{href}").query)
        target = query.get("uddg", [href])[0]
        return target
    return href


def _parse_ddg_results(html: str, limit: int = 8) -> list[tuple[str, str]]:
    """Extract (title, url) pairs from DuckDuckGo HTML-endpoint result page."""
    import re

    # 每条结果是一个 <a class="result__a" href="...">标题</a>
    pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    results: list[tuple[str, str]] = []
    for href, title_html in pattern.findall(html)[:limit]:
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if title:
            results.append((title, _decode_ddg_href(href)))
    return results


def make_web_tools() -> list[Tool]:
    """Web access tools (web_search + web_fetch). Registered globally like remember."""

    async def web_search(query: str) -> str:
        """Search the web for `query` (DuckDuckGo, no API key) and return the top
        results as a numbered list of 'title — url'. Use to find sources/pages for
        external or realtime questions, then web_fetch the most relevant URL."""
        import urllib.parse

        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "guard-arch/0.1"})
        except httpx.HTTPError as exc:
            return f"Error: search failed: {exc}"
        results = _parse_ddg_results(resp.text)
        if not results:
            return f"no results for {query!r}"
        return "\n".join(f"{i}. {title} — {link}" for i, (title, link) in enumerate(results, 1))

    async def web_fetch(url: str) -> str:
        """Fetch a URL over HTTP(S) and return the response body as text.
        Use when you need external or realtime information (public web pages,
        public HTTP APIs) that is not in your context or memory."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "guard-arch/0.1"})
        except httpx.HTTPError as exc:
            # 网络层失败以 Error: 文本返回给模型，让它据此调整策略（换 URL/说明限制）
            return f"Error: request failed: {exc}"
        text = resp.text
        if len(text) > MAX_RESPONSE_CHARS:
            text = text[:MAX_RESPONSE_CHARS] + "\n...[truncated]"
        return text

    return [
        Tool(
            "web_search",
            "Search the web by keyword and get top results (title + url); use to find "
            "sources for external/realtime questions, then web_fetch the relevant page",
            web_search,
        ),
        Tool(
            "web_fetch",
            "Fetch a URL over HTTP(S) and return the response body as text; use to "
            "retrieve external or realtime information (public web pages or HTTP APIs)",
            web_fetch,
        ),
    ]
