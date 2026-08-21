import json

import httpx
import pytest

from guard_arch.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def collect_sse(client, url, payload) -> list[dict]:
    """POST and collect the SSE stream into [(event, data)] dicts."""
    events: list[dict] = []
    current: dict = {}
    async with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                current["event"] = line[len("event: "):]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[len("data: "):])
            elif line == "" and current:
                events.append(current)
                current = {}
    return events


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


async def test_list_agents(client):
    response = await client.get("/api/v1/agents")
    assert response.status_code == 200
    agents = response.json()
    assistant = next(a for a in agents if a["id"] == "assistant")
    assert assistant["name"] == "通用助手"
    assert assistant["model"] == "default"
    assert "coding" in assistant["skills"]
    assert "read_file" in assistant["tools"]


async def test_list_skills(client):
    response = await client.get("/api/v1/skills")
    assert response.status_code == 200
    skills = {s["name"] for s in response.json()}
    assert {"coding", "research"} <= skills


async def test_chat_sse_event_sequence(client, tmp_path):
    created = await client.post("/api/v1/sessions", json={"workspace": str(tmp_path)})
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    events = await collect_sse(
        client,
        "/api/v1/chat",
        {"session_id": session_id, "message": "你好", "model": "test"},
    )
    types = [e["event"] for e in events]
    assert types[0] == "agent_started"
    assert types[-1] == "agent_finished"
    assert types.count("message_delta") >= 1
    # session_id delivered in the event stream
    assert events[0]["data"]["session_id"] == session_id
    delta_text = "".join(
        e["data"].get("delta", "") for e in events if e["event"] == "message_delta"
    )
    assert "Guard Arch" in delta_text

    history = await client.get(f"/api/v1/sessions/{session_id}/messages")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "你好"
    assert "Guard Arch" in messages[1]["content"]


async def test_chat_creates_session_when_omitted(client, tmp_path):
    events = await collect_sse(
        client,
        "/api/v1/chat",
        {"workspace": str(tmp_path), "message": "hi", "model": "test"},
    )
    assert events[0]["event"] == "agent_started"
    assert events[0]["data"]["session_id"].startswith("s-")


async def test_chat_tool_flow_events(client, tmp_path):
    (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
    events = await collect_sse(
        client,
        "/api/v1/chat",
        {"workspace": str(tmp_path), "message": "读一下 README", "model": "test-demo"},
    )
    types = [e["event"] for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert types.index("tool_call") < types.index("tool_result") < types.index("agent_finished")
    call = next(e for e in events if e["event"] == "tool_call")
    assert call["data"]["tool"] == "read_file"
    result = next(e for e in events if e["event"] == "tool_result")
    assert result["data"]["ok"] is True
    assert "# demo" in result["data"]["output"]


async def test_chat_ask_permission_denied_by_default(client, tmp_path):
    events = await collect_sse(
        client,
        "/api/v1/chat",
        {"workspace": str(tmp_path), "message": "跑个命令", "model": "test-shell"},
    )
    types = [e["event"] for e in events]
    assert "permission_required" in types
    result = next(e for e in events if e["event"] == "tool_result")
    assert result["data"]["tool"] == "run_command"
    assert result["data"]["ok"] is False
    assert "permission denied" in result["data"]["output"]


async def test_chat_auto_approve_allows_benign_command(client, tmp_path):
    events = await collect_sse(
        client,
        "/api/v1/chat",
        {
            "workspace": str(tmp_path),
            "message": "跑个命令",
            "model": "test-shell",
            "auto_approve": True,
        },
    )
    types = [e["event"] for e in events]
    assert "permission_required" not in types
    result = next(e for e in events if e["event"] == "tool_result")
    assert result["data"]["ok"] is True
    assert "hello-from-tool" in result["data"]["output"]


async def test_unknown_session_messages_404(client):
    response = await client.get("/api/v1/sessions/s-nope/messages")
    assert response.status_code == 404
