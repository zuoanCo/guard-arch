from guard_arch.permissions.engine import PermissionDecision, PermissionEngine


def test_read_tools_allowed():
    engine = PermissionEngine()
    assert engine.decide("read_file", {"path": "src/main.py"}) is PermissionDecision.ALLOW
    assert engine.decide("list_directory", {"path": "."}) is PermissionDecision.ALLOW
    assert engine.decide("search_text", {"pattern": "foo"}) is PermissionDecision.ALLOW


def test_write_tools_allowed():
    engine = PermissionEngine()
    assert engine.decide("write_file", {"path": "a.py", "content": "x"}) is PermissionDecision.ALLOW
    assert engine.decide("edit_file", {"path": "a.py"}) is PermissionDecision.ALLOW
    assert engine.decide("remember", {"layer": "project", "key": "k", "value": "v"}) is (
        PermissionDecision.ALLOW
    )


def test_dangerous_commands_denied():
    engine = PermissionEngine()
    denied = [
        "rm -rf /",
        "rm -rf ./build",
        "rm -fr ~",
        "del /f /s C:\\Windows",
        "format C:",
        "shutdown /s /t 0",
        ":(){ :|:& };:",
        "dd if=/dev/zero of=/dev/sda",
    ]
    for cmd in denied:
        assert engine.decide("run_command", {"command": cmd}) is PermissionDecision.DENY, cmd


def test_other_shell_commands_ask():
    engine = PermissionEngine()
    for cmd in ["ls", "git status", "npm install", "python -m pytest"]:
        assert engine.decide("run_command", {"command": cmd}) is PermissionDecision.ASK, cmd


def test_unknown_tools_ask_by_default():
    engine = PermissionEngine()
    assert engine.decide("some_mcp_tool", {"x": 1}) is PermissionDecision.ASK


def test_auto_approve_allows_ask_but_never_bypasses_deny():
    engine = PermissionEngine(auto_approve=True)
    assert engine.decide("run_command", {"command": "npm install"}) is PermissionDecision.ALLOW
    assert engine.decide("run_command", {"command": "rm -rf /"}) is PermissionDecision.DENY


async def test_authorize_uses_callback_for_ask():
    engine = PermissionEngine(approval_callback=lambda tool, args, reason: True)
    assert await engine.authorize("run_command", {"command": "ls"}) is True
    engine_no = PermissionEngine(approval_callback=lambda tool, args, reason: False)
    assert await engine_no.authorize("run_command", {"command": "ls"}) is False


async def test_authorize_deny_never_calls_callback():
    called = []
    engine = PermissionEngine(approval_callback=lambda *a: called.append(a) or True)
    assert await engine.authorize("run_command", {"command": "rm -rf /"}) is False
    assert called == []
