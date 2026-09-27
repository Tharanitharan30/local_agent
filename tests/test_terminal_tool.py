import pytest
from tools.terminal import TerminalTool


def test_terminal_tool_schema():
    tool = TerminalTool()
    assert tool.name == "terminal"
    assert "command" in tool.input_schema["properties"]
    assert "command" in tool.input_schema["required"]
    schema = tool.to_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "terminal"


def test_terminal_tool_successful_command():
    tool = TerminalTool(timeout=5)
    result = tool.execute(command="echo 'Hello Zia'")
    assert result["success"] is True
    assert "Hello Zia" in result["stdout"]
    assert result["exit_code"] == 0
    assert result["stderr"] == ""


def test_terminal_tool_failed_command():
    tool = TerminalTool(timeout=5)
    result = tool.execute(command="ls /non_existent_directory_zia_xyz_987")
    assert result["success"] is False
    assert result["exit_code"] != 0
    assert len(result["stderr"]) > 0


def test_terminal_tool_timeout():
    tool = TerminalTool(timeout=1)
    result = tool.execute(command="sleep 3")
    assert result["success"] is False
    assert result["exit_code"] == 124
    assert "timed out" in result["stderr"]


def test_terminal_tool_blocked_dangerous_command():
    tool = TerminalTool(timeout=5)
    dangerous_commands = [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~/*",
        "mkfs.ext4 /dev/sdb1",
        "dd if=/dev/zero of=/dev/sda bs=1M",
        "cat /dev/zero > /dev/sda",
        ":(){ :|:& };:",
        "shutdown -h now",
        "reboot",
    ]

    for cmd in dangerous_commands:
        result = tool.execute(command=cmd)
        assert result["success"] is False, f"Command '{cmd}' should have been blocked!"
        assert result.get("blocked") is True
        assert "reason" in result
        assert result["exit_code"] == 1
        assert "blocked by safety layer" in result["stderr"]


def test_terminal_tool_empty_command():
    tool = TerminalTool(timeout=5)
    result = tool.execute(command="")
    assert result["success"] is False
    assert result["exit_code"] == 1
    assert "Empty command" in result["stderr"]


def test_terminal_tool_kwarg_execution():
    tool = TerminalTool(timeout=5)
    result = tool.execute(cmd="echo 'kwarg test'")
    assert result["success"] is True
    assert "kwarg test" in result["stdout"]
