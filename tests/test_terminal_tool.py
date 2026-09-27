import pytest
from tools.terminal import TerminalTool


def test_terminal_tool_successful_command():
    tool = TerminalTool(timeout=5)
    result = tool.execute(command="echo 'Hello Zia'")
    assert result["success"] is True
    assert "Hello Zia" in result["stdout"]
    assert result["exit_code"] == 0


def test_terminal_tool_failed_command():
    tool = TerminalTool(timeout=5)
    result = tool.execute(command="ls /non_existent_directory_zia_12345")
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
    result = tool.execute(command="rm -rf /")
    assert result["success"] is False
    assert result["exit_code"] == 1
    assert "blocked by safety layer" in result["stderr"]


def test_terminal_tool_empty_command():
    tool = TerminalTool(timeout=5)
    result = tool.execute(command="")
    assert result["success"] is False
    assert result["exit_code"] == 1
