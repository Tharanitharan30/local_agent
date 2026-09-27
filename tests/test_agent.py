import pytest
from agent import AgentCore
from tools.terminal import TerminalTool


class DummyModel:
    """Mock model for fast unit testing of AgentCore logic without loading GPU weights."""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0

    def generate_response(self, messages: list) -> str:
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
            self.call_count += 1
            return res
        return "Final response"


def test_agent_initialization():
    agent = AgentCore(model=None)
    assert len(agent.tools) == 0

    terminal_tool = TerminalTool()
    agent.register_tool(terminal_tool)
    assert "terminal" in agent.tools
    assert agent.tools["terminal"] == terminal_tool


def test_agent_tool_call_parsing():
    agent = AgentCore(model=None)
    text = (
        "Let me run a command.\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "terminal",\n'
        '  "arguments": {"command": "ls -l"}\n'
        "}\n"
        "</tool_call>"
    )
    name, args = agent.parse_tool_call(text)
    assert name == "terminal"
    assert args == {"command": "ls -l"}


def test_agent_loop_with_tool_call():
    responses = [
        '<tool_call>{"name": "terminal", "arguments": {"command": "echo test"}}</tool_call>',
        "The command output is 'test'."
    ]
    mock_model = DummyModel(responses)
    agent = AgentCore(model=mock_model)
    agent.register_tool(TerminalTool())

    final_res = agent.run("Run echo test")
    assert final_res == "The command output is 'test'."
    assert len(agent.messages) >= 4  # System, User, Assistant tool call, User tool result, Assistant final
