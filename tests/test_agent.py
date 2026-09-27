import pytest
from agent import Agent, AgentCore, clean_response_text
from tools.base import BaseTool
from tools.terminal import TerminalTool


class DummyModel:
    """Mock model for fast unit testing of Agent logic without loading GPU weights."""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0

    def generate_response(self, messages: list, **kwargs) -> str:
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
            self.call_count += 1
            return res
        return "Fallback final response"


def test_agent_initialization():
    agent = Agent(model=None, max_tool_iterations=3)
    assert len(agent.tools) == 0
    assert agent.max_tool_iterations == 3
    assert len(agent.messages) == 0


def test_tool_registration():
    agent = Agent(model=None)
    terminal_tool = TerminalTool()
    agent.register_tool(terminal_tool)

    assert "terminal" in agent.tools
    assert agent.tools["terminal"] == terminal_tool
    schemas = agent.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "terminal"


def test_agent_tool_call_parsing():
    agent = Agent(model=None)
    text = (
        "<think>Looking up directory...</think>\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "terminal",\n'
        '  "arguments": {"command": "ls -la"}\n'
        "}\n"
        "</tool_call>"
    )
    name, args = agent.parse_tool_call(text)
    assert name == "terminal"
    assert args == {"command": "ls -la"}


def test_agent_loop_with_tool_call():
    responses = [
        '<tool_call>{"name": "terminal", "arguments": {"command": "echo test"}}</tool_call>',
        "The command output is 'test'."
    ]
    mock_model = DummyModel(responses)
    agent = Agent(model=mock_model)
    agent.register_tool(TerminalTool())

    final_res = agent.run("Run echo test")
    assert final_res == "The command output is 'test'."
    # Messages should include: System, User, Assistant tool call, Tool result, Assistant final
    assert len(agent.messages) == 5
    assert agent.messages[3]["role"] == "tool"
    assert agent.messages[3]["name"] == "terminal"


def test_unknown_tool_handling():
    responses = [
        '<tool_call>{"name": "non_existent_tool", "arguments": {"foo": "bar"}}</tool_call>',
        "I realized the tool does not exist."
    ]
    mock_model = DummyModel(responses)
    agent = Agent(model=mock_model)
    agent.register_tool(TerminalTool())

    final_res = agent.run("Use nonexistent tool")
    assert final_res == "I realized the tool does not exist."
    # The tool result should contain an error message about the unknown tool
    tool_msg = agent.messages[3]
    assert tool_msg["role"] == "tool"
    assert "Unknown tool 'non_existent_tool'" in tool_msg["content"]


def test_invalid_arguments_handling():
    responses = [
        '<tool_call>{"name": "terminal", "arguments": {}}</tool_call>',
        "Command was missing."
    ]
    mock_model = DummyModel(responses)
    agent = Agent(model=mock_model)
    agent.register_tool(TerminalTool())

    final_res = agent.run("Run empty command")
    assert final_res == "Command was missing."
    tool_msg = agent.messages[3]
    assert "Missing required argument" in tool_msg["content"]


def test_agent_loop_max_iterations_termination():
    # Model endlessly attempts tool calls
    responses = [
        '<tool_call>{"name": "terminal", "arguments": {"command": "echo loop"}}</tool_call>'
    ] * 10
    mock_model = DummyModel(responses)
    agent = Agent(model=mock_model, max_tool_iterations=3)
    agent.register_tool(TerminalTool())

    final_res = agent.run("Loop test")
    assert "maximum tool iteration limit" in final_res
    # Should have stopped after exactly 3 iterations
    assert mock_model.call_count == 3


def test_agent_direct_response():
    mock_model = DummyModel(["Hello! I am Zia, how can I help you?"])
    agent = Agent(model=mock_model)

    res = agent.run("Hello")
    assert res == "Hello! I am Zia, how can I help you?"
    assert len(agent.messages) == 3  # System, User, Assistant


def test_clean_response_text():
    raw = "<think>\nThinking about life...\n</think>\nHere is the answer."
    cleaned = clean_response_text(raw)
    assert cleaned == "Here is the answer."

    raw_no_think = "Direct answer without thinking tags."
    assert clean_response_text(raw_no_think) == raw_no_think
