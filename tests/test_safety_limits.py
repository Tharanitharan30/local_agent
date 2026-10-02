from unittest.mock import MagicMock, patch
import pytest

from tools.safety import ActionSafetyPolicy, ActionLimitTracker
from tools.mouse import MouseTool
from tools.keyboard import KeyboardTool
from agent import Agent


def test_high_impact_mouse_detection():
    """Verify that dangerous mouse intents trigger high-impact flag."""
    # High impact intent
    is_high, reason = ActionSafetyPolicy.is_high_impact_mouse_action(
        action="click", x=500, y=500, intent="delete all user files"
    )
    assert is_high is True
    assert "consequential" in reason.lower()

    # Purchase intent
    is_high, reason = ActionSafetyPolicy.is_high_impact_mouse_action(
        action="click", x=800, y=600, intent="checkout and buy item"
    )
    assert is_high is True

    # Safe intent
    is_high_safe, _ = ActionSafetyPolicy.is_high_impact_mouse_action(
        action="click", x=100, y=100, intent="focus text editor window"
    )
    assert is_high_safe is False


def test_high_impact_keyboard_detection():
    """Verify that dangerous hotkeys and destructive typing trigger confirmation."""
    # Dangerous hotkey: Ctrl+Alt+Del
    is_high, _ = ActionSafetyPolicy.is_high_impact_keyboard_action(
        action_type="hotkey", keys_or_text=["CTRL", "ALT", "DEL"]
    )
    assert is_high is True

    # Dangerous hotkey: Alt+F4
    is_high, _ = ActionSafetyPolicy.is_high_impact_keyboard_action(
        action_type="hotkey", keys_or_text=["ALT", "F4"]
    )
    assert is_high is True

    # Destructive typing: rm -rf
    is_high, _ = ActionSafetyPolicy.is_high_impact_keyboard_action(
        action_type="type", keys_or_text="rm -rf /home/user/data"
    )
    assert is_high is True

    # Safe typing
    is_high_safe, _ = ActionSafetyPolicy.is_high_impact_keyboard_action(
        action_type="type", keys_or_text="Hello world"
    )
    assert is_high_safe is False


def test_confirmation_denial_blocks_mouse_action():
    """Verify that user denying confirmation halts execution safely."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)
    tool = MouseTool(backend=mock_backend)

    with patch("tools.safety.ActionSafetyPolicy.request_confirmation", return_value=False):
        result = tool.execute(action="click", x=500, y=500, intent="delete repository", confirm=True)

        assert result["success"] is False
        assert result["blocked"] is True
        assert "cancelled by user" in result["reason"]
        mock_backend.click.assert_not_called()


def test_confirmation_approval_allows_mouse_action():
    """Verify that user approving confirmation allows execution to proceed."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)
    tool = MouseTool(backend=mock_backend)

    with patch("tools.safety.ActionSafetyPolicy.request_confirmation", return_value=True):
        result = tool.execute(action="click", x=500, y=500, intent="delete repository", confirm=True)

        assert result["success"] is True
        mock_backend.click.assert_called_once()


def test_action_limit_stops_runaway_loop():
    """Verify that ActionLimitTracker blocks requests exceeding MAX_ACTIONS_PER_TASK."""
    tracker = ActionLimitTracker(max_actions=5, max_retries=2)

    for i in range(5):
        allowed, err = tracker.record_and_validate(f"action_{i}")
        assert allowed is True
        assert err is None

    # 6th action should be blocked
    allowed, err = tracker.record_and_validate("action_6")
    assert allowed is False
    assert "limit exceeded" in err.lower()
    assert "runaway" in err.lower()


def test_repeated_identical_action_limit():
    """Verify that repeating the exact same action repeatedly is blocked."""
    tracker = ActionLimitTracker(max_actions=10, max_retries=2)

    sig = "mouse.click:x=100,y=100"
    allowed1, _ = tracker.record_and_validate(sig)
    assert allowed1 is True

    allowed2, _ = tracker.record_and_validate(sig)
    assert allowed2 is True

    # 3rd consecutive repeat exceeds max_retries=2
    allowed3, err = tracker.record_and_validate(sig)
    assert allowed3 is False
    assert "Repeated action limit exceeded" in err


def test_agent_enforces_action_limits():
    """Verify that Agent Core stops execution when tool calls exceed action limits."""
    mock_model = MagicMock()
    # Repeatedly emit the exact same tool call
    mock_model.generate_response.return_value = (
        '<tool_call>{"name": "mouse.click", "arguments": {"x": 100, "y": 100}}</tool_call>'
    )

    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)

    mouse_tool = MouseTool(backend=mock_backend)
    agent = Agent(model=mock_model, max_tool_iterations=5)
    agent.register_tool(mouse_tool)
    agent.action_tracker = ActionLimitTracker(max_actions=10, max_retries=2)

    agent.run("Click the button repeatedly")

    # Verify that the tool was blocked by repeat limit
    assert any(
        m.get("role") == "tool" and "Repeated action limit exceeded" in m.get("content", "")
        for m in agent.messages
    )
