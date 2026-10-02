from unittest.mock import MagicMock, patch
import pytest

from tools.keyboard import (
    KeyboardTool,
    KeyboardTypeTool,
    KeyboardPressTool,
    KeyboardHotkeyTool,
)


def test_keyboard_tool_initialization():
    """Verify tool schemas, name, and input schema for KeyboardTool."""
    tool = KeyboardTool()
    assert tool.name == "keyboard"
    assert "keyboard_tool" in tool.aliases
    schema = tool.to_tool_schema()
    assert schema["function"]["name"] == "keyboard"
    props = schema["function"]["parameters"]["properties"]
    assert "action" in props
    assert "text" in props
    assert "key" in props
    assert "keys" in props


def test_keyboard_type_valid_text():
    """Verify typing normal text."""
    mock_backend = MagicMock()
    tool = KeyboardTool(backend=mock_backend)

    result = tool.execute(action="type", text="Hello from Zia")
    assert result["success"] is True
    assert result["action"] == "type"
    mock_backend.type_text.assert_called_once_with("Hello from Zia")


def test_keyboard_press_enter():
    """Verify pressing named key 'ENTER'."""
    mock_backend = MagicMock()
    tool = KeyboardTool(backend=mock_backend)

    result = tool.execute(action="press", key="ENTER")
    assert result["success"] is True
    assert result["action"] == "press"
    mock_backend.press_key.assert_called_once_with("ENTER")


def test_keyboard_hotkey_combination():
    """Verify hotkey shortcut execution with list of keys."""
    mock_backend = MagicMock()
    tool = KeyboardTool(backend=mock_backend)

    result = tool.execute(action="hotkey", keys=["CTRL", "L"])
    assert result["success"] is True
    assert result["action"] == "hotkey"
    mock_backend.hotkey.assert_called_once_with(["CTRL", "L"])


def test_keyboard_hotkey_string_format():
    """Verify hotkey shortcut accepts string combinations like 'Ctrl+A'."""
    mock_backend = MagicMock()
    tool = KeyboardTool(backend=mock_backend)

    result = tool.execute(action="hotkey", keys="Ctrl+A")
    assert result["success"] is True
    mock_backend.hotkey.assert_called_once_with(["Ctrl", "A"])


def test_keyboard_reject_empty_text():
    """Reject empty text for type action."""
    mock_backend = MagicMock()
    tool = KeyboardTool(backend=mock_backend)

    result = tool.execute(action="type", text="")
    assert result["success"] is False
    assert "requires non-empty 'text'" in result["error"]
    mock_backend.type_text.assert_not_called()


def test_keyboard_reject_invalid_key():
    """Reject unknown key names with helpful error listing supported keys."""
    mock_backend = MagicMock()
    tool = KeyboardTool(backend=mock_backend)

    result = tool.execute(action="press", key="NON_EXISTENT_KEY_12345")
    assert result["success"] is False
    assert "Unknown key" in result["error"]
    assert "Supported named keys" in result["error"]
    mock_backend.press_key.assert_not_called()


def test_keyboard_modular_tools():
    """Verify modular wrapper tools (KeyboardTypeTool, KeyboardPressTool, KeyboardHotkeyTool)."""
    mock_backend = MagicMock()

    # TypeTool
    type_tool = KeyboardTypeTool(backend=mock_backend)
    assert type_tool.name == "keyboard.type"
    res1 = type_tool.execute(text="Test input")
    assert res1["success"] is True
    mock_backend.type_text.assert_called_once_with("Test input")

    # PressTool
    press_tool = KeyboardPressTool(backend=mock_backend)
    assert press_tool.name == "keyboard.press"
    res2 = press_tool.execute(key="ESCAPE")
    assert res2["success"] is True
    mock_backend.press_key.assert_called_once_with("ESCAPE")

    # HotkeyTool
    hotkey_tool = KeyboardHotkeyTool(backend=mock_backend)
    assert hotkey_tool.name == "keyboard.hotkey"
    res3 = hotkey_tool.execute(keys=["ALT", "TAB"])
    assert res3["success"] is True
    mock_backend.hotkey.assert_called_once_with(["ALT", "TAB"])


def test_keyboard_screen_verification():
    """Verify screen verification after typing."""
    mock_backend = MagicMock()

    with patch("tools.keyboard.run_screen_verification", return_value={"verified": True, "observation": "Text visible in field"}):
        tool = KeyboardTool(backend=mock_backend)
        res = tool.execute(action="type", text="Verified text", verify=True, expected_state="text is visible")

        assert res["success"] is True
        assert "verification" in res
        assert res["verification"]["verified"] is True
