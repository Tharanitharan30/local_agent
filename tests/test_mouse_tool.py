from unittest.mock import MagicMock, patch
import pytest

from tools.mouse import (
    MouseTool,
    MouseMoveTool,
    MouseClickTool,
    MouseDoubleClickTool,
    MouseRightClickTool,
    MouseScrollTool,
)


def test_mouse_tool_initialization():
    """Verify tool schemas, name, and input schema for MouseTool."""
    tool = MouseTool()
    assert tool.name == "mouse"
    assert "mouse_tool" in tool.aliases
    schema = tool.to_tool_schema()
    assert schema["function"]["name"] == "mouse"
    props = schema["function"]["parameters"]["properties"]
    assert "action" in props
    assert "x" in props
    assert "y" in props
    assert "button" in props
    assert "clicks" in props
    assert "steps" in props


def test_mouse_move_harmless_location():
    """Verify mouse movement to a safe screen coordinate."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)

    tool = MouseTool(backend=mock_backend)
    result = tool.execute(action="move", x=500, y=300)

    assert result["success"] is True
    assert result["action"] == "move"
    assert result["x"] == 500
    assert result["y"] == 300
    mock_backend.move.assert_called_once_with(500, 300)


def test_mouse_click_execution():
    """Verify single, double, and right click execution."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)

    tool = MouseTool(backend=mock_backend)

    # Left click
    res1 = tool.execute(action="click", x=400, y=200, button="left")
    assert res1["success"] is True
    mock_backend.click.assert_called_with(400, 200, button="left", clicks=1)

    # Double click
    res2 = tool.execute(action="double_click", x=400, y=200)
    assert res2["success"] is True
    mock_backend.click.assert_called_with(400, 200, button="left", clicks=2)

    # Right click
    res3 = tool.execute(action="right_click", x=400, y=200)
    assert res3["success"] is True
    mock_backend.click.assert_called_with(400, 200, button="right", clicks=1)


def test_mouse_scroll_execution():
    """Verify vertical scrolling with discrete steps."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)

    tool = MouseTool(backend=mock_backend)
    res = tool.execute(action="scroll", steps=3, x=500, y=500)

    assert res["success"] is True
    assert res["action"] == "scroll"
    mock_backend.scroll.assert_called_once_with(steps=3, x=500, y=500)


def test_mouse_reject_negative_coordinates():
    """Reject negative screen coordinates."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (False, "Coordinates (-50, 100) are outside active screen bounds.")

    tool = MouseTool(backend=mock_backend)
    res = tool.execute(action="move", x=-50, y=100)

    assert res["success"] is False
    assert "outside" in res["error"]
    mock_backend.move.assert_not_called()


def test_mouse_reject_out_of_bounds_coordinates():
    """Reject coordinates exceeding active screen dimensions."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (False, "Coordinates (2500, 1500) are outside active screen bounds.")

    tool = MouseTool(backend=mock_backend)
    res = tool.execute(action="click", x=2500, y=1500)

    assert res["success"] is False
    assert "outside" in res["error"]
    assert res["screen_bounds"]["width"] == 1920
    assert res["screen_bounds"]["height"] == 1080
    mock_backend.click.assert_not_called()


def test_mouse_modular_tools():
    """Verify modular wrapper tools (MouseMoveTool, MouseClickTool, etc.)."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)

    # MoveTool
    move_tool = MouseMoveTool(backend=mock_backend)
    assert move_tool.name == "mouse.move"
    res1 = move_tool.execute(x=100, y=200)
    assert res1["success"] is True

    # ClickTool
    click_tool = MouseClickTool(backend=mock_backend)
    assert click_tool.name == "mouse.click"
    res2 = click_tool.execute(x=150, y=250)
    assert res2["success"] is True

    # DoubleClickTool
    dclick_tool = MouseDoubleClickTool(backend=mock_backend)
    assert dclick_tool.name == "mouse.double_click"
    res3 = dclick_tool.execute(x=150, y=250)
    assert res3["success"] is True

    # RightClickTool
    rclick_tool = MouseRightClickTool(backend=mock_backend)
    assert rclick_tool.name == "mouse.right_click"
    res4 = rclick_tool.execute(x=150, y=250)
    assert res4["success"] is True

    # ScrollTool
    scroll_tool = MouseScrollTool(backend=mock_backend)
    assert scroll_tool.name == "mouse.scroll"
    res5 = scroll_tool.execute(steps=-2)
    assert res5["success"] is True


def test_mouse_screen_verification():
    """Verify that verify=True executes screen observation post-action."""
    mock_backend = MagicMock()
    mock_backend.get_screen_size.return_value = (1920, 1080)
    mock_backend.validate_coordinates.return_value = (True, None)

    with patch("tools.mouse.run_screen_verification", return_value={"verified": True, "observation": "Button state active"}):
        tool = MouseTool(backend=mock_backend)
        res = tool.execute(action="click", x=100, y=100, verify=True, expected_state="button is active")

        assert res["success"] is True
        assert "verification" in res
        assert res["verification"]["verified"] is True
        assert "Button state active" in res["verification"]["observation"]
