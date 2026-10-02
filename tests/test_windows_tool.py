import pytest
from unittest.mock import patch, MagicMock

from tools.windows import (
    WindowListTool,
    WindowGetActiveTool,
    WindowFocusTool,
    WindowMinimizeTool,
    WindowMaximizeTool,
    WindowCloseTool,
)


MOCK_WINDOWS = [
    {
        "window_id": "win_microsoft_edge_37717",
        "title": "Choose Local LLM - Microsoft Edge - Personal",
        "application": "Microsoft Edge",
        "pid": 37717,
        "active": False,
        "focused": False,
        "minimized": False,
        "maximized": True,
        "position": {"x": 0, "y": 0},
        "size": {"width": 1920, "height": 1048},
    },
    {
        "window_id": "win_antigravity-ide_38892",
        "title": "zia - Antigravity IDE - pyvenv.cfg",
        "application": "antigravity-ide",
        "pid": 38892,
        "active": True,
        "focused": True,
        "minimized": False,
        "maximized": True,
        "position": {"x": 0, "y": 0},
        "size": {"width": 1920, "height": 1048},
    },
    {
        "window_id": "win_resources_43421",
        "title": "Resources",
        "application": "resources",
        "pid": 43421,
        "active": False,
        "focused": False,
        "minimized": False,
        "maximized": False,
        "position": {"x": 100, "y": 100},
        "size": {"width": 800, "height": 600},
    },
]


def test_window_list_all():
    tool = WindowListTool()
    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS):
        result = tool.execute()
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["windows"]) == 3
        assert result["windows"][1]["window_id"] == "win_antigravity-ide_38892"


def test_window_list_filter_by_application():
    tool = WindowListTool()
    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS):
        result = tool.execute(application="resources")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["windows"][0]["application"] == "resources"


def test_window_get_active_success():
    tool = WindowGetActiveTool()
    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS):
        result = tool.execute()
        assert result["success"] is True
        assert result["window"]["window_id"] == "win_antigravity-ide_38892"
        assert result["window"]["application"] == "antigravity-ide"


def test_window_get_active_none():
    tool = WindowGetActiveTool()
    with patch("tools.windows.query_desktop_windows", return_value=[]):
        result = tool.execute()
        assert result["success"] is False
        assert "No active" in result["error"]


def test_window_focus_by_id():
    mock_backend = MagicMock()
    tool = WindowFocusTool(backend=mock_backend)

    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS), \
         patch("tools.windows.execute_atspi_action", return_value={"success": True}), \
         patch("tools.windows.get_active_desktop_window", return_value=MOCK_WINDOWS[0]):

        result = tool.execute(window_id="win_microsoft_edge_37717", verify=True)
        assert result["success"] is True
        assert result["window_id"] == "win_microsoft_edge_37717"
        assert result["verified"] is True
        mock_backend.click.assert_called_once()


def test_window_focus_nonexistent():
    tool = WindowFocusTool(backend=MagicMock())
    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS):
        result = tool.execute(application="nonexistent_app")
        assert result["success"] is False
        assert "No open window matching" in result["error"]


def test_window_minimize():
    mock_backend = MagicMock()
    tool = WindowMinimizeTool(backend=mock_backend)

    with patch("tools.windows.execute_atspi_action", return_value={"success": True}):
        result = tool.execute(window_id="win_resources_43421")
        assert result["success"] is True
        assert result["action"] == "minimize"


def test_window_maximize():
    mock_backend = MagicMock()
    tool = WindowMaximizeTool(backend=mock_backend)

    with patch("tools.windows.execute_atspi_action", return_value={"success": True}):
        result = tool.execute(window_id="win_resources_43421")
        assert result["success"] is True
        assert result["action"] == "maximize"


def test_window_close_denied_confirmation():
    mock_backend = MagicMock()
    tool = WindowCloseTool(backend=mock_backend)

    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS), \
         patch("tools.safety.ActionSafetyPolicy.request_confirmation", return_value=False):

        result = tool.execute(window_id="win_resources_43421", confirm=False)
        assert result["success"] is False
        assert result["blocked"] is True
        assert "cancelled by user confirmation" in result["reason"]


def test_window_close_confirmed_execution():
    mock_backend = MagicMock()
    tool = WindowCloseTool(backend=mock_backend)

    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS), \
         patch("tools.safety.ActionSafetyPolicy.request_confirmation", return_value=True), \
         patch("tools.windows.execute_atspi_action", return_value={"success": True}):

        result = tool.execute(window_id="win_resources_43421", confirm=False)
        assert result["success"] is True
        assert result["action"] == "window.close"
