import pytest
from unittest.mock import patch, MagicMock

from tools.applications import ApplicationListTool, ApplicationLaunchTool
from tests.test_windows_tool import MOCK_WINDOWS


def test_application_list():
    tool = ApplicationListTool()
    with patch("tools.windows.query_desktop_windows", return_value=MOCK_WINDOWS):
        result = tool.execute()
        assert result["success"] is True
        assert result["count"] == 3
        app_names = [a["application"] for a in result["applications"]]
        assert "Microsoft Edge" in app_names
        assert "antigravity-ide" in app_names
        assert "resources" in app_names


def test_application_launch_allowed_app():
    tool = ApplicationLaunchTool()
    mock_proc = MagicMock()
    mock_proc.pid = 99999

    with patch("shutil.which", return_value="/usr/bin/gnome-calculator"), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("tools.windows.query_desktop_windows", return_value=[{"application": "gnome-calculator", "pid": 99999, "title": "Calculator"}]):

        result = tool.execute(application="calculator", verify=True)
        assert result["success"] is True
        assert result["executable"] == "/usr/bin/gnome-calculator"
        assert result["pid"] == 99999
        assert result["verified"] is True


def test_application_launch_blocks_shell_metacharacters():
    tool = ApplicationLaunchTool()
    result = tool.execute(application="code; rm -rf /")
    assert result["success"] is False
    assert "Shell metacharacters are forbidden" in result["error"]


def test_application_launch_blocks_shell_interpreters():
    tool = ApplicationLaunchTool()
    result = tool.execute(application="bash -c 'whoami'")
    assert result["success"] is False
    assert "forbidden" in result["error"]


def test_application_launch_unresolved_application():
    tool = ApplicationLaunchTool()
    with patch("shutil.which", return_value=None):
        result = tool.execute(application="completely_unknown_app_xyz")
        assert result["success"] is False
        assert "Could not resolve approved application" in result["error"]


def test_application_launch_unlisted_requires_confirmation():
    tool = ApplicationLaunchTool()
    mock_proc = MagicMock()
    mock_proc.pid = 88888

    with patch("shutil.which", return_value="/usr/bin/some-custom-tool"), \
         patch("tools.safety.ActionSafetyPolicy.request_confirmation", return_value=False):

        result = tool.execute(application="some-custom-tool", confirm=False)
        assert result["success"] is False
        assert result["blocked"] is True
        assert "cancelled by user confirmation" in result["reason"]
