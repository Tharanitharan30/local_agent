import json
import os
import signal
import subprocess
import time
from typing import Dict, Any, List, Optional

import config
from tools.base import BaseTool
from tools.input_backend import get_input_backend, DesktopInputBackend
from tools.safety import ActionSafetyPolicy


# Python script executed via system Python (/usr/bin/python3) where PyGObject & Atspi are present
ATSPI_QUERY_SCRIPT = '''
import json, sys
try:
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi

    def check_focus(obj, depth=0):
        if not obj or depth > 4:
            return False
        try:
            ss = obj.get_state_set()
            if ss and ss.contains(Atspi.StateType.FOCUSED):
                return True
            for i in range(min(obj.get_child_count(), 20)):
                if check_focus(obj.get_child_at_index(i), depth + 1):
                    return True
        except Exception:
            pass
        return False

    root = Atspi.get_desktop(0)
    screen_w, screen_h = 1920, 1080
    windows = []

    IGNORE_APPS = {'gnome-shell', 'gjs', 'ibus-extension-gtk3', 'update-notifier', 'evolution-alarm-notify'}
    IGNORE_TITLES = {'Desktop Icons 1', 'Main stage'}

    for i in range(root.get_child_count()):
        child = root.get_child_at_index(i)
        if not child:
            continue
        app_name = child.get_name()
        if app_name in IGNORE_APPS:
            continue

        for j in range(child.get_child_count()):
            win = child.get_child_at_index(j)
            if not win or win.get_role_name() not in ('frame', 'window', 'dialog'):
                continue
            title = win.get_name()
            if not title or title in IGNORE_TITLES:
                continue

            states = [s.value_nick for s in win.get_state_set().get_states()]
            comp = win.get_component()
            rect = comp.get_extents(Atspi.CoordType.SCREEN) if comp else None

            w = rect.width if rect else 0
            h = rect.height if rect else 0
            if w < 50 or h < 50:
                continue

            pid = getattr(win, 'get_process_id', lambda: None)()
            has_focus = check_focus(win)
            is_active = ('active' in states) or has_focus
            is_minimized = ('iconified' in states) or not ('showing' in states)
            is_maximized = (w >= screen_w and h >= screen_h - 60)

            app_slug = app_name.lower().replace(" ", "_")
            win_id = f"win_{app_slug}_{pid or j}"

            windows.append({
                "window_id": win_id,
                "title": title,
                "application": app_name,
                "pid": pid,
                "active": is_active,
                "focused": has_focus,
                "minimized": is_minimized,
                "maximized": is_maximized,
                "position": {"x": rect.x if rect else 0, "y": rect.y if rect else 0},
                "size": {"width": w, "height": h},
            })

    print(json.dumps({"success": True, "windows": windows}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e), "windows": []}))
'''


ATSPI_ACTION_SCRIPT = '''
import json, sys
target_query = sys.argv[1].lower() if len(sys.argv) > 1 else ""
action_type = sys.argv[2].lower() if len(sys.argv) > 2 else ""

try:
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi

    root = Atspi.get_desktop(0)
    for i in range(root.get_child_count()):
        child = root.get_child_at_index(i)
        if not child:
            continue
        app_name = child.get_name()
        for j in range(child.get_child_count()):
            win = child.get_child_at_index(j)
            if not win or win.get_role_name() not in ('frame', 'window', 'dialog'):
                continue
            title = win.get_name()
            pid = getattr(win, 'get_process_id', lambda: None)()
            app_slug = app_name.lower().replace(" ", "_")
            win_id = f"win_{app_slug}_{pid or j}"

            # Match target window
            if (target_query == win_id.lower() or
                target_query in title.lower() or
                target_query in app_name.lower()):

                act = win.get_action()
                if not act:
                    print(json.dumps({"success": False, "error": "No action interface available on window"}))
                    sys.exit(0)

                for idx in range(act.get_n_actions()):
                    aname = act.get_action_name(idx).lower()
                    if action_type in aname:
                        res = act.do_action(idx)
                        print(json.dumps({
                            "success": res,
                            "action": aname,
                            "title": title,
                            "application": app_name,
                            "window_id": win_id,
                            "pid": pid,
                        }))
                        sys.exit(0)

                print(json.dumps({
                    "success": False,
                    "error": f"Action '{action_type}' not found for window '{title}'",
                    "window_id": win_id,
                    "pid": pid,
                }))
                sys.exit(0)

    print(json.dumps({"success": False, "error": f"Window matching '{target_query}' not found."}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
'''


def query_desktop_windows(timeout: float = 4.0) -> List[Dict[str, Any]]:
    """
    Queries visible and managed desktop windows using Linux AT-SPI over system Python.
    Falls back gracefully to empty list if inaccessible.
    """
    try:
        proc = subprocess.run(
            ["/usr/bin/python3", "-c", ATSPI_QUERY_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stdout:
            data = json.loads(proc.stdout.strip())
            if data.get("success"):
                return data.get("windows", [])
    except Exception:
        pass
    return []


def get_active_desktop_window() -> Optional[Dict[str, Any]]:
    """Returns the currently active / focused application window."""
    windows = query_desktop_windows()
    if not windows:
        return None

    # First priority: explicitly focused window
    for w in windows:
        if w.get("focused") and not w.get("minimized"):
            return w

    # Second priority: active window that is not minimized
    for w in windows:
        if w.get("active") and not w.get("minimized"):
            return w

    # Third priority: first visible window
    return windows[0] if windows else None


def execute_atspi_action(target_query: str, action_type: str, timeout: float = 4.0) -> Dict[str, Any]:
    """Executes an AT-SPI window action (activate, minimize, maximize, close)."""
    try:
        proc = subprocess.run(
            ["/usr/bin/python3", "-c", ATSPI_ACTION_SCRIPT, target_query, action_type],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stdout:
            return json.loads(proc.stdout.strip())
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": False, "error": "No response from desktop window action helper."}


class WindowListTool(BaseTool):
    """Lists visible and managed windows on the Linux desktop."""

    name = "window.list"
    description = (
        "Returns structured information about all currently open/visible windows on the desktop, "
        "including title, application name, window identifier, active status, position, and dimensions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "default": "",
                "description": "Optional filter by application name (e.g. 'code', 'edge', 'calculator').",
            },
        },
    }

    def __init__(self):
        super().__init__()
        self.aliases = ["windows", "list_windows", "window_list"]

    def execute(self, application: str = "", **kwargs: Any) -> Dict[str, Any]:
        windows = query_desktop_windows()
        if application:
            app_norm = application.strip().lower()
            windows = [
                w for w in windows
                if app_norm in w.get("application", "").lower() or app_norm in w.get("title", "").lower()
            ]

        return {
            "success": True,
            "count": len(windows),
            "windows": windows,
        }


class WindowGetActiveTool(BaseTool):
    """Retrieves structured information about the currently active/focused window."""

    name = "window.get_active"
    description = "Returns structured details about the currently active or focused desktop window."
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self):
        super().__init__()
        self.aliases = ["get_active_window", "active_window", "window.active"]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        active = get_active_desktop_window()
        if active:
            return {
                "success": True,
                "window": active,
                "title": active.get("title"),
                "application": active.get("application"),
                "window_id": active.get("window_id"),
            }
        return {
            "success": False,
            "error": "No active application window could be determined.",
            "window": None,
        }


class WindowFocusTool(BaseTool):
    """Brings a target window to the foreground and grants it focus."""

    name = "window.focus"
    description = (
        "Focuses and raises the specified window by window_id, application name, or title. "
        "Verifies that the window became active."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "window_id": {
                "type": "string",
                "default": "",
                "description": "Identifier of the window to focus (from window.list).",
            },
            "application": {
                "type": "string",
                "default": "",
                "description": "Optional application name to focus (e.g. 'code', 'edge', 'resources').",
            },
            "title": {
                "type": "string",
                "default": "",
                "description": "Optional window title or subtitle substring.",
            },
            "verify": {
                "type": "boolean",
                "default": True,
                "description": "Whether to verify active window state after focusing.",
            },
        },
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["focus_window", "window_focus", "switch_window"]
        self.backend = backend or get_input_backend()

    def execute(
        self,
        window_id: str = "",
        application: str = "",
        title: str = "",
        verify: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target = (window_id or application or title).strip()
        if not target:
            return {"success": False, "error": "Must provide 'window_id', 'application', or 'title' to focus."}

        # Locate target window in current desktop list
        windows = query_desktop_windows()
        matched = None
        for w in windows:
            if target == w.get("window_id"):
                matched = w
                break
            if target.lower() in w.get("application", "").lower():
                matched = w
                break
            if target.lower() in w.get("title", "").lower():
                matched = w
                break

        if not matched:
            return {
                "success": False,
                "error": f"No open window matching '{target}'. Available windows: {[w.get('title') for w in windows]}",
            }

        # Step 1: Attempt native activate action via AT-SPI
        action_res = execute_atspi_action(matched.get("window_id", target), "activate")

        # Step 2: Reinforce focus by clicking the window header if coordinates available
        pos = matched.get("position", {})
        size = matched.get("size", {})
        x = pos.get("x", 0) + min(120, max(20, size.get("width", 200) // 2))
        y = pos.get("y", 0) + 20
        try:
            self.backend.click(x, y)
        except Exception:
            pass

        time.sleep(0.3)

        # Verification step
        verified = False
        if verify:
            active = get_active_desktop_window()
            if active and (
                active.get("window_id") == matched.get("window_id") or
                matched.get("application", "").lower() in active.get("application", "").lower() or
                matched.get("title", "").lower() in active.get("title", "").lower()
            ):
                verified = True

        return {
            "success": True,
            "window_id": matched.get("window_id"),
            "title": matched.get("title"),
            "application": matched.get("application"),
            "verified": verified,
            "message": f"Focused window '{matched.get('title')}' ({matched.get('application')}).",
        }


class WindowMinimizeTool(BaseTool):
    """Minimizes a target or currently active window."""

    name = "window.minimize"
    description = "Minimizes the specified window (or the currently active window if unspecified)."
    input_schema = {
        "type": "object",
        "properties": {
            "window_id": {
                "type": "string",
                "default": "",
                "description": "Identifier of the window to minimize.",
            },
            "application": {"type": "string", "default": ""},
            "title": {"type": "string", "default": ""},
        },
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["minimize_window", "window_minimize"]
        self.backend = backend or get_input_backend()

    def execute(self, window_id: str = "", application: str = "", title: str = "", **kwargs: Any) -> Dict[str, Any]:
        target = (window_id or application or title).strip()
        if not target:
            active = get_active_desktop_window()
            if active:
                target = active.get("window_id", "")

        if not target:
            return {"success": False, "error": "No window specified and no active window detected."}

        # Step 1: AT-SPI minimize action
        res = execute_atspi_action(target, "minimize")
        if res.get("success"):
            return {
                "success": True,
                "message": f"Minimized window matching '{target}'.",
                "action": "minimize",
            }

        # Step 2: Desktop shortcut fallback (Super + h)
        try:
            self.backend.hotkey(["super", "h"])
            return {
                "success": True,
                "message": f"Dispatched minimize shortcut for window matching '{target}'.",
                "action": "shortcut",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to minimize window: {str(e)}"}


class WindowMaximizeTool(BaseTool):
    """Maximizes or un-maximizes a target or currently active window."""

    name = "window.maximize"
    description = "Maximizes the specified window (or toggles maximization)."
    input_schema = {
        "type": "object",
        "properties": {
            "window_id": {"type": "string", "default": ""},
            "application": {"type": "string", "default": ""},
            "title": {"type": "string", "default": ""},
        },
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["maximize_window", "window_maximize"]
        self.backend = backend or get_input_backend()

    def execute(self, window_id: str = "", application: str = "", title: str = "", **kwargs: Any) -> Dict[str, Any]:
        target = (window_id or application or title).strip()
        if not target:
            active = get_active_desktop_window()
            if active:
                target = active.get("window_id", "")

        if not target:
            return {"success": False, "error": "No window specified and no active window detected."}

        # Step 1: AT-SPI toggle-maximized action
        res = execute_atspi_action(target, "maximized")
        if res.get("success"):
            return {
                "success": True,
                "message": f"Toggled maximization for window matching '{target}'.",
                "action": "maximize",
            }

        # Step 2: Desktop shortcut fallback (Super + Up)
        try:
            self.backend.hotkey(["super", "up"])
            return {
                "success": True,
                "message": f"Dispatched maximize shortcut for window matching '{target}'.",
                "action": "shortcut",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to maximize window: {str(e)}"}


class WindowCloseTool(BaseTool):
    """Closes a target window. This is a consequential action requiring user confirmation."""

    name = "window.close"
    description = (
        "Gracefully closes the specified window by window_id, application, or title. "
        "Requires explicit confirmation to prevent accidental data loss. Does not force-kill processes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "window_id": {"type": "string", "default": ""},
            "application": {"type": "string", "default": ""},
            "title": {"type": "string", "default": ""},
            "confirm": {"type": "boolean", "default": False, "description": "Explicit confirmation flag."},
        },
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["close_window", "window_close"]
        self.backend = backend or get_input_backend()

    def execute(
        self,
        window_id: str = "",
        application: str = "",
        title: str = "",
        confirm: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target = (window_id or application or title).strip()
        if not target:
            active = get_active_desktop_window()
            if active:
                target = active.get("window_id", "")

        if not target:
            return {"success": False, "error": "No window specified and no active window detected to close."}

        # Locate window info for descriptive confirmation
        windows = query_desktop_windows()
        matched = None
        for w in windows:
            if target == w.get("window_id") or target.lower() in w.get("title", "").lower() or target.lower() in w.get("application", "").lower():
                matched = w
                break

        desc_title = matched.get("title") if matched else target
        desc_app = matched.get("application") if matched else "unknown app"

        # Consequential confirmation check
        should_confirm = config.REQUIRE_ACTION_CONFIRMATION or config.CONFIRM_HIGH_IMPACT_ACTIONS
        if should_confirm and not confirm:
            prompt_desc = f"Window Close: '{desc_title}' ({desc_app})\nReason: Closing a window can cause unsaved data loss."
            confirmed = ActionSafetyPolicy.request_confirmation(prompt_desc)
            if not confirmed:
                return {
                    "success": False,
                    "blocked": True,
                    "reason": "Window close action cancelled by user confirmation.",
                    "window": desc_title,
                }

        # Step 1: AT-SPI graceful close action
        res = execute_atspi_action(target, "close")
        if res.get("success"):
            return {
                "success": True,
                "message": f"Closed window '{desc_title}' via desktop close action.",
                "action": "window.close",
            }

        # Step 2: Focus and Alt + F4 graceful shortcut
        try:
            if matched:
                WindowFocusTool(backend=self.backend).execute(window_id=matched.get("window_id"))
            self.backend.hotkey(["alt", "f4"])
            time.sleep(0.5)
            return {
                "success": True,
                "message": f"Dispatched graceful close shortcut (Alt+F4) for window '{desc_title}'.",
                "action": "alt_f4",
            }
        except Exception:
            pass

        # Step 3: Graceful SIGTERM if PID known (never SIGKILL / kill -9)
        if matched and matched.get("pid"):
            try:
                os.kill(matched["pid"], signal.SIGTERM)
                return {
                    "success": True,
                    "message": f"Sent graceful termination signal (SIGTERM) to process PID {matched['pid']} ({desc_app}).",
                    "action": "sigterm",
                }
            except Exception as e:
                return {"success": False, "error": f"Failed to signal process: {str(e)}"}

        return {"success": False, "error": f"Failed to close window '{desc_title}'."}
