import os
import re
import shutil
import subprocess
import time
from typing import Dict, Any, List, Optional

import config
from tools.base import BaseTool
from tools.safety import ActionSafetyPolicy


DANGEROUS_SHELL_CHARS = re.compile(r"[;&|><$`\\\"'\n\r]")


class ApplicationListTool(BaseTool):
    """Lists currently running applications on the Linux desktop."""

    name = "application.list"
    description = "Returns a concise list of currently running GUI applications detected on the desktop."
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self):
        super().__init__()
        self.aliases = ["list_applications", "running_applications", "applications", "apps"]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from tools.windows import query_desktop_windows

        windows = query_desktop_windows()
        app_map: Dict[str, Dict[str, Any]] = {}

        for w in windows:
            app_name = w.get("application", "unknown")
            if app_name not in app_map:
                app_map[app_name] = {
                    "application": app_name,
                    "running": True,
                    "window_count": 0,
                    "pid": w.get("pid"),
                    "windows": [],
                }
            app_map[app_name]["window_count"] += 1
            app_map[app_name]["windows"].append(w.get("title", ""))

        apps = list(app_map.values())
        return {
            "success": True,
            "count": len(apps),
            "applications": apps,
        }


class ApplicationLaunchTool(BaseTool):
    """
    Launches desktop applications using a controlled, allowlisted mechanism.
    Rejects arbitrary shell strings or injection attacks.
    """

    name = "application.launch"
    description = (
        "Launches an allowed desktop application by name or category (e.g. 'calculator', 'terminal', "
        "'code', 'text editor', 'file manager', 'browser'). Verifies application launch."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": "Application name or category to launch (e.g. 'calculator', 'terminal', 'code', 'browser').",
            },
            "verify": {
                "type": "boolean",
                "default": True,
                "description": "Whether to verify that the application window opened after launch.",
            },
            "confirm": {
                "type": "boolean",
                "default": False,
                "description": "Explicit confirmation flag for unlisted application launch.",
            },
        },
        "required": ["application"],
    }

    def __init__(self):
        super().__init__()
        self.aliases = ["launch_application", "open_application", "launch"]

    def resolve_executable(self, requested_app: str) -> Optional[str]:
        """
        Resolves requested application name or category to an executable on the system.
        Consults APPLICATION_ALLOWLIST first, then desktop applications.
        """
        app_norm = requested_app.strip().lower()

        # Check direct allowlist categories
        allowlist = getattr(config, "APPLICATION_ALLOWLIST", {})
        if app_norm in allowlist:
            for candidate in allowlist[app_norm]:
                found = shutil.which(candidate)
                if found:
                    return found

        # Check if the requested string is directly one of the allowed binaries
        for category, candidates in allowlist.items():
            if app_norm in [c.lower() for c in candidates]:
                found = shutil.which(app_norm)
                if found:
                    return found

        # Fallback check: check if it matches an installed executable directly if safe
        found = shutil.which(app_norm)
        if found:
            return found

        return None

    def execute(
        self,
        application: str = "",
        verify: bool = True,
        confirm: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not application or not application.strip():
            return {"success": False, "error": "Application name parameter must not be empty."}

        app_raw = application.strip()

        # Safety Check 1: Reject shell metacharacters and command injection
        if DANGEROUS_SHELL_CHARS.search(app_raw):
            return {
                "success": False,
                "error": f"Rejected potentially unsafe application parameter '{app_raw}'. Shell metacharacters are forbidden.",
            }

        # Safety Check 2: Reject shell invocations
        lower_app = app_raw.lower()
        if lower_app.startswith(("bash", "sh", "zsh", "python", "perl", "eval", "exec", "sudo", "rm")):
            return {
                "success": False,
                "error": f"Launching shell interpreters or system commands via application.launch is forbidden: '{app_raw}'.",
            }

        executable = self.resolve_executable(app_raw)
        if not executable:
            approved = sorted(list(getattr(config, "APPLICATION_ALLOWLIST", {}).keys()))
            return {
                "success": False,
                "error": (
                    f"Could not resolve approved application for '{app_raw}'. "
                    f"Supported categories / applications include: {approved}"
                ),
            }

        # Safety Check 3: Check if unlisted application requires confirmation
        is_allowlisted = False
        allowlist = getattr(config, "APPLICATION_ALLOWLIST", {})
        for cat, cands in allowlist.items():
            if lower_app == cat.lower() or os.path.basename(executable).lower() in [c.lower() for c in cands]:
                is_allowlisted = True
                break

        if not is_allowlisted:
            prompt_desc = f"Application Launch: '{executable}'\nReason: Application is not in the default allowlist."
            confirmed = ActionSafetyPolicy.request_confirmation(prompt_desc)
            if not confirmed:
                return {
                    "success": False,
                    "blocked": True,
                    "reason": "Application launch cancelled by user confirmation.",
                    "application": app_raw,
                }

        # Launch the application in a detached session
        try:
            proc = subprocess.Popen(
                [executable],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            return {"success": False, "error": f"Failed to launch '{executable}': {str(e)}"}

        # Verification step: Poll window list for the new window/application
        verified = False
        matched_window = None
        if verify:
            from tools.windows import query_desktop_windows

            start_poll = time.time()
            exe_base = os.path.basename(executable).lower()

            while time.time() - start_poll < 2.5:
                time.sleep(0.5)
                windows = query_desktop_windows()
                for w in windows:
                    app_name = w.get("application", "").lower()
                    title = w.get("title", "").lower()
                    if (
                        exe_base in app_name or
                        lower_app in app_name or
                        lower_app in title or
                        (w.get("pid") == proc.pid)
                    ):
                        verified = True
                        matched_window = w
                        break
                if verified:
                    break

        return {
            "success": True,
            "application": app_raw,
            "executable": executable,
            "pid": proc.pid,
            "verified": verified,
            "window": matched_window,
            "message": f"Successfully launched {app_raw} ({os.path.basename(executable)}).",
        }
