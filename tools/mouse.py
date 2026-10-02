from typing import Dict, Any, Optional

import config
from tools.base import BaseTool
from tools.input_backend import DesktopInputBackend, get_input_backend
from tools.safety import ActionSafetyPolicy


def run_screen_verification(expected_state: str) -> Dict[str, Any]:
    """Runs a quick screen observation to verify expected state post-action."""
    try:
        from tools.screen import ScreenTool
        screen_tool = ScreenTool()
        obs = screen_tool.execute(query=f"Verify if the expected change or state is visible: {expected_state}")
        if obs.get("success"):
            desc = obs.get("description", "")
            return {
                "verified": True,
                "observation": desc,
            }
        return {
            "verified": False,
            "error": obs.get("error", "Screen observation failed during verification"),
        }
    except Exception as e:
        return {
            "verified": False,
            "error": f"Verification error: {str(e)}",
        }


class MouseTool(BaseTool):
    """
    Consolidated mouse tool for interacting with the desktop GUI.
    Supports move, click, double_click, right_click, and scroll operations.
    Validates screen bounds, enforces safety policies, and optionally verifies state.
    """

    name = "mouse"
    description = (
        "Controls the mouse on the Linux desktop. Supports actions: "
        "'move', 'click', 'double_click', 'right_click', 'scroll'. Coordinates must be in screen bounds."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["move", "click", "double_click", "right_click", "scroll"],
                "description": "Mouse action to perform.",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate in screen pixels (0 <= x < screen_width).",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate in screen pixels (0 <= y < screen_height).",
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "default": "left",
                "description": "Mouse button for click action.",
            },
            "clicks": {
                "type": "integer",
                "default": 1,
                "description": "Number of clicks to perform.",
            },
            "steps": {
                "type": "integer",
                "default": 1,
                "description": "Scroll discrete steps. Positive scrolls down; negative scrolls up.",
            },
            "intent": {
                "type": "string",
                "default": "",
                "description": "Optional brief explanation of what is being clicked (e.g. 'close window', 'submit form').",
            },
            "confirm": {
                "type": "boolean",
                "default": False,
                "description": "Set to true to explicitly prompt user for confirmation before executing.",
            },
            "verify": {
                "type": "boolean",
                "default": False,
                "description": "Whether to perform a screen observation after the action to verify outcome.",
            },
            "expected_state": {
                "type": "string",
                "default": "",
                "description": "Expected visual outcome after action to check if verify is enabled.",
            },
        },
        "required": [],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = [
            "mouse.action",
            "mouse_tool",
            "mouse.move",
            "mouse.click",
            "mouse.double_click",
            "mouse.right_click",
            "mouse.scroll",
        ]
        self.backend = backend or get_input_backend()

    def execute(
        self,
        action: str = "click",
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
        steps: int = 1,
        intent: str = "",
        confirm: bool = False,
        verify: bool = False,
        expected_state: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the requested mouse action."""
        action_norm = action.lower().strip()
        try:
            screen_size = self.backend.get_screen_size()
            if isinstance(screen_size, (tuple, list)) and len(screen_size) >= 2:
                screen_w, screen_h = int(screen_size[0]), int(screen_size[1])
            else:
                screen_w, screen_h = 1920, 1080
        except Exception:
            screen_w, screen_h = 1920, 1080

        # Coordinate validation for actions that require x, y
        if action_norm in ("move", "click", "double_click", "right_click"):
            if x is None or y is None:
                return {
                    "success": False,
                    "error": f"Action '{action}' requires both 'x' and 'y' screen coordinates.",
                    "screen_bounds": {"width": screen_w, "height": screen_h},
                }

            valid, err = self.backend.validate_coordinates(x, y)
            if not valid:
                return {
                    "success": False,
                    "error": err,
                    "screen_bounds": {"width": screen_w, "height": screen_h},
                }

        # Safety & Confirmation check
        should_confirm = False
        reason = ""
        if config.REQUIRE_ACTION_CONFIRMATION:
            should_confirm = True
            reason = "Policy requires confirmation for all actions."
        elif config.CONFIRM_HIGH_IMPACT_ACTIONS:
            is_high, impact_reason = ActionSafetyPolicy.is_high_impact_mouse_action(
                action=action_norm,
                x=x if x is not None else 0,
                y=y if y is not None else 0,
                intent=intent,
                confirm=confirm,
            )
            if is_high:
                should_confirm = True
                reason = impact_reason

        if should_confirm:
            action_desc = f"Mouse Action: {action_norm.upper()} at ({x}, {y}) - Intent: '{intent or 'none'}'"
            confirmed = ActionSafetyPolicy.request_confirmation(f"{action_desc}\nReason: {reason}")
            if not confirmed:
                return {
                    "success": False,
                    "blocked": True,
                    "reason": "Action cancelled by user confirmation.",
                    "action": action_norm,
                    "x": x,
                    "y": y,
                }

        # Execute action
        try:
            if action_norm == "move":
                assert x is not None and y is not None
                self.backend.move(x, y)
                msg = f"Moved cursor to ({x}, {y})."
            elif action_norm == "click":
                assert x is not None and y is not None
                self.backend.click(x, y, button=button, clicks=clicks)
                msg = f"Clicked {button} button ({clicks} time(s)) at ({x}, {y})."
            elif action_norm == "double_click":
                assert x is not None and y is not None
                self.backend.click(x, y, button="left", clicks=2)
                msg = f"Double-clicked at ({x}, {y})."
            elif action_norm == "right_click":
                assert x is not None and y is not None
                self.backend.click(x, y, button="right", clicks=1)
                msg = f"Right-clicked at ({x}, {y})."
            elif action_norm == "scroll":
                self.backend.scroll(steps=steps, x=x, y=y)
                msg = f"Scrolled {'down' if steps > 0 else 'up'} by {abs(steps)} step(s)."
            else:
                return {
                    "success": False,
                    "error": f"Unknown mouse action '{action}'. Supported: move, click, double_click, right_click, scroll.",
                }

            result: Dict[str, Any] = {
                "success": True,
                "action": action_norm,
                "message": msg,
                "x": x,
                "y": y,
            }

            # Optional post-action screen verification
            if verify or expected_state:
                state_query = expected_state or f"Outcome of mouse {action_norm}"
                verification = run_screen_verification(state_query)
                result["verification"] = verification

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Mouse execution failed: {str(e)}",
                "action": action_norm,
            }


class MouseMoveTool(BaseTool):
    """Moves mouse cursor to specified coordinates."""

    name = "mouse.move"
    description = "Moves mouse cursor to specified screen coordinates (x, y)."
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate in pixels."},
            "y": {"type": "integer", "description": "Y coordinate in pixels."},
        },
        "required": ["x", "y"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["move"]
        self.backend = backend or get_input_backend()

    def execute(self, x: int, y: int, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = MouseTool(backend=self.backend)
        return tool.execute(action="move", x=x, y=y, **kwargs)


class MouseClickTool(BaseTool):
    """Clicks the specified screen coordinate."""

    name = "mouse.click"
    description = "Clicks the specified screen coordinate (x, y) with left, right, or middle button."
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate in pixels."},
            "y": {"type": "integer", "description": "Y coordinate in pixels."},
            "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
            "intent": {"type": "string", "default": "", "description": "What is being clicked."},
            "verify": {"type": "boolean", "default": False, "description": "Whether to verify state after click."},
            "expected_state": {"type": "string", "default": "", "description": "Expected visual outcome."},
        },
        "required": ["x", "y"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["click"]
        self.backend = backend or get_input_backend()

    def execute(self, x: int, y: int, button: str = "left", **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = MouseTool(backend=self.backend)
        return tool.execute(action="click", x=x, y=y, button=button, **kwargs)


class MouseDoubleClickTool(BaseTool):
    """Double-clicks the specified screen coordinate."""

    name = "mouse.double_click"
    description = "Double-clicks the specified screen coordinate (x, y)."
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate in pixels."},
            "y": {"type": "integer", "description": "Y coordinate in pixels."},
            "intent": {"type": "string", "default": ""},
            "verify": {"type": "boolean", "default": False},
            "expected_state": {"type": "string", "default": ""},
        },
        "required": ["x", "y"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["double_click"]
        self.backend = backend or get_input_backend()

    def execute(self, x: int, y: int, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = MouseTool(backend=self.backend)
        return tool.execute(action="double_click", x=x, y=y, **kwargs)


class MouseRightClickTool(BaseTool):
    """Right-clicks the specified screen coordinate."""

    name = "mouse.right_click"
    description = "Right-clicks the specified screen coordinate (x, y) to open context menus."
    input_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate in pixels."},
            "y": {"type": "integer", "description": "Y coordinate in pixels."},
            "intent": {"type": "string", "default": ""},
            "verify": {"type": "boolean", "default": False},
            "expected_state": {"type": "string", "default": ""},
        },
        "required": ["x", "y"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["right_click"]
        self.backend = backend or get_input_backend()

    def execute(self, x: int, y: int, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = MouseTool(backend=self.backend)
        return tool.execute(action="right_click", x=x, y=y, **kwargs)


class MouseScrollTool(BaseTool):
    """Scrolls vertically on the screen."""

    name = "mouse.scroll"
    description = "Scrolls vertically by discrete steps (positive down, negative up)."
    input_schema = {
        "type": "object",
        "properties": {
            "steps": {"type": "integer", "description": "Number of scroll steps. Positive down, negative up."},
            "x": {"type": "integer", "description": "Optional X coordinate to scroll over."},
            "y": {"type": "integer", "description": "Optional Y coordinate to scroll over."},
            "verify": {"type": "boolean", "default": False},
            "expected_state": {"type": "string", "default": ""},
        },
        "required": ["steps"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["scroll"]
        self.backend = backend or get_input_backend()

    def execute(self, steps: int, x: Optional[int] = None, y: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = MouseTool(backend=self.backend)
        return tool.execute(action="scroll", steps=steps, x=x, y=y, **kwargs)
