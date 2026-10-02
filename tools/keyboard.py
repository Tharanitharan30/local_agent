from typing import Dict, Any, List, Optional, Union

import config
from tools.base import BaseTool
from tools.input_backend import DesktopInputBackend, get_input_backend, KEYSYM_MAP
from tools.safety import ActionSafetyPolicy
from tools.mouse import run_screen_verification


class KeyboardTool(BaseTool):
    """
    Consolidated keyboard tool for interacting with desktop applications.
    Supports 'type', 'press', and 'hotkey' operations.
    Validates key names, enforces safety policies, and optionally verifies state.
    """

    name = "keyboard"
    description = (
        "Types text, presses individual keys, or executes key combination shortcuts "
        "on the Linux desktop. Supports actions: 'type', 'press', 'hotkey'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["type", "press", "hotkey"],
                "description": "Keyboard action: 'type' (text), 'press' (single key), 'hotkey' (combination).",
            },
            "text": {
                "type": "string",
                "default": "",
                "description": "Text to type into the currently focused window (for 'type' action).",
            },
            "key": {
                "type": "string",
                "default": "",
                "description": "Named key to press (e.g. 'ENTER', 'TAB', 'ESCAPE', 'BACKSPACE', 'DELETE', 'UP', 'DOWN').",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "List of key names for a shortcut combination (e.g. ['CTRL', 'L'] or ['ALT', 'TAB']).",
            },
            "intent": {
                "type": "string",
                "default": "",
                "description": "Optional brief explanation of what the keystroke intends to do.",
            },
            "confirm": {
                "type": "boolean",
                "default": False,
                "description": "Set to true to prompt user for confirmation before typing/pressing.",
            },
            "verify": {
                "type": "boolean",
                "default": False,
                "description": "Whether to perform a screen observation after the keystroke to verify outcome.",
            },
            "expected_state": {
                "type": "string",
                "default": "",
                "description": "Expected visual outcome after keystroke.",
            },
        },
        "required": [],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = [
            "keyboard.action",
            "keyboard_tool",
            "keyboard.type",
            "keyboard.press",
            "keyboard.hotkey",
        ]
        self.backend = backend or get_input_backend()

    def execute(
        self,
        action: str = "type",
        text: str = "",
        key: str = "",
        keys: Optional[Union[List[str], str]] = None,
        intent: str = "",
        confirm: bool = False,
        verify: bool = False,
        expected_state: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the requested keyboard action."""
        action_norm = action.lower().strip()

        # Parse and normalize keys if passed as string or list
        normalized_keys: List[str] = []
        if keys:
            if isinstance(keys, str):
                normalized_keys = [k.strip() for k in keys.replace("+", ",").split(",") if k.strip()]
            elif isinstance(keys, (list, tuple)):
                normalized_keys = [str(k).strip() for k in keys if str(k).strip()]

        # Validate inputs per action
        if action_norm == "type":
            if not text:
                return {"success": False, "error": "Action 'type' requires non-empty 'text' parameter."}
        elif action_norm == "press":
            if not key:
                return {"success": False, "error": "Action 'press' requires non-empty 'key' parameter."}
            key_norm = key.strip().lower()
            if key_norm not in KEYSYM_MAP and len(key) != 1:
                supported = sorted(list(KEYSYM_MAP.keys()))
                return {
                    "success": False,
                    "error": f"Unknown key '{key}'. Supported named keys: {supported}",
                }
        elif action_norm == "hotkey":
            if not normalized_keys:
                return {"success": False, "error": "Action 'hotkey' requires a list of 'keys' (e.g. ['CTRL', 'L'])."}
            for k in normalized_keys:
                kn = k.lower()
                if kn not in KEYSYM_MAP and len(k) != 1:
                    supported = sorted(list(KEYSYM_MAP.keys()))
                    return {
                        "success": False,
                        "error": f"Unknown key in hotkey '{k}'. Supported named keys: {supported}",
                    }
        else:
            return {
                "success": False,
                "error": f"Unknown keyboard action '{action}'. Supported: type, press, hotkey.",
            }

        # Safety & Confirmation check
        should_confirm = False
        reason = ""
        param_target = text if action_norm == "type" else (key if action_norm == "press" else normalized_keys)

        if config.REQUIRE_ACTION_CONFIRMATION:
            should_confirm = True
            reason = "Policy requires confirmation for all actions."
        elif config.CONFIRM_HIGH_IMPACT_ACTIONS:
            is_high, impact_reason = ActionSafetyPolicy.is_high_impact_keyboard_action(
                action_type=action_norm,
                keys_or_text=param_target,
                intent=intent,
                confirm=confirm,
            )
            if is_high:
                should_confirm = True
                reason = impact_reason

        if should_confirm:
            action_desc = f"Keyboard Action: {action_norm.upper()} ({param_target}) - Intent: '{intent or 'none'}'"
            confirmed = ActionSafetyPolicy.request_confirmation(f"{action_desc}\nReason: {reason}")
            if not confirmed:
                return {
                    "success": False,
                    "blocked": True,
                    "reason": "Action cancelled by user confirmation.",
                    "action": action_norm,
                    "target": param_target,
                }

        # Execute action
        try:
            if action_norm == "type":
                self.backend.type_text(text)
                msg = f"Typed text ({len(text)} chars) into active window."
            elif action_norm == "press":
                self.backend.press_key(key)
                msg = f"Pressed key '{key.upper()}'."
            elif action_norm == "hotkey":
                self.backend.hotkey(normalized_keys)
                msg = f"Executed hotkey: {' + '.join(k.upper() for k in normalized_keys)}."

            result: Dict[str, Any] = {
                "success": True,
                "action": action_norm,
                "message": msg,
            }

            # Optional post-action screen verification
            if verify or expected_state:
                state_query = expected_state or f"Outcome of keyboard {action_norm}"
                verification = run_screen_verification(state_query)
                result["verification"] = verification

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Keyboard execution failed: {str(e)}",
                "action": action_norm,
            }


class KeyboardTypeTool(BaseTool):
    """Types supplied text into the active/focused desktop window."""

    name = "keyboard.type"
    description = "Types the supplied text string into the currently focused window or text field."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to type."},
            "intent": {"type": "string", "default": "", "description": "Purpose of the typed text."},
            "verify": {"type": "boolean", "default": False},
            "expected_state": {"type": "string", "default": ""},
        },
        "required": ["text"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["type"]
        self.backend = backend or get_input_backend()

    def execute(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = KeyboardTool(backend=self.backend)
        return tool.execute(action="type", text=text, **kwargs)


class KeyboardPressTool(BaseTool):
    """Presses a single key by name."""

    name = "keyboard.press"
    description = "Presses a single key by name (e.g. 'ENTER', 'TAB', 'ESCAPE', 'BACKSPACE', 'DELETE', 'UP', 'DOWN')."
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Key name to press (e.g. 'ENTER', 'TAB', 'ESCAPE')."},
            "intent": {"type": "string", "default": ""},
            "verify": {"type": "boolean", "default": False},
            "expected_state": {"type": "string", "default": ""},
        },
        "required": ["key"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["press"]
        self.backend = backend or get_input_backend()

    def execute(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = KeyboardTool(backend=self.backend)
        return tool.execute(action="press", key=key, **kwargs)


class KeyboardHotkeyTool(BaseTool):
    """Executes a key combination shortcut."""

    name = "keyboard.hotkey"
    description = "Executes a keyboard shortcut combination using specified keys (e.g. ['CTRL', 'L'] or ['ALT', 'TAB'])."
    input_schema = {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key names to press together (e.g. ['CTRL', 'L']).",
            },
            "intent": {"type": "string", "default": ""},
            "verify": {"type": "boolean", "default": False},
            "expected_state": {"type": "string", "default": ""},
        },
        "required": ["keys"],
    }

    def __init__(self, backend: Optional[DesktopInputBackend] = None):
        super().__init__()
        self.aliases = ["hotkey"]
        self.backend = backend or get_input_backend()

    def execute(self, keys: Union[List[str], str], **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("action", None)
        tool = KeyboardTool(backend=self.backend)
        return tool.execute(action="hotkey", keys=keys, **kwargs)
