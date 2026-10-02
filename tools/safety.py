import re
import sys
from typing import Dict, Any, List, Optional, Tuple

from rich.console import Console

import config

console = Console()

# Keywords indicating consequential / high-impact operations
HIGH_IMPACT_INTENTS = [
    r"\bdelete\b",
    r"\bremove\b",
    r"\berase\b",
    r"\bformat\b",
    r"\bdestroy\b",
    r"\bdrop\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpower\s*off\b",
    r"\bpay\b",
    r"\bpurchase\b",
    r"\bbuy\b",
    r"\bcheckout\b",
    r"\btransfer\b",
    r"\bsend\b",
    r"\bsubmit\b",
    r"\bpassword\b",
    r"\bsecurity\b",
]

HIGH_IMPACT_HOTKEYS = [
    {"ctrl", "alt", "del"},
    {"ctrl", "alt", "delete"},
    {"alt", "f4"},
    {"ctrl", "w"},
    {"ctrl", "q"},
    {"shift", "delete"},
]

DANGEROUS_TYPING_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0\b",
    r"\bsudo\b",
]


class ActionSafetyPolicy:
    """
    Evaluates actions against safety policies and executes terminal confirmation
    when high-impact or potentially consequential actions are detected.
    """

    @staticmethod
    def is_high_impact_mouse_action(
        action: str,
        x: int,
        y: int,
        intent: str = "",
        confirm: bool = False,
    ) -> Tuple[bool, str]:
        """Checks if a mouse action triggers high-impact confirmation."""
        if confirm:
            return True, f"Explicit confirmation requested for {action} at ({x}, {y})."

        combined = f"{action} {intent}".lower()
        for pattern in HIGH_IMPACT_INTENTS:
            if re.search(pattern, combined):
                return True, f"Potentially consequential action detected matching '{pattern}': {intent or action}"

        return False, ""

    @staticmethod
    def is_high_impact_keyboard_action(
        action_type: str,
        keys_or_text: Any,
        intent: str = "",
        confirm: bool = False,
    ) -> Tuple[bool, str]:
        """Checks if a keyboard action (type/press/hotkey) triggers high-impact confirmation."""
        if confirm:
            return True, f"Explicit confirmation requested for keyboard {action_type}."

        # Check hotkey combinations
        if action_type == "hotkey":
            if isinstance(keys_or_text, (list, tuple)):
                key_set = {str(k).strip().lower() for k in keys_or_text}
                for dangerous in HIGH_IMPACT_HOTKEYS:
                    if dangerous.issubset(key_set):
                        return True, f"High-impact key combination detected: {keys_or_text}"

        # Check single keypress
        if action_type == "press" and str(keys_or_text).strip().lower() in ("delete", "del"):
            return True, "Potentially destructive Delete key press detected."

        # Check typed text
        if action_type == "type" and isinstance(keys_or_text, str):
            for pattern in DANGEROUS_TYPING_PATTERNS:
                if re.search(pattern, keys_or_text, re.IGNORECASE):
                    return True, f"Potentially destructive text input detected matching '{pattern}'."

        combined = f"{action_type} {str(keys_or_text)} {intent}".lower()
        for pattern in HIGH_IMPACT_INTENTS:
            if re.search(pattern, combined):
                return True, f"Consequential operation intent detected matching '{pattern}': {intent}"

        return False, ""

    @staticmethod
    def request_confirmation(description: str) -> bool:
        """Prompts user in terminal for explicit yes/no confirmation."""
        console.print(f"\n[bold yellow]╔══════════════════ [ZIA ACTION CONFIRMATION] ══════════════════╗[/bold yellow]")
        console.print(f"[bold white]{description}[/bold white]")
        console.print(f"[bold yellow]╚════════════════════════════════════════════════════════════════╝[/bold yellow]")
        sys.stdout.flush()

        try:
            prompt_text = "Execute? [y/N]: "
            response = input(prompt_text).strip().lower()
            confirmed = response in ("y", "yes")
            if not confirmed:
                console.print("[bold red][ACTION DENIED][/bold red] Operation cancelled by user.")
            else:
                console.print("[bold green][ACTION CONFIRMED][/bold green] Proceeding with operation.")
            sys.stdout.flush()
            return confirmed
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red][ACTION DENIED][/bold red] Operation cancelled.")
            sys.stdout.flush()
            return False


class ActionLimitTracker:
    """
    Enforces action limits to prevent runaway loops:
    - Maximum total actions per task (MAX_ACTIONS_PER_TASK).
    - Maximum repeated identical action without state change (MAX_RETRIES_PER_ACTION).
    """

    def __init__(
        self,
        max_actions: int = config.MAX_ACTIONS_PER_TASK,
        max_retries: int = config.MAX_RETRIES_PER_ACTION,
    ):
        self.max_actions = max_actions
        self.max_retries = max_retries
        self.action_count = 0
        self.last_action_signature: Optional[str] = None
        self.consecutive_repeats = 0

    def reset(self) -> None:
        """Resets counters at the start of each user request."""
        self.action_count = 0
        self.last_action_signature = None
        self.consecutive_repeats = 0

    def record_and_validate(self, action_signature: str) -> Tuple[bool, Optional[str]]:
        """
        Records an action and validates limit boundaries.
        Returns (is_allowed, error_message).
        """
        self.action_count += 1
        if self.action_count > self.max_actions:
            return (
                False,
                f"Action limit exceeded: Reached maximum allowed actions per task ({self.max_actions}). "
                "Stopping execution to prevent runaway loop.",
            )

        if self.last_action_signature == action_signature:
            self.consecutive_repeats += 1
            if self.consecutive_repeats > self.max_retries:
                return (
                    False,
                    f"Repeated action limit exceeded: Identical action '{action_signature}' "
                    f"attempted {self.consecutive_repeats} consecutive times without state progress. "
                    "Stopping repeated action loop.",
                )
        else:
            self.last_action_signature = action_signature
            self.consecutive_repeats = 1

        return True, None
