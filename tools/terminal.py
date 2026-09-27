import re
import subprocess
from typing import Any, Dict, Tuple
import config
from tools.base import BaseTool

MAX_OUTPUT_CHARS = 4000


class TerminalTool(BaseTool):
    """
    Terminal tool for executing controlled shell commands on Linux.
    Captures stdout, stderr, exit_code, enforces timeouts, limits output size,
    and applies safety checks.
    """

    name = "terminal"
    description = (
        "Executes a Linux shell command in the terminal and returns structured "
        "stdout, stderr, and exit code. Use this for file operations, system queries, etc."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The Linux shell command to execute."
            }
        },
        "required": ["command"]
    }

    # Dangerous command regex patterns for basic safety layer.
    # NOTE: This is NOT a complete security sandbox, but a defense-in-depth barrier
    # to prevent obviously catastrophic commands.
    DANGEROUS_PATTERNS = [
        # Recursive delete of root, home, or wildcards
        r"rm\s+-[a-zA-Z]*rf?\s+(/|\/\*|~|~/\*|\$HOME|/home/[^/]+/\*)",
        r"rm\s+.*--no-preserve-root",
        # Disk formatting / partitioning
        r"\bmkfs(\.[a-zA-Z0-9]+)?\b",
        r"\b(fdisk|sfdisk|cfdisk|parted|wipefs)\b",
        # Direct raw disk writing
        r"\bdd\s+.*of=/dev/(sd|nvme|hd|vd|mapper)",
        r">\s*/dev/(sd[a-z]|nvme[0-9]|hd[a-z]|vd[a-z])",
        # Recursive permissive permissions on root
        r"chmod\s+-[a-zA-Z]*R\s+777\s+/",
        # Fork bombs
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
        # System shutdown/reboot
        r"\b(shutdown|reboot|init\s+0|poweroff)\b",
    ]

    def __init__(self, timeout: int = config.TERMINAL_TIMEOUT, max_output_chars: int = MAX_OUTPUT_CHARS):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]

    def _is_safe(self, command: str) -> Tuple[bool, str]:
        """
        Basic safety layer check against dangerous commands.
        Returns (is_safe, reason).
        """
        cmd = command.strip()
        for pattern in self._compiled_patterns:
            if pattern.search(cmd):
                return False, f"Potentially destructive command matching pattern: '{pattern.pattern}'"
        return True, ""

    def _truncate_output(self, text: str) -> str:
        """Truncate large output to prevent token blowout."""
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + f"\n... [Output truncated to {self.max_output_chars} characters]"
        return text

    def execute(self, command: str = "", **kwargs: Any) -> Dict[str, Any]:
        """
        Execute a shell command safely with timeout.

        Returns structured dict:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "blocked": bool (optional),
            "reason": str (optional)
        }
        """
        # Support kwargs if command wasn't passed directly as positional/named
        if not command and "cmd" in kwargs:
            command = kwargs["cmd"]

        if not command or not command.strip():
            return {
                "success": False,
                "stdout": "",
                "stderr": "Error: Empty command provided.",
                "exit_code": 1
            }

        is_safe, reason = self._is_safe(command)
        if not is_safe:
            return {
                "success": False,
                "blocked": True,
                "reason": reason,
                "stdout": "",
                "stderr": f"Command blocked by safety layer: {reason}",
                "exit_code": 1
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            stdout = self._truncate_output(result.stdout)
            stderr = self._truncate_output(result.stderr)
            return {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command execution timed out after {self.timeout} seconds.",
                "exit_code": 124
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "exit_code": 1
            }
