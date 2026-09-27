import re
import subprocess
from typing import Any, Dict
import config
from tools.base import BaseTool


class TerminalTool(BaseTool):
    """
    Terminal tool for executing controlled shell commands on Linux.
    Captures stdout, stderr, exit_code, enforces timeouts, and applies safety checks.
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

    # Dangerous command regex patterns for basic safety layer
    DANGEROUS_PATTERNS = [
        r"rm\s+-[a-zA-Z]*rf?\s+(/|\/\*|~|~/\*|\$HOME|/home/[^/]+/\*)",
        r"rm\s+.*--no-preserve-root",
        r"\bmkfs\b",
        r"\bdd\s+if=.*of=/dev/",
        r">\s*/dev/sd[a-z]",
        r"chmod\s+-[a-zA-Z]*R\s+777\s+/",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",  # Fork bomb
        r"\b(shutdown|reboot|init\s+0|poweroff)\b",
    ]

    def __init__(self, timeout: int = config.TERMINAL_TIMEOUT):
        self.timeout = timeout
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]

    def _is_safe(self, command: str) -> tuple[bool, str]:
        """
        Basic safety layer check.
        NOTE: This is NOT a complete security sandbox. It is a basic defense-in-depth filter
        to prevent accidental filesystem destruction or dangerous operations.
        """
        cmd = command.strip()
        for pattern in self._compiled_patterns:
            if pattern.search(cmd):
                return False, f"Command matched safety pattern: '{pattern.pattern}'"
        return True, ""

    def execute(self, command: str = "", **kwargs: Any) -> Dict[str, Any]:
        """
        Execute a shell command safely with timeout.

        Returns structured dict:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "exit_code": int
        }
        """
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
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
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
