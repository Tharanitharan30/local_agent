import json
import re
import sys
from typing import Dict, Any, Tuple, Optional, List

from rich.console import Console

import config
from model import QwenModel
from tools.base import BaseTool
from tools.terminal import TerminalTool
from tools.filesystem import (
    FilesystemTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    FileInfoTool,
    CreateFileTool,
    WriteFileTool,
    EditFileTool,
)
from tools.screen import ScreenTool
from tools.safety import ActionLimitTracker
from tools.mouse import (
    MouseTool,
    MouseMoveTool,
    MouseClickTool,
    MouseDoubleClickTool,
    MouseRightClickTool,
    MouseScrollTool,
)
from tools.keyboard import (
    KeyboardTool,
    KeyboardTypeTool,
    KeyboardPressTool,
    KeyboardHotkeyTool,
)
from tools.windows import (
    WindowListTool,
    WindowGetActiveTool,
    WindowFocusTool,
    WindowMinimizeTool,
    WindowMaximizeTool,
    WindowCloseTool,
)
from tools.applications import (
    ApplicationListTool,
    ApplicationLaunchTool,
)

console = Console()


def clean_response_text(text: str) -> str:
    """
    Remove internal model reasoning blocks (<think>...</think>) for clean user output.
    Preserves actual response content.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


class Agent:
    """
    Zia Agent Core: Manages user conversation, reasoning loop,
    tool schema dispatch, structured tool call detection, validation, execution,
    and feeding results back to Qwen for bounded iterative reasoning.
    """

    def __init__(
        self,
        model: Optional[QwenModel] = None,
        max_tool_iterations: int = config.MAX_TOOL_ITERATIONS,
    ):
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.tools: Dict[str, BaseTool] = {}
        self.system_prompt = self._load_system_prompt()
        self.messages: List[Dict[str, Any]] = []
        self.action_tracker = ActionLimitTracker()

    def _load_system_prompt(self) -> str:
        if config.SYSTEM_PROMPT_PATH.exists():
            return config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return "You are Zia, a local computer-use AI assistant running on Ubuntu Linux."

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance into the agent's tool registry, including aliases."""
        self.tools[tool.name] = tool
        # Register explicit aliases if declared on tool
        for alias in getattr(tool, "aliases", []):
            self.tools[alias] = tool
        # Register short/dotted cross-aliases
        if "." in tool.name:
            short_name = tool.name.split(".", 1)[1]
            self.tools[short_name] = tool
        else:
            self.tools[f"filesystem.{tool.name}"] = tool

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return unique function calling schemas for registered tools."""
        seen = set()
        schemas = []
        for tool in self.tools.values():
            if id(tool) not in seen:
                seen.add(id(tool))
                schemas.append(tool.to_tool_schema())
        return schemas

    def parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Parses structured model tool call in standard Qwen format:
        <tool_call>
        {
            "name": "terminal",
            "arguments": {"command": "..."}
        }
        </tool_call>
        Returns (tool_name, arguments) or (None, None).
        """
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            try:
                candidate = text.strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    data = json.loads(candidate)
                    if isinstance(data, dict) and "name" in data and "arguments" in data:
                        args = data.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        return data["name"], args
            except (json.JSONDecodeError, TypeError):
                pass
            return None, None

        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None, None
            tool_name = data.get("name")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            return tool_name, arguments if isinstance(arguments, dict) else {}
        except json.JSONDecodeError:
            return None, None

    def resolve_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Resolve a tool name or alias to registered BaseTool instance."""
        if tool_name in self.tools:
            return self.tools[tool_name]
        for prefix in ("filesystem.", "screen.", "mouse.", "keyboard.", "window.", "application."):
            if f"{prefix}{tool_name}" in self.tools:
                return self.tools[f"{prefix}{tool_name}"]
        if "." in tool_name:
            suffix = tool_name.split(".", 1)[1]
            if suffix in self.tools:
                return self.tools[suffix]
        return None

    def validate_tool_call(
        self, tool_name: Optional[str], arguments: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[BaseTool]]:
        """
        Validates whether the tool exists and required parameters are supplied.
        Returns (is_valid, error_message, resolved_tool).
        """
        if not tool_name:
            return False, "Missing tool name.", None

        tool = self.resolve_tool(tool_name)
        if not tool:
            available = sorted(list(set(self.tools.keys())))
            return False, f"Unknown tool '{tool_name}'. Available tools: {available}", None

        if arguments is None or not isinstance(arguments, dict):
            return False, f"Invalid arguments format for tool '{tool_name}'. Expected dictionary.", None

        required_params = tool.input_schema.get("required", [])
        missing = [p for p in required_params if p not in arguments]
        if missing:
            return False, f"Missing required argument(s) for tool '{tool_name}': {missing}", None

        return True, None, tool

    def _prune_messages(self, max_history_turns: int = 12) -> None:
        """
        Keep system prompt intact and retain the most recent conversational turns
        to prevent runaway context growth in long sessions.
        """
        if len(self.messages) > (max_history_turns * 2) + 1:
            system_msg = self.messages[0]
            recent_msgs = self.messages[-(max_history_turns * 2):]
            self.messages = [system_msg] + recent_msgs

    def run(self, user_input: str, reset_history: bool = False) -> str:
        """
        Processes a user request through the bounded agent loop up to max_tool_iterations.
        Maintains conversational context across turns.
        """
        if reset_history or not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        self.action_tracker.reset()
        self._prune_messages()
        self.messages.append({"role": "user", "content": user_input})
        tool_schemas = self.get_tool_schemas()

        iteration = 0
        while iteration < self.max_tool_iterations:
            iteration += 1

            console.print("[bold yellow][ZIA] Thinking...[/bold yellow]")
            sys.stdout.flush()

            if self.model is None:
                return "Error: No model loaded in Agent."

            # Generate response from Qwen with registered tool schemas
            raw_response = self.model.generate_response(
                self.messages,
                tools=tool_schemas if tool_schemas else None,
            )

            tool_name, arguments = self.parse_tool_call(raw_response)

            if not tool_name:
                # Model provided a direct natural language response
                self.messages.append({"role": "assistant", "content": raw_response})
                return clean_response_text(raw_response)

            if isinstance(arguments, dict) and "." in tool_name:
                prefix, suffix = tool_name.split(".", 1)
                if prefix in ("mouse", "keyboard") and "action" not in arguments:
                    arguments["action"] = suffix

            # Structured tool call requested by model
            console.print(f"[bold cyan][TOOL][/bold cyan] {tool_name}")
            if tool_name == "terminal" and isinstance(arguments, dict) and "command" in arguments:
                console.print(f"[bold white][CMD][/bold white]  {arguments['command']}")
            elif isinstance(arguments, dict):
                if tool_name in ("screen.capture", "screen", "capture_screen", "capture"):
                    disp = arguments.get("display", "primary")
                    qry = arguments.get("query", "")
                    console.print(f"[bold white][SCREEN][/bold white] display='{disp}' query='{qry}'")
                elif "mouse" in tool_name:
                    act = arguments.get("action", tool_name.replace("mouse.", ""))
                    x = arguments.get("x")
                    y = arguments.get("y")
                    btn = arguments.get("button", "left")
                    console.print(f"[bold white][MOUSE][/bold white] action='{act}' at=({x}, {y}) button='{btn}'")
                elif "keyboard" in tool_name:
                    act = arguments.get("action", tool_name.replace("keyboard.", ""))
                    txt = arguments.get("text", "")
                    k = arguments.get("key", "")
                    ks = arguments.get("keys", "")
                    console.print(f"[bold white][KEYBOARD][/bold white] action='{act}' text='{txt}' key='{k or ks}'")
                elif "window" in tool_name:
                    win_id = arguments.get("window_id", "")
                    app = arguments.get("application", "")
                    title = arguments.get("title", "")
                    console.print(f"[bold white][WINDOW][/bold white] id='{win_id}' app='{app}' title='{title}'")
                elif "application" in tool_name:
                    app = arguments.get("application", "")
                    console.print(f"[bold white][APP][/bold white] app='{app}'")
                if "path" in arguments:
                    console.print(f"[bold white][PATH][/bold white] {arguments['path']}")
                if "pattern" in arguments:
                    console.print(f"[bold white][PATTERN][/bold white] {arguments['pattern']}")
                if "old_text" in arguments and "new_text" in arguments:
                    console.print(f"[bold white][EDIT][/bold white] Replace '{arguments['old_text']}' -> '{arguments['new_text']}'")
            sys.stdout.flush()

            # Validate tool call
            is_valid, validation_err, tool = self.validate_tool_call(tool_name, arguments)
            if not is_valid or tool is None:
                result = {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Error: {validation_err}",
                    "exit_code": 1,
                }
            else:
                # Enforce action limits to prevent runaway loops
                action_sig = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                is_allowed, limit_err = self.action_tracker.record_and_validate(action_sig)
                if not is_allowed:
                    result = {
                        "success": False,
                        "blocked": True,
                        "reason": limit_err,
                        "error": limit_err,
                    }
                else:
                    try:
                        result = tool.execute(**arguments)
                    except Exception as e:
                        result = {
                            "success": False,
                            "stdout": "",
                            "stderr": f"Execution error in tool '{tool_name}': {str(e)}",
                            "exit_code": 1,
                        }

            # Log formatted tool result status
            status_color = "green" if result.get("success") else "red"
            exit_code = result.get("exit_code")
            blocked = result.get("blocked", False)
            if blocked:
                console.print(f"[bold red][BLOCKED][/bold red] {result.get('reason', 'Operation blocked by policy')}")
            elif exit_code is not None:
                err_msg = result.get("stderr") or result.get("error") or ""
                if exit_code != 0 and err_msg:
                    console.print(f"[bold {status_color}][RESULT][/bold {status_color}] exit_code={exit_code} ({err_msg})")
                else:
                    console.print(f"[bold {status_color}][RESULT][/bold {status_color}] exit_code={exit_code}")
            else:
                status_text = result.get("message") or ("OK" if result.get("success") else result.get("error", "Failed"))
                console.print(f"[bold {status_color}][RESULT][/bold {status_color}] {status_text}")
            sys.stdout.flush()

            # Append assistant tool call and tool execution result to message history
            self.messages.append({"role": "assistant", "content": raw_response})
            self.messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            })

        fallback = "Reached maximum tool iteration limit without completing the task."
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback


# Backward-compatible alias
AgentCore = Agent


def main():
    console.print("\n[bold magenta]╔════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║                   ZIA                      ║[/bold magenta]")
    console.print("[bold magenta]║      Local AI Computer-Use Agent           ║[/bold magenta]")
    console.print("[bold magenta]╚════════════════════════════════════════════╝[/bold magenta]\n")
    console.print(f"Configured Context:     {config.MAX_CONTEXT_LENGTH} tokens")
    console.print(f"Configured Max Tokens:  {config.MAX_NEW_TOKENS}\n")

    # Initialize Qwen Model and Agent Core
    model = QwenModel()
    agent = Agent(model=model)
    agent.register_tool(TerminalTool())
    agent.register_tool(ListDirectoryTool())
    agent.register_tool(ReadFileTool())
    agent.register_tool(SearchFilesTool())
    agent.register_tool(FileInfoTool())
    agent.register_tool(CreateFileTool())
    agent.register_tool(WriteFileTool())
    agent.register_tool(EditFileTool())
    agent.register_tool(FilesystemTool())
    agent.register_tool(ScreenTool())
    agent.register_tool(MouseTool())
    agent.register_tool(MouseMoveTool())
    agent.register_tool(MouseClickTool())
    agent.register_tool(MouseDoubleClickTool())
    agent.register_tool(MouseRightClickTool())
    agent.register_tool(MouseScrollTool())
    agent.register_tool(KeyboardTool())
    agent.register_tool(KeyboardTypeTool())
    agent.register_tool(KeyboardPressTool())
    agent.register_tool(KeyboardHotkeyTool())
    agent.register_tool(WindowListTool())
    agent.register_tool(WindowGetActiveTool())
    agent.register_tool(WindowFocusTool())
    agent.register_tool(WindowMinimizeTool())
    agent.register_tool(WindowMaximizeTool())
    agent.register_tool(WindowCloseTool())
    agent.register_tool(ApplicationListTool())
    agent.register_tool(ApplicationLaunchTool())

    console.print("\n[bold green]Zia is ready! Type 'exit' or 'quit' to stop.[/bold green]\n")

    try:
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    console.print("[bold yellow]Goodbye![/bold yellow]")
                    break

                response = agent.run(user_input)
                console.print(f"\n[bold green][ZIA][/bold green] {response}\n")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold yellow]Exiting...[/bold yellow]")
                break
    finally:
        from tools.input_backend import get_input_backend
        get_input_backend().close()
        from model.vision import get_vision_model
        get_vision_model().unload()
        model.unload()


if __name__ == "__main__":
    main()
