import json
import re
import sys
from typing import Dict, Any, Tuple, Optional, List

from rich.console import Console

import config
from model import QwenModel
from tools import BaseTool, TerminalTool

console = Console()


class AgentCore:
    """
    Zia Agent Core: Responsible for managing the reasoning loop,
    tool registration, tool invocation parsing, executing tools,
    and feeding results back to Qwen for iterative reasoning.
    """

    def __init__(self, model: Optional[QwenModel] = None):
        self.model = model
        self.tools: Dict[str, BaseTool] = {}
        self.system_prompt = self._load_system_prompt()
        self.messages: List[Dict[str, str]] = []

    def _load_system_prompt(self) -> str:
        if config.SYSTEM_PROMPT_PATH.exists():
            return config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        return "You are Zia, a local AI terminal computer agent."

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool into Zia's tool registry."""
        self.tools[tool.name] = tool

    def parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Parses model response for a tool call tag:
        <tool_call>
        {
            "name": "tool_name",
            "arguments": {...}
        }
        </tool_call>

        Returns (tool_name, arguments) or (None, None) if no valid tool call found.
        """
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            # Fallback parsing: direct JSON block
            try:
                data = json.loads(text.strip())
                if isinstance(data, dict) and "name" in data and "arguments" in data:
                    return data["name"], data["arguments"]
            except (json.JSONDecodeError, TypeError):
                pass
            return None, None

        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            tool_name = data.get("name")
            arguments = data.get("arguments", {})
            return tool_name, arguments
        except json.JSONDecodeError:
            return None, None

    def run(self, user_input: str) -> str:
        """
        Processes user request through bounded Agent reasoning loop up to MAX_TOOL_ITERATIONS.
        """
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.messages.append({"role": "user", "content": user_input})

        iteration = 0
        while iteration < config.MAX_TOOL_ITERATIONS:
            iteration += 1

            console.print("[bold yellow][ZIA] Thinking...[/bold yellow]")
            sys.stdout.flush()
            response = self.model.generate_response(self.messages)

            tool_name, arguments = self.parse_tool_call(response)

            if not tool_name:
                # Final response from model
                self.messages.append({"role": "assistant", "content": response})
                return response

            # Tool call requested by model
            console.print(f"[bold cyan][TOOL][/bold cyan] {tool_name}")
            sys.stdout.flush()
            
            tool = self.tools.get(tool_name)
            if not tool:
                result = {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Error: Tool '{tool_name}' is not registered.",
                    "exit_code": 1
                }
            else:
                try:
                    result = tool.execute(**arguments)
                except Exception as e:
                    result = {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Error executing tool '{tool_name}': {str(e)}",
                        "exit_code": 1
                    }

            # Log formatted tool result status
            status_color = "green" if result.get("success") else "red"
            exit_code = result.get("exit_code", -1)
            console.print(f"[bold {status_color}][TOOL RESULT][/bold {status_color}] exit_code={exit_code}")
            sys.stdout.flush()

            # Append assistant tool call and tool execution result to message history
            self.messages.append({"role": "assistant", "content": response})
            result_str = json.dumps(result, indent=2)
            self.messages.append({
                "role": "user",
                "content": f"<tool_result>\n{result_str}\n</tool_result>"
            })

        fallback = "Reached maximum tool iteration limit without completing the task."
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback


def main():
    console.print("\n[bold magenta]========================================[/bold magenta]")
    console.print("[bold magenta]           ZIA LOCAL AI AGENT           [/bold magenta]")
    console.print("[bold magenta]========================================[/bold magenta]\n")

    # Initialize Qwen Model and Agent Core
    model = QwenModel()
    agent = AgentCore(model=model)
    agent.register_tool(TerminalTool())

    console.print("\n[bold green]Zia is ready! Type 'exit' or 'quit' to stop.[/bold green]\n")

    while True:
        try:
            user_input = console.input("[bold blue][USER][/bold blue] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break

            response = agent.run(user_input)
            console.print(f"\n[bold green][ZIA][/bold green]\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Exiting...[/bold yellow]")
            break


if __name__ == "__main__":
    main()
