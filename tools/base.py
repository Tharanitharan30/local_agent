from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Abstract base class for all Zia tools.
    Every tool defines a name, description, input schema (JSON schema),
    and an execute method returning a structured dictionary.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool with given arguments.

        Returns:
            Dict[str, Any]: Structured result dictionary.
        """
        pass

    def to_tool_schema(self) -> Dict[str, Any]:
        """Returns standard function definition schema for model tool calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Legacy helper returning tool specification."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# Alias for flexibility
Tool = BaseTool
