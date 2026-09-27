from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Abstract base class for all Zia tools.
    Every tool must define a name, description, input schema,
    and an execute method returning a structured dictionary.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool with given keyword arguments.

        Returns:
            Dict[str, Any]: Structured result dictionary.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Returns tool specification for prompting or inspection."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
