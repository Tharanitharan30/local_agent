"""
Integration test for Zia using real local Qwen/Qwen3-4B model on CUDA PyTorch.
"""
import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import AgentCore
from model import QwenModel
from tools import TerminalTool


@pytest.mark.integration
def test_full_agent_integration():
    print("\n--- Initializing Real Model ---")
    model = QwenModel()
    agent = AgentCore(model=model)
    agent.register_tool(TerminalTool())

    user_query = "What is the present working directory? Execute a command to find out."
    print(f"\nUser Query: {user_query}")

    response = agent.run(user_query)
    print(f"\nFinal Response from Zia:\n{response}")

    assert response is not None
    assert len(response) > 0


if __name__ == "__main__":
    test_full_agent_integration()
