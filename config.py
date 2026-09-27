import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.txt"

# Model Configuration
MODEL_NAME = os.getenv("ZIA_MODEL_NAME", "Qwen/Qwen3-4B")
DEVICE_MAP = os.getenv("ZIA_DEVICE_MAP", "cuda:0")
TORCH_DTYPE = os.getenv("ZIA_TORCH_DTYPE", "float16")

# Quantization Configuration (4-bit BitsAndBytes)
LOAD_IN_4BIT = os.getenv("ZIA_LOAD_IN_4BIT", "true").lower() in ("true", "1", "yes")
BNB_4BIT_QUANT_TYPE = os.getenv("ZIA_BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_USE_DOUBLE_QUANT = os.getenv("ZIA_BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() in ("true", "1", "yes")
BNB_4BIT_COMPUTE_DTYPE = os.getenv("ZIA_BNB_4BIT_COMPUTE_DTYPE", "float16")

# Context & Generation Configuration
MAX_CONTEXT_LENGTH = int(os.getenv("ZIA_MAX_CONTEXT_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("ZIA_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("ZIA_TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("ZIA_TOP_P", "0.8"))
ENABLE_THINKING = os.getenv("ZIA_ENABLE_THINKING", "false").lower() in ("true", "1", "yes")

# Agent Execution Configuration
MAX_TOOL_ITERATIONS = int(os.getenv("ZIA_MAX_TOOL_ITERATIONS", "5"))

# Terminal Tool Configuration
TERMINAL_TIMEOUT = int(os.getenv("ZIA_TERMINAL_TIMEOUT", "10"))
