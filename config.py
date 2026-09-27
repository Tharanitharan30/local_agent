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
MAX_CONTEXT_LENGTH = int(os.getenv("ZIA_MAX_CONTEXT_LENGTH", "4096"))
MAX_NEW_TOKENS = int(os.getenv("ZIA_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("ZIA_TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("ZIA_TOP_P", "0.8"))
ENABLE_THINKING = os.getenv("ZIA_ENABLE_THINKING", "false").lower() in ("true", "1", "yes")

# Agent Execution Configuration
MAX_TOOL_ITERATIONS = int(os.getenv("ZIA_MAX_TOOL_ITERATIONS", "5"))

# Terminal Tool Configuration
TERMINAL_TIMEOUT = int(os.getenv("ZIA_TERMINAL_TIMEOUT", "10"))

# Filesystem Tool Configuration
WORKSPACE_ROOT = Path(os.getenv("ZIA_WORKSPACE_ROOT", str(BASE_DIR))).resolve()
MAX_READ_FILE_SIZE = int(os.getenv("ZIA_MAX_READ_FILE_SIZE", str(64 * 1024)))    # 64 KB limit
MAX_WRITE_FILE_SIZE = int(os.getenv("ZIA_MAX_WRITE_FILE_SIZE", str(64 * 1024)))  # 64 KB limit
MAX_SEARCH_RESULTS = int(os.getenv("ZIA_MAX_SEARCH_RESULTS", "50"))
MAX_SEARCH_DEPTH = int(os.getenv("ZIA_MAX_SEARCH_DEPTH", "5"))

# Backup Configuration for File Modifications
ENABLE_BACKUPS = os.getenv("ZIA_ENABLE_BACKUPS", "true").lower() in ("true", "1", "yes")
BACKUP_DIR_NAME = os.getenv("ZIA_BACKUP_DIR_NAME", ".zia_backups")
MAX_BACKUPS_PER_FILE = int(os.getenv("ZIA_MAX_BACKUPS_PER_FILE", "5"))

# Screen & Vision Configuration
SCREEN_TIMEOUT = int(os.getenv("ZIA_SCREEN_TIMEOUT", "10"))
SCREEN_MAX_WIDTH = int(os.getenv("ZIA_SCREEN_MAX_WIDTH", "1280"))
SCREEN_MAX_HEIGHT = int(os.getenv("ZIA_SCREEN_MAX_HEIGHT", "720"))
VISION_MODEL_NAME = os.getenv("ZIA_VISION_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
VISION_LOAD_IN_4BIT = os.getenv("ZIA_VISION_LOAD_IN_4BIT", "true").lower() in ("true", "1", "yes")
VISION_MAX_NEW_TOKENS = int(os.getenv("ZIA_VISION_MAX_NEW_TOKENS", "300"))
VISION_TEMPERATURE = float(os.getenv("ZIA_VISION_TEMPERATURE", "0.2"))
