# Zia - Local AI Computer-Use Agent

Zia is a fully local, terminal-based AI computer-use agent for Linux powered by `Qwen/Qwen3-4B` and PyTorch (CUDA). Zia interacts with your system through modular, controlled tools without relying on cloud APIs or external GUI frameworks.

---

## 🏗️ Architecture

```
User
  ↓
Terminal Interface (agent.py)
  ↓
Agent Core (agent.py)
  ↓
Qwen3-4B Model Wrapper (model/qwen.py)
  ↓
Tool Selection & Validation (tools/base.py)
  ↓
Tool Execution (tools/terminal.py)
  ↓
Structured Tool Result (JSON)
  ↓
Qwen3-4B Reasoning Continuation
  ↓
Final Natural Language Response to User
```

---

## 📁 Directory Structure

```text
zia/
├── agent.py               # Main CLI interface & Agent Core reasoning loop
├── config.py              # Centralized configuration (model, timeouts, parameters)
├── model/
│   ├── __init__.py
│   └── qwen.py            # Local Qwen/Qwen3-4B PyTorch CUDA wrapper
├── tools/
│   ├── __init__.py
│   ├── base.py            # Generic abstract tool interface (BaseTool)
│   └── terminal.py        # Subprocess terminal tool with safety & structured output
├── prompts/
│   └── system.txt         # Agent system prompt & tool instructions
├── tests/
│   ├── __init__.py
│   ├── test_terminal_tool.py  # Unit tests for terminal execution & safety
│   └── test_agent.py          # Unit tests for Agent Core reasoning loop & parsing
├── activate.sh            # Virtual environment activation script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Requirements & Prerequisites
* Ubuntu Linux
* NVIDIA GPU (GeForce RTX 3050 or higher recommended)
* Python 3.12+
* CUDA-enabled PyTorch environment

### 2. Activate Environment
Run:
```bash
source activate.sh
```

Or manually activate the virtual environment:
```bash
source .venv/bin/activate
```

---

## 💻 Running Zia

Start the interactive terminal CLI:
```bash
python agent.py
```

### Example Usage:
```text
[USER] What files are in the current directory?
[ZIA] Thinking...
[TOOL] terminal
[TOOL RESULT] exit_code=0
[ZIA]
The current directory contains the following files:
- agent.py
- config.py
- README.md
...
```

---

## 🛡️ Tool Safety & Capabilities

### Terminal Tool Capabilities
- Executes Linux shell commands using standard Python `subprocess`.
- Returns structured JSON data (`success`, `stdout`, `stderr`, `exit_code`).
- Enforces configurable execution timeouts (default: 30 seconds).

### Command Safety Layer
Zia includes a basic regex-based safety filter (`TerminalTool._is_safe`) blocking obvious destructive operations such as:
- `rm -rf /` or `rm -rf ~` or `rm -rf /*`
- `rm --no-preserve-root`
- Filesystem formatting commands (`mkfs`)
- Direct block device overwrites (`dd if=... of=/dev/...`, `> /dev/sda`)
- Fork bombs (`:(){ :|:& };:`)
- System shutdown/reboot commands (`shutdown`, `reboot`, `init 0`)

> [!WARNING]
> **Basic Safety Layer Notice:** The safety layer is a basic defense-in-depth filter against accidental destructive execution and does NOT constitute a completely isolated OS sandbox.

---

## 🧪 Testing

Run all unit tests using `pytest`:

```bash
pytest
```

---

## 🎯 Current Milestone & Future Roadmap

- **Milestone 1 (Current)**: Qwen3-4B + Agent Core + Terminal Tool (Subprocess execution, safety checks, structured JSON outputs, bounded loop).
- **Future Milestone 2**: File System & File Content Operations (Read, Edit, Search).
- **Future Milestone 3**: GUI Automation (Screen vision, keyboard/mouse control).
- **Future Milestone 4**: Browser Automation & Standalone Desktop Sidecar.
