#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "❌ Zia virtual environment not found."
    echo "Create it first with:"
    echo "  python -m venv .venv"
    exit 1
fi

source "$PROJECT_DIR/.venv/bin/activate"

echo ""
echo "╔══════════════════════════════════╗"
echo "║          ZIA ENVIRONMENT         ║"
echo "╚══════════════════════════════════╝"
echo ""
echo "📁 Project : $PROJECT_DIR"
echo "🐍 Python  : $(python --version)"
echo "⚡ PyTorch : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo ""
echo "Zia environment activated 🚀"
echo ""
