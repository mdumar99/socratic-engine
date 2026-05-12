#!/usr/bin/env bash
# bootstrap.sh — sets up the Python venv for this project only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/python/venv"

echo "→ Creating Python venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "→ Installing requirements"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_ROOT/python/requirements.txt"

echo "✓ Done. Activate with: source python/venv/bin/activate"
