#!/bin/bash
# run.sh
# Launches the PhishNet AI terminal application.
# Activates the virtual environment first if one exists.

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "No virtual environment found. Run ./setup.sh first (or continuing with system python3)."
fi

python3 main.py
