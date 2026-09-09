#!/bin/bash
# setup.sh
# Creates a clean virtual environment and installs dependencies
# for the PhishNet AI project.

set -e   # exit immediately if any command fails

VENV_DIR="venv"

echo "=== PhishNet AI: Setup ==="

# 1. create the virtual environment (only if it doesn't already exist)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists, skipping creation."
fi

# 2. activate it
source "$VENV_DIR/bin/activate"

# 3. install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "To activate the environment yourself later, run:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "Now run ./run.sh to launch PhishNet AI."
