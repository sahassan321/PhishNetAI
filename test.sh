#!/bin/bash
# test.sh
# Runs the PhishNet AI unit test suite.

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

python3 -m unittest test_phishing_detector.py -v
