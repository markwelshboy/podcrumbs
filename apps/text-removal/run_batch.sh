#!/usr/bin/env bash
set -euo pipefail
source "${VENV:-.venv}/bin/activate"
python remove_text.py "$@"
