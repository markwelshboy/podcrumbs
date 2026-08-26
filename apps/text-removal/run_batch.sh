#!/usr/bin/env bash
set -euo pipefail
source "${VENV:-.venv}/bin/activate"
python run.py "$@"
