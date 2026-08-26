#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-$ROOT/.venv}"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p vendor models
if [[ ! -d vendor/FBCNN/.git ]]; then
  git clone --depth 1 https://github.com/jiaxi-jiang/FBCNN.git vendor/FBCNN
else
  git -C vendor/FBCNN pull --ff-only || true
fi

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo
echo "Environment ready."
echo "Activate with: source $VENV/bin/activate"
echo "Run smoke test: python smoke_test.py"
