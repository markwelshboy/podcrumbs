#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"$PYTHON_BIN" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if ! python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
then
    echo "Installing CUDA PyTorch from: $TORCH_INDEX_URL"
    python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
    echo "Existing CUDA PyTorch in the venv is usable; keeping it."
fi

python -m pip install -r requirements.txt
python -m pip install --upgrade 'git+https://github.com/huggingface/diffusers.git'

echo
echo "Environment ready."
echo "Activate: source $VENV/bin/activate"
echo "Help:     python remove_text.py --help"
echo
echo "NOTE: FLUX.2 Klein 9B is gated on Hugging Face and uses the FLUX non-commercial license."
echo "      Accept its model terms first, then set HF_TOKEN or run: hf auth login"
