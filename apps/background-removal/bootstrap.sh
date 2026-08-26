#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-.venv}"

"$PYTHON_BIN" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
# Install BEN2 code without allowing its dependency metadata to replace the pod's CUDA torch stack.
python -m pip install --no-deps git+https://github.com/PramaLLC/BEN2.git

python - <<'PY'
import sys
print('Python:', sys.version)
try:
    import kornia
    print('Kornia:', kornia.__version__)
except Exception as e:
    print('WARNING: kornia import failed:', e)
try:
    import torch
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('WARNING: torch import failed:', e)
PY

echo
echo "Environment ready. Activate with: source $VENV/bin/activate"
