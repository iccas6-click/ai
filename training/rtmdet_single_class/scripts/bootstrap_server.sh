#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install the NVIDIA driver before continuing." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
else
  echo "Reusing existing virtual environment: $PROJECT_ROOT/.venv"
fi
uv pip install --python .venv/bin/python \
  "torch==2.1.0+cu118" "torchvision==0.16.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python "mmcv==2.1.0" \
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
uv pip install --python .venv/bin/python -r requirements/training.txt

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA on this server.")
print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
PY

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
.venv/bin/python -m training.rtmdet_single_class.scripts.download_datasets synthetic-v3
.venv/bin/python -m training.rtmdet_single_class.scripts.prepare_single_class \
  datasets/raw/healtheat-pill-synthetic-v3/extracted

echo "Server setup complete. Run training/rtmdet_single_class/scripts/run_training.sh."
