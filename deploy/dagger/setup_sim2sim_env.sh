#!/usr/bin/env bash
# One-shot creation of the sim2sim conda env used by deploy/dagger (GO1 DAgger depth
# sim2sim).  Usage:  bash deploy/dagger/setup_sim2sim_env.sh [env_name]
#
# Steps:
#   1. conda env create from environment_sim2sim.yml (python 3.10 + mujoco 3.11.0 +
#      opencv-python + pyyaml + numpy). No gym-quadruped: the GO1 sim2sim loads a copy
#      of mujoco_menagerie's go1.xml committed under deploy/mujoco_models/go1/.
#   2. install CPU-only torch from the official PyTorch cpu index (~200 MB instead of
#      the multi-GB CUDA bundle; plenty for sim2sim policy inference).
#   3. sanity-check the imports the play script needs.
set -euo pipefail

ENV_NAME="${1:-sim2sim_go1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO1_XML="$(cd "$HERE/.." && pwd)/mujoco_models/go1/go1.xml"

# Locate conda.
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="${CONDA_EXE%/bin/conda}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "ERROR: conda not found on PATH. Install conda/miniforge first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/3] creating conda env '$ENV_NAME' from environment_sim2sim.yml ..."
conda env create -n "$ENV_NAME" -f "$HERE/environment_sim2sim.yml"

echo "[2/3] installing CPU-only torch (enough for sim2sim inference) ..."
conda activate "$ENV_NAME"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[3/3] verifying imports ..."
python - <<'PY'
import os
import mujoco
import torch
import yaml
import cv2
import numpy as np

print(f"mujoco {mujoco.__version__} | torch {torch.__version__} | numpy {np.__version__}")
xml = os.environ["GO1_XML"]
assert os.path.exists(xml), f"GO1 model not found: {xml}"
print(f"go1.xml exists: {xml}")
import mujoco.viewer  # noqa: F401  (imports fine headless; only window creation needs a display)
print("mujoco.viewer import OK")
PY

echo
echo "DONE. Activate with:  conda activate $ENV_NAME"
echo "Run sim2sim:          python deploy/dagger/play_mujoco_dagger_go1.py --ckpt <ckpt> --scene stairs --viewer"
