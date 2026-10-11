#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE="${1:-flat}"
STAIR_HEIGHT_CM="${2:-12}"
CKPT="${GO1_STUDENT_CKPT:-$ROOT/logs/rsl_rl/rough_direct/2026-09-09_01-41-12/dagger_raycaster_v3_ft2.pt}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
exec conda run --no-capture-output -n sim2sim python \
  "$ROOT/deploy/dagger/play_mujoco_dagger_go1.py" \
  --ckpt "$CKPT" --scene "$SCENE" --stair_height_cm "$STAIR_HEIGHT_CM" \
  --fsm --viewer --policy_steps -1 "$@"
