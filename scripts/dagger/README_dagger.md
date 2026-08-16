## Overview

Distills the GO1 **heightmap-based** teacher (`Locomotion-Go1-Rough-Vision`) into a
**depth-conditioned student** that uses only the D435 (TiledCamera depth) + proprioception
(no `base_lin_vel`, since the real robot has no odometry). The student env is
`Locomotion-Go1-Rough-Vision-Tiled`.

The teacher reads `obs["teacher_obs"]` (sim base_lin_vel + heightmap, 323 dims); the
student reads `obs["common"]` (no velocity) + the TiledCamera depth image.

## 1. Train the teacher

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Go1-Rough-Vision --num_envs=4096 --headless
```

The teacher's latest checkpoint is used as the DAgger expert (e.g. run `11-47-38`,
`model_7999.pt`).

### Finetune / resume the teacher

Continue training from an existing checkpoint. A **new** timestamped log dir is
created, so the original run is not overwritten.

```bash
python scripts/rsl_rl/train.py \
  --task=Locomotion-Go1-Rough-Vision \
  --num_envs=4096 --headless \
  --resume=True \
  --load_run=2026-08-14_11-47-38 --checkpoint=model_7999.pt
```

Notes:
- `--resume` is `type=bool`, so write `--resume=True` (not a bare `--resume`).
- `--checkpoint=last` auto-picks the newest checkpoint in `--load_run`.
- Changed reward/terrain configs apply when resuming — only the weights are loaded.

## 2. Run DAgger (student distillation)

```bash
python scripts/dagger/train_dagger_go1.py \
  --task=Locomotion-Go1-Rough-Vision-Tiled \
  --num_envs=1024 \
  --checkpoint=logs/rsl_rl/rough_direct/<run>/model_<N>.pt \
  --headless \
  --video --dual_pane --video_length=300 --video_interval=1000 \
  --max_training_steps=3000
```

### Key requirements / gotchas

- **`--headless` is required** on a no-display server. The script uses
  `AppLauncher(args_cli)`, so `headless` comes from this CLI flag — without it the
  rendering kit tries to open a window and hangs.
- **`--num_envs` must be > 500**: envs 0-499 are command-zero (stand still); only
  envs 500+ actually move. `--follow_env` defaults to 600 to follow a moving robot.
  The Tiled terrain is 46x46 sub-terrains (max 2116 envs).
- The TiledCamera student env requires the rendering kit; the script forces
  `--enable_cameras`.

### Outputs

- Dual-pane videos (left = Isaac Sim view, right = live grayscale depth):
  `logs/rsl_rl/rough_direct/<run>/videos/dagger/dual_step-<N>.mp4`
- Student policy checkpoint: `logs/rsl_rl/rough_direct/<run>/dagger_policy.pt`

`<run>` is the directory of the loaded teacher checkpoint (see `--checkpoint`
below). If no `--checkpoint`/`--load_run` is given, the script auto-selects the
**latest** run and checkpoint in `logs/rsl_rl/rough_direct/`.

### Useful flags

- `--resume_from <path>`: resume the student from a saved `dagger_policy.pt`.
- `--video_interval 500`: record videos more often (default 1000 steps).
- `--follow_env 500`: follow a specific env index (must be >= 500).
- `--terrain <rough|stairs|slope|flat>`: pick the Tiled env terrain. `stairs` and
  `slope` select a single terrain type (handy for recording, e.g. stair climbing).
- `--difficulty <0.0-1.0>`: pin every sub-terrain to exactly this difficulty
  (default = random per sub-terrain). Combine with `--terrain stairs --difficulty 1.0`
  to record **all-max-height stairs** (0.20 m).
- `--dual_pane_student_depth`: the right dual-pane shows the **student actor's
  sanitized depth** (nan_to_num + clip [0.1, 2.0] + depth_min_z mask) instead of the
  raw camera depth — i.e. exactly what feeds the student GRU.

### Depth sensor simulation (noise + latency)

By default the student depth is passed through a small D435-like sensor model (in
`_sanitize_depth_data`, applied on GPU before the frame enters the history buffer):

- `--depth_blur_sigma` (default `1.0` px): Gaussian blur to simulate optics / the
  848→240 downscale smoothing. `0` disables.
- `--depth_additive_noise_std` (default `0.0` m): additive Gaussian depth noise.
- `--depth_dropout_prob` (default `0.0`): probability of dropping a pixel to the
  far-saturation code `2.0` (simulates D435 holes / invalid depth).
- `--depth_delay_frames` (default `1`): delays the depth fed to the student by N
  env steps, modelling capture→inference latency. Must be `< --depth_history_length`.

Processing order: blur (replicate padding, no border artifact) → additive noise →
re-clip to `[0.1, 2.0]` → dropout holes → `depth_min_z` mask. Keep these knobs in
sync with the on-robot depth preprocessing. To reproduce the old clean-depth
behavior, run with `--depth_blur_sigma 0 --depth_delay_frames 0`.

### Depth encoding & sim-to-real contract

The camera config is **unchanged** (`clipping_range=(0.01, 3.0)`,
`depth_clipping_behavior="max"`). The depth fed to the student follows the
"everything unmeasurable → far" convention (same as the InstinctLab reference):

| Pixel | Raw camera | After pipeline |
|---|---|---|
| valid in `[0.1, 2.0]` m | real depth | real depth |
| no-hit / far (> 3.0 m) | 3.0 | `2.0` (far-saturation) |
| holes (`--depth_dropout_prob`) | — | `2.0` |
| sub-`depth_min_z` (if enabled) | real depth | `2.0` |

**Deploy-side contract (mandatory):** a real D435 emits `0` (16-bit raw) for any
pixel it cannot measure (too close / too far / specular / hole). Feed the real
depth through the **exact same** pipeline as `_sanitize_depth_data`, and map
invalid pixels to far before/after the clip so they stay in-distribution:
`real_depth[real_depth == 0] = 2.0` (or `3.0` before the `[0.1, 2.0]` clip). If you
skip this, real holes read as `0.1` (near) while training only ever showed `2.0`
(far) — an out-of-distribution input.

> **Provenance:** blur, latency and the "everything unmeasurable → far" encoding
> follow the InstinctLab reference pipeline (`NoisyGroupedRayCasterCamera` +
> `crop → blur → depth_normalization`). `--depth_dropout_prob` and `depth_min_z`
> are **optional extensions (default off)** to make the student robust to the real
> D435's holes and near-field min-z; they are *not* part of InstinctLab's pipeline.
> Keep the camera config (`clipping_range`, `depth_clipping_behavior`) unchanged.

### Record all-max stairs + student depth

```bash
python scripts/dagger/train_dagger_go1.py \
  --task=Locomotion-Go1-Rough-Vision-Tiled \
  --checkpoint=logs/rsl_rl/rough_direct/<run>/model_<N>.pt \
  --terrain stairs --difficulty 1.0 \
  --num_envs=1024 --headless --video --dual_pane \
  --dual_pane_student_depth --video_length=1200
```

Videos land in `logs/rsl_rl/rough_direct/<run>/videos/dagger/dual_step-<N>.mp4`
(the directory of the teacher loaded via `--checkpoint`).
