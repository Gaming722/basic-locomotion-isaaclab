## Overview

Distills the GO1 **heightmap-based** teacher (`Locomotion-Go1-Rough-Vision`) into a
**depth-conditioned student** that uses only a D435-like depth image + proprioception
(no `base_lin_vel`, since the real robot has no odometry). Two student envs are provided:

- `Locomotion-Go1-Rough-Vision-Tiled` — GPU-rendered TiledCamera depth (default).
- `Locomotion-Go1-Rough-Vision-RayCaster` — Warp `MultiMeshRayCasterCamera` depth
  (no renderer, cheaper, inherently never sees neighbouring envs).

The teacher reads `obs["teacher_obs"]` (sim base_lin_vel + heightmap, 323 dims); the
student reads `obs["common"]` (no velocity) + the depth image.

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
- `--dagger_policy_path <path>`: where to save the student checkpoint (default
  `<teacher_run>/dagger_policy.pt`). Give a distinct path per student variant so
  concurrent runs don't overwrite each other.
- `--video_output_dir <path>`: where to save dagger videos (default
  `<teacher_run>/videos/dagger`). Use a distinct dir per concurrent run so videos
  don't clobber each other.
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

- `--depth_blur_sigma` (default `0.5` px): Gaussian blur to simulate the residual
  smoothing of the real D435 848→106 (8× area-average) downscale / optics. The depth
  is now rendered at native 106×60 (= 848×480 / 8, exact aspect), so there is **no**
  in-sim 848→240 downscale to smooth any more. `0` disables.
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

**Resolution contract:** both student cameras now render depth at **native 106×60** —
exactly 1/8 of the D435 848×480 frame (aspect 1.7667), so at 87° HFOV the vertical FOV
(~56.5°) and the per-pixel angular mapping match the real frame 1:1. The on-robot
preprocessing must therefore downscale the real depth to 106×60 (e.g. an 8×
area-average / `cv2.INTER_AREA`), not 240×140. Dagger checkpoints trained at the old
240×140 are incompatible with a 106×60-trained student — retrain after this change.

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

### Raycast student (`Locomotion-Go1-Rough-Vision-RayCaster`)

Alternative student that uses a Warp `MultiMeshRayCasterCamera` for depth instead of
the rendered TiledCamera. Same D435 intrinsics (87° HFOV, **106×60** = D435 848×480
downsampled by exactly 8×, so the aspect — and therefore the vertical FOV, ~56.5° —
matches the real frame) and same d435 mount
pose as the Tiled student, so the two students see the same view. The depth ray-casts
`/World/ground` + the robot's own links, so it sees its own legs (self-occlusion) but
**never neighbouring envs** — no `enforce_env_spacing` needed, and the standard
curriculum terrain (`GO1_ROUGH_TERRAINS_CFG`, curriculum=True) trains 4096 envs
without the Tiled env's 46×46 cap.

```bash
python scripts/dagger/train_dagger_go1.py \
  --task=Locomotion-Go1-Rough-Vision-RayCaster \
  --checkpoint=logs/rsl_rl/rough_direct/<run>/model_<N>.pt \
  --dagger_policy_path=logs/rsl_rl/rough_direct/<run>/dagger_policy_raycaster.pt \
  --num_envs=1024 --headless \
  --video --dual_pane --dual_pane_student_depth \
  --video_length=300 --video_interval=1000
```

Differences vs the Tiled student:

- **No renderer** — the ray-cast depth runs headless without the rendering kit
  (`--enable_cameras` is still forced by the script but is a no-op for ray-casting).
- **Faster startup** — the robot-link warp meshes are shared across envs
  (`RaycastTargetCfg(is_shared=True, ...)`), so 1024 envs init in seconds instead of
  the cold-start minutes the per-env mesh parsing caused.
- **Neighbour isolation is inherent** — each env's rays only ever hit that env's
  ground + own links (`_mesh_ids_wp` is per-env; the shared geometry is only shared
  at the vertex/face level, transforms stay per-env), so there is no
  `enforce_env_spacing` and no 46×46 env cap.
- **Self-occlusion is modelled** — the robot's legs are ray-cast at their *current*
  pose (`track_mesh_transforms=True`); plain-string mesh targets would freeze them at
  the initial pose, so the links must use `RaycastTargetCfg`.
- `--terrain` / `--difficulty` behave the same; `--num_envs` up to 4096.
- **Watch step throughput** — 106×60 rays (≈6.4k/env) × `num_envs` (+ link meshes) is the
  dominant cost; ~5× cheaper than the old 240×140. If the step rate is still too low,
  reduce `--num_envs`.

### Watch an already-trained student (`--eval`)

Plays a saved `dagger_policy.pt` **without the DAgger machinery**: no teacher is loaded, no
replay buffer is filled, no gradient updates run, and **no checkpoint is written** (neither
periodically nor at the end). The step counter starts at `0` — it is never restored from the
checkpoint (unlike `--resume_from`, which restores the training `step` and therefore makes a
small `--max_training_steps` exit immediately). Depth sanitize / delay / history, camera follow
and dual-pane recording are identical to training, so the video shows the true student
input/output. Add `--cmd "vx vy wz"` to hold a constant velocity command (e.g. forward on
stairs) and `--real-time` to pace interactive viewing to real time.

```bash
python scripts/dagger/train_dagger_go1.py \
  --task=Locomotion-Go1-Rough-Vision-RayCaster \
  --eval --eval_policy=logs/rsl_rl/rough_direct/<run>/dagger_policy_raycaster_106x60.pt \
  --terrain stairs --difficulty 0.667 \
  --cmd "0.5 0 0" --num_envs=1024 --headless \
  --video --dual_pane --dual_pane_student_depth \
  --video_length=600 --video_interval=1000000000 \
  --video_output_dir=logs/rsl_rl/rough_direct/<run>/videos/eval_stairs_15cm \
  --max_training_steps=620
```

- `--eval` requires `--eval_policy`; the teacher `--checkpoint` / `--load_run` is **not** needed.
- `--max_training_steps` is the pure eval budget (counted from 0) and is **mandatory with
  `--headless`** (otherwise the evaluation would run forever on a server — guarded at startup).
  Without `--headless` and without a budget it runs until the window is closed.
- Videos default to `<student_ckpt_dir>/videos/dagger_eval/dual_step-<N>.mp4` (separate from
  training's `videos/dagger` so the two never overwrite `dual_step-0.mp4`).
- `--follow_env` is auto-clamped onto a moving env: with `--num_envs > 500` and no `--cmd`,
  envs 0-499 stand still, so it is forced to `>= 500`; with `--cmd` (or `--num_envs <= 500`)
  every env moves and it may point anywhere.
- A checkpoint trained at a different obs/action/depth shape is rejected with a clear error;
  a metadata mismatch (`depth_image_size`, `depth_history_length`, `depth_delay_frames`) that
  still loads warns about an input-distribution shift.
