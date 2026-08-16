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
