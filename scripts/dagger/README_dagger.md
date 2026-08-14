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

### Useful flags

- `--resume_from <path>`: resume the student from a saved `dagger_policy.pt`.
- `--video_interval 500`: record videos more often (default 1000 steps).
- `--follow_env 500`: follow a specific env index (must be >= 500).
