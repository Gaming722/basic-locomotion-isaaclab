## Overview

Distills the GO1 **heightmap-based** teacher (`Locomotion-Go1-Rough-Vision`) into a
**depth-conditioned student** that uses only a D435-like depth image + proprioception
(no `base_lin_vel`, since the real robot has no odometry). Two student envs are provided:

- `Locomotion-Go1-Rough-Vision-Tiled` — GPU-rendered TiledCamera depth (default).
- `Locomotion-Go1-Rough-Vision-RayCaster` — Warp `MultiMeshRayCasterCamera` depth
  (no renderer, cheaper, inherently never sees neighbouring envs).

The teacher reads `obs["teacher_obs"]` (sim base_lin_vel + heightmap, 303 dims); the
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
- **When `--num_envs` > 500**, envs 0-499 are command-zero (stand still); only
  envs 500+ move. With <=500 envs, this fixed-standing group is disabled. `--follow_env` defaults to 600 to follow a moving robot.
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
- DAgger defaults are tuned for the 1024-env GO1 RayCaster workload:
  `--dagger_buffer_size 8192`, `--dagger_samples_per_step 32`,
  `--dagger_batch_size 128`, `--dagger_train_micro_batch_size 64`, and
  `--dagger_inference_batch_size 128`. With the default `--dagger_train_every 4`
  and `--dagger_updates_per_train 1`, the replay ratio is 1.0 and the full buffer
  spans 256 control steps. Teacher action mixing decays over 20000 steps by default.

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
  control steps, modelling capture→inference latency. Additional buffer slots preserve the full history.

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
1/8 of an 848×480 frame (aspect 1.7667), with a nominal 87° HFOV and
~56.5° VFOV. This is an ideal pinhole approximation, not a measured calibration. The on-robot
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


### Corrected depth encoding and history (introduced in version 2)

Training, evaluation and MuJoCo use `depth_pipeline.py`: metric optical-axis
Z depth, invalid/nonpositive/at-or-beyond-3 m pixels -> 2 m, clip [0.1, 2],
blur -> measurement noise -> clip -> dropout. Original invalid pixels retain
2 m after filtering. `--depth_min_z` (metres, default 0/off) masks measured
near-invalid pixels before filtering and again after noise.

A length H sequence with delay D uses a ring buffer of H+D frames. It feeds
[t-D-H+1, ..., t-D], never the newer D frames. Reset fills all H+D slots with
the new episode frame. Old students with nonzero delay were trained with a
wrapped, nonchronological sequence; retraining is preferred. The teacher PPO
checkpoint is unaffected. New checkpoints record `depth_pipeline_version`,
`depth_min_z`, actual camera intrinsics, base-relative mount and frame periods.

### Real-camera calibration and measured noise

Before claiming sim-to-real alignment, collect:

- Actual depth stream resolution and fx/fy/cx/cy, distortion/rectification status,
  depth units, and whether depth is aligned to the colour camera.
- Base-to-depth-optical-centre position and quaternion, with coordinate convention.
  Bottom-screw mount (0.26, 0, 0.12), ROS optical wxyz quaternion
  (-0.353553, 0.612372, -0.612372, 0.353553), uses the official nominal screw-to-depth-origin offset; the installation is not calibrated.
- Crop origin/size, resize method and final 106x60 intrinsics. Transform the
  calibrated intrinsics with the same crop/resize; matching nominal FOV alone
  does not establish alignment. Native low-resolution rendering plus blur only
  approximates high-resolution area downsampling at occlusion edges.
- Frame timestamps/rate and capture-to-policy latency. Student consumption is
  50 Hz; `depth_delay_frames` now counts control steps, not captured camera frames.
- Depth error and invalid-pixel fraction on static planes at several distances;
  measure proprioceptive bias/noise separately. Do not add noise to teacher labels.

Until these measurements are available, additive noise and dropout remain off;
blur sigma 0.5 is a nominal approximation. Use measured values with
`--depth_additive_noise_std`, `--depth_dropout_prob`, `--depth_min_z` and
`--depth_delay_frames`. Constant Gaussian noise and independent dropout are
simplified models and do not represent distance-dependent or spatially correlated
stereo errors. MuJoCo reads these depth settings from student metadata.

Run CPU regression checks without Isaac Sim:

```bash
python -m unittest discover -s scripts/dagger/tests -v
```


### D435 nominal 30 Hz sensor with original DAgger history (version 4)

`--depth_fps 30` sets the IsaacLab camera `update_period=1/30` directly.
There is no custom capture clock and no override to update_period=0.
Like the original `train_dagger.py`, every 50 Hz control step reads the camera
buffer, preprocesses it, and appends a depth snapshot to history. Cached raw
frames may repeat; configured random noise is applied to each read.

Five history entries span four control intervals, approximately 80 ms, not
133 ms. `--depth_delay_frames` counts control-step history entries: 1 is
20 ms, and the default is 1 control step (20 ms at 50 Hz).
The H+D buffer and chronological sequence fix remain in place.

IsaacLab owns sensor refresh. With lazy reads at 20 ms intervals, a 33.3 ms
sensor period can yield a new image every 40 ms (approximately 25 Hz), rather
than strict hardware 30 FPS. The nominal period does not guarantee an independent
30 FPS capture thread. This approximation is retained to follow the original
DAgger handling. MuJoCo reads every control step and caches raw depth according
to the saved sensor period, matching the same lazy-refresh convention.

Version-3 students used captured-frame history/delay with a custom capture clock;
version-4 uses control-step history/delay. Evaluation/resume warns about changed
pipeline versions; retraining students is preferred. The teacher is unaffected.

The hardware source is planned as 848x480; simulator rendering remains native
106x60 for efficiency. Source resolution is recorded as planned metadata, not
proof of calibrated projection or high-resolution downsampling equivalence.
No crop or colour alignment is assumed. Read real intrinsics/depth scale from
the active depth stream once hardware is available, then measure the mount pose
and noise before changing the nominal geometry or enabling random noise.

Example with the existing teacher:

```bash
python scripts/dagger/train_dagger_go1.py \
  --task Locomotion-Go1-Rough-Vision-Tiled \
  --checkpoint logs/rsl_rl/rough_direct/2026-09-09_01-41-12/model_96950.pt \
  --num_envs 1024 --headless \
  --depth_fps 30 --depth_history_length 5 --depth_delay_frames 1
```


### D435 bottom-screw versus depth optical origin

The user mount position `(0.26, 0, 0.12)` is the **bottom screw**, with
body axes X forward/Y left/Z up and Ry(+30 deg) mounting pitch. It is not
the depth optical origin. The official nominal model gives screw-to-depth
translation `(0.0106, 0.0175, 0.0125)` m: forward offset is
`0.0149 - 0.0001 - 0.0042`, including front-glass/zero-depth-reference corrections.
The depth frame coincides nominally with the left IR frame, not RGB or case centre.

Compose `p_base_depth = p_base_screw + R_base_screw * p_screw_depth`.
This gives `(0.27542987, 0.0175, 0.12552532)` m. The existing ROS optical
quaternion already contains the 30-degree mounting pitch and body-to-optical
axis rotation, so its orientation is retained. Do not rotate the translation
using the ROS optical quaternion: the local offset above uses screw body axes.

Both student camera configurations share `assets/d435_geometry.py`.
MuJoCo uses the optical pose saved in student metadata; older checkpoints
without pose metadata retain their previous screw-origin pose. New students
should be trained with the corrected geometry. The teacher heightmap checkpoint
is unaffected. This is a nominal manufacturer-model correction, not a substitute
for measuring the actual robot installation or per-device calibration.

Source: [RealSense official D435 model](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro).


### Rotate video subjects across terrains

Add `--rotate_video_env` with `--video --dual_pane`. Training and evaluation
start with `--follow_env` (default 600), then cycle the available terrain groups
at each clip boundary. Within a clip both panes follow the same selected env.
Files include its ID (`dual_step-10000_env-700.mp4`); the log reports the
subject, terrain group and starting terrain level. Environment resets may still
change its location/difficulty during a clip, as in fixed-subject recording.

For curriculum terrains, groups use the generator's normalized proportion and
column assignment. For random non-curriculum terrain generation, exact type is
not available from the importer; rotation uses columns and logs `column-N`.
Tiled row-major spacing is handled separately. No terrain or command is changed.
With >500 envs and random commands, envs 0-499 are excluded because they stand
still; only terrain groups occupied by moving envs can be covered. A single
terrain configuration can vary subject/difficulty but cannot introduce other types.
Without the flag, video subject stays fixed. Rotation requires dual-pane video.
