## Installation Train

1. Install Isaac Lab by following the [installation guide](https://github.com/isaac-sim/IsaacLab). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

2. Install git for very large file
```bash
sudo apt install git-lfs
```

3. Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)


4. Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/basic_locomotion_isaaclab
```

5. If you want to play with [Morphologycal Symmetries](https://arxiv.org/pdf/2403.17320), install the repo [morphosymm-rl](https://github.com/iit-DLSLab/morphosymm-rl)

6. If you want to play with [Adversarial Motion Priors](https://arxiv.org/pdf/2104.02180), install the repo [amp-rsl-rl](https://github.com/ami-iit/amp-rsl-rl) from the [AMI](https://github.com/ami-iit) research lab.

## Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Flat --num_envs=4096 --headless
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=4096 --headless
```

- GO1 teacher (heightmap-based, used as the DAgger expert):

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Go1-Rough-Vision --num_envs=4096 --headless
```

  To continue / finetune from a checkpoint, add `--resume=True --load_run=<run> --checkpoint=model_<N>.pt`
  (details in `scripts/dagger/README_dagger.md`).

- GO1 DAgger student env (TiledCamera depth, for the train_dagger_go1.py pipeline):
  see `scripts/dagger/README_dagger.md`.

- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=Locomotion-Aliengo-Flat --num_envs=16
python scripts/rsl_rl/play.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=16
```

- GO1 teacher play on a **controlled terrain** (e.g. verify stair-climbing on 12 cm stairs with a
  fixed forward command):

```bash
python scripts/rsl_rl/play.py \
  --task=Locomotion-Go1-Rough-Vision \
  --checkpoint=tested_policies/go1/rough_vision/model_70800.pt \
  --terrain stairs --difficulty 0.467 \
  --cmd "0.5 0 0" --num_envs 8
```

  Play-time options (`scripts/rsl_rl/play.py`):
  - `--terrain <rough|stairs|slope|flat>`: single-type terrain (applied only if the env cfg has
    `rebuild_terrain()`, i.e. the GO1 teacher env; other tasks ignore it).
  - `--difficulty <0.0-1.0>`: fixed terrain difficulty. For stairs, `step_height = 0.05 + difficulty*0.15`,
    so 10 cm -> 0.333, 12 cm -> 0.467, 15 cm -> 0.667, 20 cm -> 1.0.
  - `--cmd "vx vy wz"`: fixed velocity command for all envs (replaces the random command generator,
    e.g. `--cmd "0.5 0 0"` for constant forward).
  - `--num_envs <N>`: fewer envs for local visualization (default 4096).
  - Viewport keys during play: `E`/`Q` = next / previous env, `1`-`9` = jump to env N-1.



## Use AMP, Morphological Symmetries, DAGGER or Depth to Heightmap
Each of these modules has a specific README in its own script folder.


## Run Hyperparameter Search

```bash
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | python3 (TERMINAL 1)
```

```bash
python3 ../basic_locomotion_isaaclab/exts/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/tuner.py --run_mode local --cfg_file ../basic_locomotion_isaaclab/exts/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/locomotion_aliengo_cfg.py --cfg_class LocomotionAliengoFlatTuner (TERMINAL 2)
```