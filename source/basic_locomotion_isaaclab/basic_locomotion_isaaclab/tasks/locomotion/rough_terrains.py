"""Shared rough-terrain configurations for the locomotion environments.

Terrain is an environment property, not a robot property. Keeping the configs
here (single source of truth) lets multiple robots reference the same terrain
without cross-robot imports, while preserving the original GO1 curriculum
terrain as a switchable alternative.

* ``COMMON_ROUGH_TERRAINS_CFG``: fixed-difficulty mix (boxes + star + stairs +
  slopes), shared by GO1 and Aliengo so their reward curves are comparable.
* ``GO1_ROUGH_TERRAINS_CFG``: the original GO1 curriculum terrain (discrete
  obstacles ramp with difficulty, more stairs). Point a robot class's
  ``ROUGH_TERRAINS_CFG`` at this one to switch back. Stairs go up to 0.20 m
  (raised 2026-08-15 from 0.13 m to push the teacher onto harder terrain).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# Fixed-difficulty terrain shared by GO1 and Aliengo (aligned 2026-08-14).
COMMON_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.10), platform_width=2.0,
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.1, num_bars=10, bar_width_range=(0.15, 0.20), bar_height_range=(0.05, 0.15), platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)

# Original GO1 curriculum terrain: discrete obstacles whose height ramps with
# difficulty (0.10 -> 0.25) and more stairs. Kept for switching back.
GO1_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2
        ),
        "discrete_obstacles_terrain": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.2,
            abs_height_noise=(-0.05, 0.05),
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=40, height=0.10, size=(0.6, 0.6), max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=40, height=0.25, size=(1.2, 1.2), max_yx_angle=0.0, degrees=True
            ),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.20), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.20), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)
