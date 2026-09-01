# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from copy import deepcopy
from . import amp_cfg
from . import morphosymm_cfg


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "flat_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP Related Stuff
    dataset = deepcopy(amp_cfg.dataset)
    discriminator = deepcopy(amp_cfg.discriminator)

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg


@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "rough_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,     # = mujoco_playground brax_ppo_config entropy_cost (stability)
        num_learning_epochs=4, # = brax num_updates_per_batch
        num_mini_batches=32,   # = brax num_minibatches (smaller minibatches, smoother updates)
        learning_rate=3.0e-4,  # = brax learning_rate
        schedule="adaptive",   # KL-adaptive LR (rsl_rl's closest equivalent to brax linear decay)
        gamma=0.97,            # = brax discounting
        lam=0.95,              # = brax gae_lambda
        desired_kl=0.01,       # = brax desired_kl (adaptive-KL target)
        max_grad_norm=1.0,
    )

    # AMP Related Stuff
    dataset = deepcopy(amp_cfg.dataset)
    discriminator = deepcopy(amp_cfg.discriminator)

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg
