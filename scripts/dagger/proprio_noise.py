"""Stateful, term-wise proprioceptive noise for the Go1 DAgger student.

The environment owns the clean observation history because the privileged teacher
must remain an oracle.  This module builds a separate noisy history for the student,
so a historical sample keeps the noise it received when it was first observed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


FRAME_SIZE = 45
ANGULAR_VELOCITY = slice(0, 3)
PROJECTED_GRAVITY = slice(3, 6)
JOINT_POSITION = slice(9, 21)
JOINT_VELOCITY = slice(21, 33)


@dataclass(frozen=True)
class ProprioNoiseCfg:
    """Per-step zero-mean Gaussian white-noise standard deviations."""

    angular_velocity_std: float = 0.02
    projected_gravity_std: float = 0.002
    joint_position_std: float = 0.001
    joint_velocity_std: float = 0.02

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class ProprioceptiveNoise:
    """Maintain a temporally consistent noisy observation history per environment."""

    def __init__(self, num_envs: int, history_length: int, device, cfg: ProprioNoiseCfg):
        cfg.validate()
        if num_envs <= 0 or history_length <= 0:
            raise ValueError("num_envs and history_length must be positive")
        self.num_envs = num_envs
        self.history_length = history_length
        self.device = torch.device(device)
        self.cfg = cfg
        self.history = torch.empty(num_envs, history_length, FRAME_SIZE, device=self.device)
        self.initialized = False

    def _corrupt(self, frames: torch.Tensor) -> torch.Tensor:
        """Corrupt ``[N,...,45]`` frames while preserving command/action fields exactly."""
        if frames.shape[-1] != FRAME_SIZE:
            raise ValueError(f"Expected final observation dimension {FRAME_SIZE}, got {frames.shape}")
        # Isaac uses all-zero history frames as reset padding. They are not sensor
        # measurements and must stay zero instead of becoming random pseudo-observations.
        valid_frame = frames[..., PROJECTED_GRAVITY].norm(dim=-1, keepdim=True) > 0.5
        noisy = frames.clone()
        terms = (
            (ANGULAR_VELOCITY, self.cfg.angular_velocity_std),
            (PROJECTED_GRAVITY, self.cfg.projected_gravity_std),
            (JOINT_POSITION, self.cfg.joint_position_std),
            (JOINT_VELOCITY, self.cfg.joint_velocity_std),
        )
        for indices, std in terms:
            if std > 0:
                noisy[..., indices] += torch.randn_like(noisy[..., indices]) * std

        # Projected gravity is a direction vector; component noise must not alter its norm.
        gravity = noisy[..., PROJECTED_GRAVITY]
        noisy[..., PROJECTED_GRAVITY] = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.where(valid_frame, noisy, frames)

    def update(self, clean_common: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Push the newest clean frame and return the student's flattened noisy history."""
        expected = self.history_length * FRAME_SIZE
        if clean_common.shape != (self.num_envs, expected):
            raise ValueError(f"Expected clean common shape {(self.num_envs, expected)}, got {clean_common.shape}")
        clean = clean_common.reshape(self.num_envs, self.history_length, FRAME_SIZE)

        if not self.initialized:
            self.history.copy_(self._corrupt(clean))
            self.initialized = True
        else:
            self.history[:, :-1].copy_(self.history[:, 1:].clone())
            self.history[:, -1].copy_(self._corrupt(clean[:, -1]))

        if reset_mask is not None and reset_mask.any():
            env_ids = reset_mask.to(self.device).nonzero(as_tuple=False).flatten()
            reset_clean = clean.index_select(0, env_ids)
            self.history.index_copy_(0, env_ids, self._corrupt(reset_clean))

        return self.history.reshape(self.num_envs, expected)
