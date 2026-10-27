"""Command curricula shared by the GO1 student training environments."""

from __future__ import annotations

from collections.abc import Callable

import torch


class StopTransitionCurriculum:
    """Random per-episode stand/move modes with repeated move-to-stop transitions.

    Permanent-zero environments teach quiet standing.  Transition environments are
    deliberately different: they alternate between non-zero commands and zero so the
    student observes the closed-loop state reached *after* walking when asked to stop.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        fixed_zero_ratio: float,
        transition_ratio: float,
        move_duration_s: tuple[float, float],
        stop_duration_s: tuple[float, float],
    ) -> None:
        if not 0.0 <= fixed_zero_ratio <= 1.0:
            raise ValueError("fixed_zero_ratio must be in [0, 1]")
        if not 0.0 <= transition_ratio <= 1.0:
            raise ValueError("transition_ratio must be in [0, 1]")
        if fixed_zero_ratio + transition_ratio > 1.0:
            raise ValueError("fixed_zero_ratio + transition_ratio must not exceed 1")
        if step_dt <= 0.0:
            raise ValueError("step_dt must be positive")
        self._validate_duration("move_duration_s", move_duration_s)
        self._validate_duration("stop_duration_s", stop_duration_s)

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.fixed_zero_ratio = float(fixed_zero_ratio)
        self.transition_ratio = float(transition_ratio)
        self.move_duration_s = move_duration_s
        self.stop_duration_s = stop_duration_s

        self.fixed_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transition = torch.zeros_like(self.fixed_zero)
        self.stopped = torch.zeros_like(self.fixed_zero)
        self.just_stopped = torch.zeros_like(self.fixed_zero)
        self.timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    @staticmethod
    def _validate_duration(name: str, value: tuple[float, float]) -> None:
        if len(value) != 2 or value[0] <= 0.0 or value[1] < value[0]:
            raise ValueError(f"{name} must be a positive (min, max) pair")

    def _sample_steps(self, count: int, duration_s: tuple[float, float]) -> torch.Tensor:
        low = max(1, round(duration_s[0] / self.step_dt))
        high = max(low, round(duration_s[1] / self.step_dt))
        return torch.randint(low, high + 1, (count,), dtype=torch.int32, device=self.device)

    def reset(self, env_ids: torch.Tensor, commands: torch.Tensor) -> None:
        """Assign fresh modes to reset environments and start transition envs moving."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return

        fixed = torch.zeros(count, dtype=torch.bool, device=self.device)
        transition = torch.zeros_like(fixed)
        if count == self.num_envs:
            # Keep the requested global proportions exact while randomizing which
            # terrain/env indices receive them on every full reset.
            order = torch.randperm(count, device=self.device)
            num_fixed = round(count * self.fixed_zero_ratio)
            num_transition = round(count * self.transition_ratio)
            fixed[order[:num_fixed]] = True
            transition[order[num_fixed:num_fixed + num_transition]] = True
        else:
            draw = torch.rand(count, device=self.device)
            fixed = draw < self.fixed_zero_ratio
            transition = (draw >= self.fixed_zero_ratio) & (
                draw < self.fixed_zero_ratio + self.transition_ratio
            )

        self.fixed_zero[env_ids] = fixed
        self.transition[env_ids] = transition
        self.stopped[env_ids] = False
        self.just_stopped[env_ids] = False
        self.timer[env_ids] = 0
        transition_ids = env_ids[transition]
        if transition_ids.numel() > 0:
            self.timer[transition_ids] = self._sample_steps(
                transition_ids.numel(), self.move_duration_s
            )
        commands[env_ids[fixed], :3] = 0.0

    def step(
        self,
        commands: torch.Tensor,
        sample_commands: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Advance transition timers and update commands at phase boundaries."""
        self.just_stopped.zero_()
        active = self.transition
        self.timer[active] -= 1

        to_stop = active & ~self.stopped & (self.timer <= 0)
        stop_ids = torch.where(to_stop)[0]
        if stop_ids.numel() > 0:
            commands[stop_ids, :3] = 0.0
            self.stopped[stop_ids] = True
            self.just_stopped[stop_ids] = True
            self.timer[stop_ids] = self._sample_steps(stop_ids.numel(), self.stop_duration_s)

        to_move = active & self.stopped & (self.timer <= 0)
        move_ids = torch.where(to_move)[0]
        if move_ids.numel() > 0:
            commands[move_ids, :3] = sample_commands(move_ids)[:, :3]
            self.stopped[move_ids] = False
            self.timer[move_ids] = self._sample_steps(move_ids.numel(), self.move_duration_s)

        # This is deliberately last so permanent-zero commands can never be
        # accidentally changed by another sampler in the same control step.
        commands[self.fixed_zero, :3] = 0.0

    @property
    def normal(self) -> torch.Tensor:
        return ~(self.fixed_zero | self.transition)
