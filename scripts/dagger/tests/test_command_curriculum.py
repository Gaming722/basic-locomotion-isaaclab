"""Hardware-free tests for the student move-to-stop command curriculum."""

import pathlib
import sys
import unittest

import torch


TASKS_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks"
)
sys.path.insert(0, str(TASKS_DIR))
from command_curriculum import StopTransitionCurriculum


class StopTransitionCurriculumTests(unittest.TestCase):
    def make_curriculum(self, num_envs=100):
        return StopTransitionCurriculum(
            num_envs=num_envs,
            device="cpu",
            step_dt=0.02,
            fixed_zero_ratio=0.12,
            transition_ratio=0.25,
            move_duration_s=(2.0, 5.0),
            stop_duration_s=(2.0, 4.0),
        )

    def test_full_reset_has_exact_randomized_cohort_sizes(self):
        torch.manual_seed(7)
        curriculum = self.make_curriculum()
        commands = torch.ones((100, 3))
        curriculum.reset(torch.arange(100), commands)
        self.assertEqual(curriculum.fixed_zero.sum().item(), 12)
        self.assertEqual(curriculum.transition.sum().item(), 25)
        self.assertEqual(curriculum.normal.sum().item(), 63)
        torch.testing.assert_close(
            commands[curriculum.fixed_zero], torch.zeros((12, 3))
        )
        self.assertTrue((curriculum.timer[curriculum.transition] >= 100).all())
        self.assertTrue((curriculum.timer[curriculum.transition] <= 250).all())

    def test_transition_moves_to_zero_then_back_to_sampled_command(self):
        torch.manual_seed(3)
        curriculum = self.make_curriculum(8)
        commands = torch.full((8, 3), 0.4)
        curriculum.reset(torch.arange(8), commands)
        transition_ids = torch.where(curriculum.transition)[0]
        self.assertGreater(transition_ids.numel(), 0)

        curriculum.timer[transition_ids] = 1
        curriculum.step(commands, lambda ids: torch.full((ids.numel(), 3), 0.7))
        self.assertTrue(curriculum.stopped[transition_ids].all())
        self.assertTrue(curriculum.just_stopped[transition_ids].all())
        torch.testing.assert_close(commands[transition_ids], torch.zeros_like(commands[transition_ids]))

        curriculum.timer[transition_ids] = 1
        curriculum.step(commands, lambda ids: torch.full((ids.numel(), 3), 0.7))
        self.assertFalse(curriculum.stopped[transition_ids].any())
        torch.testing.assert_close(
            commands[transition_ids], torch.full_like(commands[transition_ids], 0.7)
        )

    def test_permanent_zero_is_reasserted_every_step(self):
        curriculum = self.make_curriculum()
        commands = torch.ones((100, 3))
        curriculum.reset(torch.arange(100), commands)
        commands[curriculum.fixed_zero] = 9.0
        curriculum.step(commands, lambda ids: torch.ones((ids.numel(), 3)))
        torch.testing.assert_close(
            commands[curriculum.fixed_zero],
            torch.zeros((curriculum.fixed_zero.sum().item(), 3)),
        )


if __name__ == "__main__":
    unittest.main()
