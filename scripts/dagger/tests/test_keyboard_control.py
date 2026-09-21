"""Hardware-free tests for GO1 interactive play commands."""

import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from keyboard_control import KeyboardVelocityCommand, install_isaac_keyboard_commands
from play_dagger_go1 import eval_command


class KeyboardCommandTests(unittest.TestCase):
    def test_bounds_stop_and_cruise(self):
        controller = KeyboardVelocityCommand((0.8, 0.4, 0.8))
        for positive, negative, axis, limit in (
            ("UP", "DOWN", 0, 0.8),
            ("LEFT", "RIGHT", 1, 0.4),
            ("Q", "E", 2, 0.8),
        ):
            for _ in range(30):
                self.assertTrue(controller.on_key(positive))
            self.assertAlmostEqual(controller.values[axis], limit)
            for _ in range(30):
                controller.on_key(negative)
            self.assertAlmostEqual(controller.values[axis], -limit)
        self.assertFalse(controller.on_key("ESCAPE"))
        controller.on_key("C")
        self.assertEqual(controller.values, [0.4, 0., 0.])
        controller.on_key("V")
        self.assertEqual(controller.values, [0., 0., 0.])

    def test_apply_to_all_and_reset_subset(self):
        controller = KeyboardVelocityCommand((.8, .4, .8))
        controller.on_key("UP")
        commands = torch.ones((4, 3))
        controller.apply(commands)
        torch.testing.assert_close(commands, torch.tensor([[.1, 0., 0.]] * 4))
        commands[:] = -1
        controller.apply(commands, torch.tensor([1, 3]))
        torch.testing.assert_close(commands[1], torch.tensor([.1, 0., 0.]))
        torch.testing.assert_close(commands[0], torch.tensor([-1., -1., -1.]))

    def test_isaac_callback_propagates_and_updates_teacher_commands(self):
        callbacks = []
        keyboard = object()
        input_iface = types.SimpleNamespace(
            subscribe_to_keyboard_events=lambda _keyboard, callback: callbacks.append(callback)
        )
        app_window = types.SimpleNamespace(get_keyboard=lambda: keyboard)
        appwindow = types.ModuleType("omni.appwindow")
        appwindow.get_default_app_window = lambda: app_window
        omni = types.ModuleType("omni")
        omni.appwindow = appwindow
        carb = types.ModuleType("carb")
        carb.input = types.SimpleNamespace(
            acquire_input_interface=lambda: input_iface,
            KeyboardEventType=types.SimpleNamespace(KEY_PRESS=1, KEY_RELEASE=2),
        )
        events = types.ModuleType("basic_locomotion_isaaclab.tasks.custom_events")
        tasks = types.ModuleType("basic_locomotion_isaaclab.tasks")
        tasks.custom_events = events
        package = types.ModuleType("basic_locomotion_isaaclab")
        package.tasks = tasks
        env_obj = types.SimpleNamespace(_commands=torch.zeros((2, 3)))
        env = types.SimpleNamespace(unwrapped=env_obj)
        modules = {
            "omni": omni, "omni.appwindow": appwindow, "carb": carb,
            "basic_locomotion_isaaclab": package,
            "basic_locomotion_isaaclab.tasks": tasks,
            "basic_locomotion_isaaclab.tasks.custom_events": events,
        }
        with mock.patch.dict(sys.modules, modules):
            install_isaac_keyboard_commands(env, (0.8, 0.4, 0.8))
            self.assertEqual(len(callbacks), 1)
            event = lambda name, kind: types.SimpleNamespace(type=kind, input=types.SimpleNamespace(name=name))
            self.assertIs(callbacks[0](event("UP", 1)), True)
            self.assertIs(callbacks[0](event("UP", 2)), True)
            self.assertIs(callbacks[0](event("UNKNOWN", 1)), True)
            events._get_new_random_commands(env_obj)
            torch.testing.assert_close(env_obj._commands, torch.tensor([[0.1, 0., 0.]] * 2))

    def test_play_entry_point(self):
        command = eval_command(["--policy", "student.pt", "--difficulty=0.6"])
        self.assertIn("--eval", command)
        self.assertIn("--keyboard", command)
        self.assertIn("--stair_direction=up", command)
        self.assertIn("--eval_policy=student.pt", command)
        self.assertEqual(command[-1], "--difficulty=0.6")
        flat = eval_command(["--policy", "student.pt", "--terrain", "flat"])
        self.assertIn("--terrain=flat", flat)
        self.assertFalse(any(arg.startswith("--stair_direction") for arg in flat))


if __name__ == "__main__":
    unittest.main()
