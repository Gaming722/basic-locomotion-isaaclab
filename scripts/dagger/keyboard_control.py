"""Keyboard velocity commands for interactive GO1 DAgger evaluation."""

import math


class KeyboardVelocityCommand:
    """Hold one bounded command for every simulated environment."""

    def __init__(self, amplitudes, step=0.1):
        if len(amplitudes) != 3 or any(not math.isfinite(v) or v < 0 for v in amplitudes):
            raise ValueError("command_a must contain three finite nonnegative amplitudes")
        self.limits = tuple(float(v) for v in amplitudes)
        self.step = step
        self.values = [0.0, 0.0, 0.0]

    def on_key(self, name):
        """Return True if the key changed or explicitly reset the command."""
        controls = {
            "UP": (0, 1), "DOWN": (0, -1),
            "LEFT": (1, 1), "A": (1, 1),
            "RIGHT": (1, -1), "D": (1, -1),
            "Q": (2, 1), "E": (2, -1),
        }
        if name in controls:
            axis, direction = controls[name]
            self.values[axis] = max(-self.limits[axis], min(
                self.limits[axis], round(self.values[axis] + direction * self.step, 6)))
        elif name == "V":
            self.values[:] = [0.0, 0.0, 0.0]
        elif name == "C":
            self.values[:] = [min(0.4, self.limits[0]), 0.0, 0.0]
        else:
            return False
        return True

    def apply(self, command_buffer, env_ids=None):
        value = command_buffer.new_tensor(self.values)
        if env_ids is None:
            command_buffer[:, :3] = value
        else:
            command_buffer[env_ids, :3] = value


def install_isaac_keyboard_commands(env, amplitudes):
    """Override GO1 command resampling and register viewport key presses."""
    if not hasattr(env.unwrapped, "_commands"):
        raise RuntimeError("Keyboard play requires an environment with a _commands buffer.")
    from omni.appwindow import get_default_app_window
    import carb
    import basic_locomotion_isaaclab.tasks.custom_events as custom_events

    app_window = get_default_app_window()
    if app_window is None or app_window.get_keyboard() is None:
        raise RuntimeError("No Isaac Sim keyboard found. Launch without --headless and focus the viewport.")
    controller = KeyboardVelocityCommand(amplitudes)

    def _set_keyboard_commands(env_obj, env_ids=None):
        controller.apply(env_obj._commands, env_ids)

    custom_events._get_new_random_commands = _set_keyboard_commands
    controller.apply(env.unwrapped._commands)
    input_iface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()

    def _on_key(event, *_):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if controller.on_key(event.input.name):
            print(f"[CMD] vx={controller.values[0]:+.2f} "
                  f"vy={controller.values[1]:+.2f} wz={controller.values[2]:+.2f}", flush=True)
        return True

    subscription = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
    print(f"[INFO] Keyboard commands enabled, limits={controller.limits}: "
          "Up/Down=vx, Left/Right or A/D=vy, Q/E=wz, V=stop, C=0.4 cruise. "
          "Click the Isaac Sim window for focus.")
    # Keep both the callback and subscription alive for the play loop.
    return input_iface, keyboard, subscription, controller
