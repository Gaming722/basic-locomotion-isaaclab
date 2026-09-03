"""MuJoCo depth renderer for the GO1 DAgger student (sim2sim).

The depth camera is baked into the scene XML (orthogonal frame, 30 deg down,
VFOV ~56.5 deg at 106x60 = D435 848x480 / 8, mounted at (0.26, 0, 0.12) in the
trunk frame — matching the IsaacLab Tiled/RayCaster students), so this class is
a thin wrapper around ``mujoco.Renderer``. Raw metric depth is returned; the full
training preprocessing (clip [0.1, 2.0] + blur + history/delay) is applied in
``play_mujoco_dagger_go1.py`` using the checkpoint metadata.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
import mujoco


class MujocoDepthCamera:
    """Render metric depth from a mujoco model/data at the baked-in depth camera."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, camera_id: int, *, width: int = 106, height: int = 60):
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self._w, self._h = width, height
        self._renderer = None

    def render(self) -> np.ndarray:
        """Return raw metric depth (H, W) in metres (no preprocessing)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self._h, width=self._w)
        self._renderer.update_scene(self.data, camera=self.camera_id)
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return np.asarray(depth, dtype=np.float32)
