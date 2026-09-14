"""MuJoCo depth renderer for the GO1 DAgger student (sim2sim).

The depth camera is baked into the scene XML (orthogonal frame, 30 deg down,
VFOV ~56.5 deg at 106x60 = D435 848x480 / 8, using the optical pose saved
in the student checkpoint), so this class is
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

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, camera_id: int, *, width: int = 106, height: int = 60, update_period: float = 0.0):
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self._w, self._h = width, height
        self._renderer = None
        self.update_period = update_period
        self._last_update_time = -float("inf")
        self._cached_depth = None

    def render(self) -> np.ndarray:
        """Return raw metric depth (H, W) in metres (no preprocessing)."""
        # Lazy sensor cache, matching IsaacLab reads at control-step boundaries.
        if self._cached_depth is not None and 0 <= self.data.time - self._last_update_time < self.update_period - 1e-6:
            return self._cached_depth
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self._h, width=self._w)
        self._renderer.update_scene(self.data, camera=self.camera_id)
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        self._cached_depth = np.array(depth, dtype=np.float32, copy=True)
        self._last_update_time = self.data.time
        return self._cached_depth
