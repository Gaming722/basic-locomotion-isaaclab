"""Live student depth preview inside the Isaac Sim UI (no OpenCV HighGUI)."""

import numpy as np


def grayscale_to_rgba(gray: np.ndarray) -> np.ndarray:
    """Convert the student's HxW grayscale preview to ByteImageProvider RGBA bytes."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Depth preview must be a two-dimensional uint8 grayscale image.")
    height, width = gray.shape
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = 255
    return rgba


class IsaacDepthPreview:
    """Small persistent Isaac Sim window showing the latest delayed actor depth frame."""

    def __init__(self, width: int, height: int):
        import omni.ui as ui

        self._width = width
        self._height = height
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window("GO1 student depth", width=width * 4 + 20, height=height * 4 + 60, visible=True)
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label("Latest delayed actor frame | near=white, far=black")
                self._image = ui.ImageWithProvider(self._provider, width=width * 4, height=height * 4)

    def update(self, gray: np.ndarray) -> None:
        if gray.shape != (self._height, self._width):
            raise ValueError("Depth preview dimensions changed during playback.")
        rgba = grayscale_to_rgba(gray)
        self._provider.set_bytes_data(rgba.flatten().data, [self._width, self._height])

    def close(self) -> None:
        self._window.visible = False
        self._window.destroy()
