import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_pipeline import depth_sequence_from_history, preprocess_depth, student_depth_grayscale
from depth_preview import IsaacDepthPreview, grayscale_to_rgba


class DepthPipelineTest(unittest.TestCase):
    def test_student_depth_preview_palette(self):
        depth = torch.tensor([[0., .1, 2., 3.]], dtype=torch.float16)
        gray = student_depth_grayscale(depth)
        self.assertEqual(gray.shape, (1, 4))
        self.assertEqual(gray.tolist(), [[255, 255, 0, 0]])
        rgba = grayscale_to_rgba(gray)
        self.assertEqual(rgba.shape, (1, 4, 4))
        self.assertEqual(rgba[0, 0].tolist(), [255, 255, 255, 255])
        self.assertEqual(rgba[0, -1].tolist(), [0, 0, 0, 255])

    def test_isaac_depth_preview_uses_byte_provider(self):
        class Context:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class Provider:
            def set_bytes_data(self, data, size):
                self.data = bytes(data)
                self.size = size

        class Window:
            def __init__(self, *_args, **_kwargs):
                self.frame = Context()
                self.visible = True
                self.destroyed = False

            def destroy(self):
                self.destroyed = True

        ui = types.ModuleType("omni.ui")
        ui.ByteImageProvider = Provider
        ui.Window = Window
        ui.VStack = lambda **_kwargs: Context()
        ui.Label = lambda *_args, **_kwargs: None
        ui.ImageWithProvider = lambda *_args, **_kwargs: object()
        omni = types.ModuleType("omni")
        omni.ui = ui
        with mock.patch.dict(sys.modules, {"omni": omni, "omni.ui": ui}):
            preview = IsaacDepthPreview(2, 1)
            preview.update(np.array([[255, 0]], dtype=np.uint8))
            self.assertEqual(preview._provider.size, [2, 1])
            self.assertEqual(preview._provider.data, bytes([255, 255, 255, 255, 0, 0, 0, 255]))
            preview.close()
            self.assertFalse(preview._window.visible)
            self.assertTrue(preview._window.destroyed)

    def test_delayed_sequence_excludes_newer_frames(self):
        for length in (1, 5):
            for delay in (0, 1, 6):
                capacity = length + delay
                history = torch.zeros(capacity, 2, 1, 1, 1)
                for t in range(3 * capacity):
                    history[t % capacity] = t
                    if t >= capacity - 1:
                        sequence = depth_sequence_from_history(
                            history, (t - delay) % capacity, length,
                            torch.tensor([1]),
                        )
                        self.assertEqual(sequence.flatten().tolist(),
                                         list(range(t - delay - length + 1, t - delay + 1)))

    def test_invalid_values_remain_far_with_blur_and_noise(self):
        depth = torch.tensor([float('nan'), float('inf'), -float('inf'),
                              0., -1., .15, .5, 1., 3.]).reshape(1, 1, 1, -1)
        output = preprocess_depth(depth, min_z=.2, blur_sigma=.5, additive_noise_std=.01)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= .1) & (output <= 2.)).all())
        self.assertEqual(output.flatten()[[0, 1, 2, 3, 4, 5, 8]].tolist(), [2.] * 7)

    def test_metric_depth_and_dropout(self):
        depth = torch.tensor([.05, .5, 1., 2.5]).reshape(1, 1, 1, -1)
        self.assertEqual(preprocess_depth(depth).flatten().tolist(),
                         torch.tensor([.1, .5, 1., 2.]).tolist())
        self.assertTrue((preprocess_depth(depth, dropout_prob=1.) == 2.).all())


if __name__ == '__main__':
    unittest.main()
