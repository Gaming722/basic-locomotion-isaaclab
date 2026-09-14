import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_pipeline import depth_sequence_from_history, preprocess_depth


class DepthPipelineTest(unittest.TestCase):
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
