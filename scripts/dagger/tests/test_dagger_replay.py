import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dagger_network import DaggerReplayBuffer


def batch(start: int, count: int):
    values = torch.arange(start, start + count, dtype=torch.float32)
    return (
        values[:, None, None, None, None].expand(-1, 1, 1, 1, 1).clone(),
        values[:, None].clone(),
        values[:, None].clone(),
    )


class DaggerReplayTests(unittest.TestCase):
    def test_fifo_keeps_latest_samples(self):
        replay = DaggerReplayBuffer(4)
        replay.add_batch(*batch(0, 4))
        replay.add_batch(*batch(4, 2))
        self.assertEqual(len(replay), 4)
        self.assertEqual(set(replay._common_obs[:, 0].tolist()), {2.0, 3.0, 4.0, 5.0})

    def test_reservoir_tracks_total_seen_without_growing(self):
        torch.manual_seed(3)
        replay = DaggerReplayBuffer(4, reservoir=True)
        replay.add_batch(*batch(0, 10))
        self.assertEqual(len(replay), 4)
        self.assertEqual(replay.total_seen, 10)
        self.assertTrue(torch.all(replay._common_obs[:, 0] >= 0))
        self.assertTrue(torch.all(replay._common_obs[:, 0] < 10))


if __name__ == "__main__":
    unittest.main()
