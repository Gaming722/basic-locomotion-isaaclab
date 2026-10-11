import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proprio_noise import ProprioNoiseCfg, ProprioceptiveNoise


class ProprioNoiseTests(unittest.TestCase):
    def test_only_sensor_terms_are_corrupted_and_gravity_remains_unit(self):
        torch.manual_seed(7)
        clean = torch.zeros(3, 5, 45)
        clean[..., 5] = -1.0
        noise = ProprioceptiveNoise(3, 5, "cpu", ProprioNoiseCfg())
        result = noise.update(clean.reshape(3, -1)).reshape(3, 5, 45)
        self.assertTrue(torch.equal(result[..., 6:9], clean[..., 6:9]))
        self.assertTrue(torch.equal(result[..., 33:45], clean[..., 33:45]))
        self.assertFalse(torch.equal(result[..., 0:3], clean[..., 0:3]))
        torch.testing.assert_close(result[..., 3:6].norm(dim=-1), torch.ones(3, 5))

    def test_history_keeps_old_noise_and_reset_rebuilds_history(self):
        torch.manual_seed(11)
        cfg = ProprioNoiseCfg(angular_velocity_std=0.1)
        noise = ProprioceptiveNoise(2, 5, "cpu", cfg)
        clean = torch.zeros(2, 5, 45)
        clean[..., 5] = -1.0
        first = noise.update(clean.reshape(2, -1)).reshape(2, 5, 45).clone()
        second = noise.update(clean.reshape(2, -1)).reshape(2, 5, 45).clone()
        torch.testing.assert_close(second[:, :-1], first[:, 1:])
        reset = torch.tensor([True, False])
        reset_result = noise.update(clean.reshape(2, -1), reset).reshape(2, 5, 45)
        self.assertFalse(torch.equal(reset_result[0], second[0]))

    def test_negative_noise_is_rejected(self):
        with self.assertRaises(ValueError):
            ProprioNoiseCfg(joint_velocity_std=-0.1).validate()

    def test_zero_reset_padding_stays_zero(self):
        clean = torch.zeros(1, 5, 45)
        clean[:, -1, 5] = -1.0
        noise = ProprioceptiveNoise(1, 5, "cpu", ProprioNoiseCfg())
        result = noise.update(clean.reshape(1, -1)).reshape(1, 5, 45)
        torch.testing.assert_close(result[:, :-1], clean[:, :-1])
        self.assertFalse(torch.equal(result[:, -1], clean[:, -1]))


if __name__ == "__main__":
    unittest.main()
