"""Hardware-free checks for the student-depth preview conversion."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from view_student_depth import prepare_depth


class StudentDepthPreviewTests(unittest.TestCase):
    def test_depth_and_grayscale_mapping(self):
        raw = np.full((480, 848), 1000, dtype=np.uint16)
        raw[0:8, 0:8] = 0
        raw[0:8, 8:16] = 50
        raw[0:8, 16:24] = 100
        raw[0:8, 24:32] = 2000
        raw[0:8, 32:40] = 4000
        depth, gray = prepare_depth(raw, 0.001)
        self.assertEqual(depth.shape, (60, 106))
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth[0, :6], [2., .1, .1, 2., 2., 1.], atol=1e-6)
        np.testing.assert_array_equal(gray[0, :5], [0, 255, 255, 0, 0])
        self.assertEqual(gray.dtype, np.uint8)

    def test_zero_remains_invalid_during_resize(self):
        raw = np.full((480, 848), 1000, dtype=np.uint16)
        raw[0:8, 0:8] = 0
        depth, gray = prepare_depth(raw, 0.001)
        self.assertEqual(depth[0, 0], 2.)
        self.assertEqual(gray[0, 0], 0)
        self.assertAlmostEqual(float(depth[0, 1]), 1., places=6)

    def test_rejects_non_z16_or_bad_scale(self):
        with self.assertRaises(ValueError):
            prepare_depth(np.ones((8, 8), dtype=np.float32), .001)
        with self.assertRaises(ValueError):
            prepare_depth(np.ones((8, 8), dtype=np.uint16), 0.)


if __name__ == '__main__':
    unittest.main()
