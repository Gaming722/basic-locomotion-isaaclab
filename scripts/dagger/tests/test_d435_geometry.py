import math
import sys
import unittest
from pathlib import Path

asset_dir = (Path(__file__).resolve().parents[3] / "source" /
             "basic_locomotion_isaaclab" / "basic_locomotion_isaaclab" / "assets")
sys.path.insert(0, str(asset_dir))
from d435_geometry import D435_DEPTH_POSITION_BASE, d435_depth_position_base


class D435GeometryTest(unittest.TestCase):
    def test_unrotated_screw_offset(self):
        for actual, expected in zip(d435_depth_position_base((0, 0, 0), 0),
                                    (.0106, .0175, .0125)):
            self.assertAlmostEqual(actual, expected)

    def test_downward_mount_rotates_translation(self):
        for actual, expected in zip(d435_depth_position_base((0, 0, 0), 90),
                                    (.0125, .0175, -.0106)):
            self.assertAlmostEqual(actual, expected)
        c = math.sqrt(3) / 2
        expected = (.26 + c * .0106 + .5 * .0125,
                    .0175, .12 - .5 * .0106 + c * .0125)
        for actual, wanted in zip(D435_DEPTH_POSITION_BASE, expected):
            self.assertAlmostEqual(actual, wanted)


if __name__ == '__main__':
    unittest.main()
