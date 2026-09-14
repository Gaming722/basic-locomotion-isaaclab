import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_follow import rotating_video_env


class VideoFollowTest(unittest.TestCase):
    def test_cycles_available_types_and_excludes_standing_group(self):
        columns = [0, 0, 1, 1, 2, 2, 3, 3]
        choices = [rotating_video_env(
            columns, first_env=2, initial_env=4, clip_index=i, num_columns=4,
            terrain_proportions=[('flat', .25), ('rough', .25), ('up', .25), ('down', .25)],
        ) for i in range(7)]
        self.assertEqual(choices[0], (4, 'up'))
        self.assertEqual([label for _, label in choices[:3]], ['up', 'down', 'rough'])
        self.assertTrue(all(env_id >= 2 for env_id, _ in choices))
        self.assertEqual(choices[3], (5, 'up'))

    def test_random_terrain_uses_column_labels(self):
        choices = [rotating_video_env([0, 1, 2], first_env=0, initial_env=0,
                                     clip_index=i, num_columns=3) for i in range(3)]
        self.assertEqual(choices, [(0, 'column-0'), (1, 'column-1'), (2, 'column-2')])


if __name__ == '__main__':
    unittest.main()
