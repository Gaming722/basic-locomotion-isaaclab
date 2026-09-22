"""Interactive IsaacLab play entry point for the GO1 depth DAgger student.

Delegates to train_dagger_go1.py's pure-student evaluation path so depth history,
sensor timing, action filtering and checkpoint loading cannot drift from training.
"""

import argparse
import os
from pathlib import Path
import sys


def eval_command(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="DAgger student checkpoint (.pt)")
    parser.add_argument("--terrain", choices=("stairs", "rough", "slope", "flat"), default="stairs")
    parser.add_argument("--stair_direction", choices=("up", "down"), default=None)
    parser.add_argument("--no_depth_window", action="store_true",
                        help="Disable the live student depth window (enabled by default).")
    args, remaining = parser.parse_known_args(argv)
    if args.stair_direction and args.terrain != "stairs":
        parser.error("--stair_direction requires --terrain stairs")
    evaluator = Path(__file__).with_name("train_dagger_go1.py")
    command = [sys.executable, str(evaluator),
            "--eval", "--keyboard", "--real-time",
            "--task=Locomotion-Go1-Rough-Vision-RayCaster",
            "--num_envs=1", f"--terrain={args.terrain}",
            "--difficulty=0.4666666667",
            f"--eval_policy={args.policy}"]
    if not args.no_depth_window:
        command.append("--show_depth")
    if args.terrain == "stairs":
        command.append(f"--stair_direction={args.stair_direction or 'up'}")
    return [*command, *remaining]


if __name__ == "__main__":
    os.execv(sys.executable, eval_command(sys.argv[1:]))
