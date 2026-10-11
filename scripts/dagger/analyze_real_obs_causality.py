#!/usr/bin/env python3
"""Counterfactual analysis of a recorded Go1 real-world RL trajectory.

This is an offline diagnostic.  It reproduces checkpoint actions from the exact
recorded inputs, then freezes one observation block at its RL-entry history to
measure which changing feedback channels influence the action escalation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dagger_network import DaggerNet


BLOCKS = {
    "angular_velocity": (0, 3),
    "projected_gravity": (3, 6),
    "command": (6, 9),
    "joint_position_offset": (9, 21),
    "joint_velocity": (21, 33),
    "previous_raw_action": (33, 45),
}


def load_network(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    network = DaggerNet(
        vec_size=metadata["common_obs_size"],
        output_size=metadata["action_size"],
        depth_channels=metadata["depth_channels"],
    )
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.eval()
    return network


def infer(network, depth, common):
    with torch.no_grad():
        action, _ = network(torch.from_numpy(depth), torch.from_numpy(common))
    return action.numpy()


def metrics(candidate, reference):
    difference = candidate - reference
    return {
        "action_abs_max": float(np.max(np.abs(candidate))),
        "final_action_abs_max": float(np.max(np.abs(candidate[-1]))),
        "action_delta_rms": float(np.sqrt(np.mean(difference**2))),
        "final_action_delta_rms": float(np.sqrt(np.mean(difference[-1] ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    network = load_network(args.checkpoint)
    with np.load(args.recording) as source:
        common = source["common"].astype(np.float32)
        depth = source["depth_sequence"].astype(np.float32)[:, :, None]
        recorded_action = source["raw_action"].astype(np.float32)
        elapsed = source["elapsed"].astype(np.float64) if "elapsed" in source else None

    if common.shape[1] != 225 or depth.shape[1:] != (5, 1, 60, 106):
        raise ValueError("Expected common [N,225] and depth [N,5,1,60,106]")

    reproduced = infer(network, depth, common)
    reproduction_error = np.abs(reproduced - recorded_action)
    report = {
        "samples": int(len(common)),
        "duration_seconds": float(elapsed[-1] - elapsed[0])
        if elapsed is not None and len(elapsed) > 1
        else None,
        "reproduction": {
            "max_abs_error": float(np.max(reproduction_error)),
            "mean_abs_error": float(np.mean(reproduction_error)),
        },
        "baseline": metrics(reproduced, reproduced),
        "entry_frozen_ablation": {},
    }

    common_frames = common.reshape(-1, 5, 45)
    for name, (start, stop) in BLOCKS.items():
        modified = common_frames.copy()
        modified[:, :, start:stop] = modified[0:1, :, start:stop]
        candidate = infer(network, depth, modified.reshape(-1, 225))
        report["entry_frozen_ablation"][name] = metrics(candidate, reproduced)

    modified = common_frames.copy()
    modified[:, :, 9:33] = modified[0:1, :, 9:33]
    candidate = infer(network, depth, modified.reshape(-1, 225))
    report["entry_frozen_ablation"]["joint_position_and_velocity"] = metrics(
        candidate, reproduced
    )

    frozen_depth = np.repeat(depth[0:1], len(depth), axis=0)
    candidate = infer(network, frozen_depth, common)
    report["entry_frozen_ablation"]["depth"] = metrics(candidate, reproduced)

    if len(depth) > 1:
        per_step_change = np.max(np.abs(np.diff(depth, axis=0)), axis=(1, 2, 3, 4))
        report["depth_temporal_change"] = {
            "max": float(np.max(per_step_change)),
            "after_first_five_steps_max": float(np.max(per_step_change[4:]))
            if len(per_step_change) > 4
            else None,
        }

    output_text = json.dumps(report, indent=2) + "\n"
    print(output_text, end="")
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_text)
        print("Saved report: {}".format(output))


if __name__ == "__main__":
    main()
