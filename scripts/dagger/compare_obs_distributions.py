#!/usr/bin/env python3
"""Compare Isaac Lab and physical-Go1 student observation distributions."""

import argparse
import json
from pathlib import Path

import numpy as np


BLOCKS = (
    ("angular_velocity", 0, 3),
    ("projected_gravity", 3, 6),
    ("command", 6, 9),
    ("joint_position_offset", 9, 21),
    ("joint_velocity", 21, 33),
    ("previous_raw_action", 33, 45),
)


def frames(array):
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 225:
        raise ValueError("common must have shape (samples, 225)")
    return value.reshape(-1, 5, 45)


def stats(value):
    flat = np.asarray(value).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="NPZ produced by Isaac Lab --obs_dump")
    parser.add_argument("--real", required=True, help="NPZ produced by physical shadow RL")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument(
        "--real-previous-action-valid",
        action="store_true",
        help="Treat real previous-action values as valid. Leave unset for legacy shadow captures that fed back unexecuted predictions.",
    )
    args = parser.parse_args()

    with np.load(args.sim) as sim_file, np.load(args.real) as real_file:
        sim = frames(sim_file["common"])
        real = frames(real_file["common"])

    report = {"sim_samples": len(sim), "real_samples": len(real), "blocks": {}}
    print("block                       sim_mean  sim_std real_mean real_std mean_shift/std status")
    for name, start, stop in BLOCKS:
        sim_stats = stats(sim[:, :, start:stop])
        real_stats = stats(real[:, :, start:stop])
        scale = max(sim_stats["std"], 1e-6)
        shift = abs(real_stats["mean"] - sim_stats["mean"]) / scale
        valid = name != "previous_raw_action" or args.real_previous_action_valid
        status = "ok" if shift < 2.0 else ("review" if shift < 4.0 else "OOD")
        if not valid:
            status = "invalid-shadow-feedback"
        report["blocks"][name] = {
            "sim": sim_stats,
            "real": real_stats,
            "mean_shift_in_sim_std": float(shift),
            "valid": valid,
            "status": status,
        }
        print(f"{name:27s} {sim_stats['mean']:+8.4f} {sim_stats['std']:8.4f} "
              f"{real_stats['mean']:+9.4f} {real_stats['std']:8.4f} {shift:14.2f} {status}")

    # Per-dimension standardized mean shifts catch opposing joint biases hidden by block means.
    sim_flat = sim.reshape(-1, 45)
    real_flat = real.reshape(-1, 45)
    sim_mean = sim_flat.mean(0)
    sim_std = sim_flat.std(0)
    real_mean = real_flat.mean(0)
    dimension_shift = np.abs(real_mean - sim_mean) / np.maximum(sim_std, 1e-6)
    report["per_dimension_mean_shift_in_sim_std"] = dimension_shift.tolist()
    ranked = np.argsort(dimension_shift)[::-1]
    print("\nLargest per-dimension mean shifts:")
    for index in ranked[:12]:
        valid = not (index >= 33 and not args.real_previous_action_valid)
        suffix = "" if valid else " (invalid shadow feedback)"
        print(f"  common[{index:02d}] shift={dimension_shift[index]:8.2f} "
              f"sim={sim_mean[index]:+.4f} real={real_mean[index]:+.4f}{suffix}")

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
