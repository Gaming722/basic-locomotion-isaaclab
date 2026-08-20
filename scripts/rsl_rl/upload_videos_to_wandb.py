# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-off: upload recorded ``rl-video-*.mp4`` files to an existing W&B run.

Useful when a training run was started with ``--video --log_videos_wandb`` under an older
``train.py`` that missed the videos (e.g. wrong filename glob), or to backfill videos into
a finished run. Run it after the training process has exited to avoid two processes writing
to the same W&B run concurrently.

Example (run with the Isaac Lab python):

.. code-block:: bash

    python scripts/rsl_rl/upload_videos_to_wandb.py \
      --run_id 6e8v5jca \
      --video_dir logs/rsl_rl/rough_direct/2026-08-20_22-16-17/videos/train \
      --project basic-locomotion --fps 50
"""

import argparse
import glob
import os

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", type=str, required=True, help="W&B run id, e.g. 6e8v5jca")
    parser.add_argument("--video_dir", type=str, required=True, help="Folder containing rl-video-*.mp4 files")
    parser.add_argument("--project", type=str, default="basic-locomotion", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/username (default: env WANDB_USERNAME)")
    parser.add_argument("--fps", type=int, default=50, help="Playback fps for the uploaded videos")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.video_dir, "rl-video-*.mp4")))
    if not videos:
        print(f"[WARN] No rl-video-*.mp4 files found in {args.video_dir}")
        return

    print(f"[INFO] Resuming W&B run {args.run_id} (project={args.project}) to upload {len(videos)} videos.")
    run = wandb.init(id=args.run_id, project=args.project, entity=args.entity, resume="must")
    try:
        for video_path in videos:
            print(f"[INFO] Uploading {os.path.basename(video_path)} ...")
            wandb.log({"train_video": wandb.Video(video_path, format="mp4", fps=args.fps)})
    finally:
        run.finish()
    print(f"[INFO] Done. Uploaded {len(videos)} videos to run {args.run_id}.")


if __name__ == "__main__":
    main()
