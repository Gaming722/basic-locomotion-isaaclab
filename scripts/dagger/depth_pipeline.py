"""Metric Z-depth preprocessing shared by DAgger and sim2sim."""

import math

import torch
import torch.nn.functional as F

PIPELINE_VERSION = 4
_kernels = {}


def preprocess_depth(depth, *, blur_sigma=0.0, additive_noise_std=0.0,
                     dropout_prob=0.0, min_z=0.0):
    """Process NCHW metres; nonfinite/nonpositive/sub-Min-Z pixels remain far.

    Invalid pixels are identified before clipping or filtering. Their far code
    is restored after filtering/noise so invalid measurements never become near.
    """
    invalid = ~torch.isfinite(depth) | (depth <= 0) | (depth >= 3.0)
    if min_z > 0:
        invalid |= depth < min_z
    depth = depth.masked_fill(invalid, 2.0).clamp(0.1, 2.0)
    if blur_sigma > 0:
        size = max(3, 2 * math.ceil(2 * blur_sigma) + 1)
        key = (size, blur_sigma, str(depth.device), depth.dtype)
        if key not in _kernels:
            axis = torch.arange(size, device=depth.device, dtype=depth.dtype) - size // 2
            weights = torch.exp(-axis.square() / (2 * blur_sigma ** 2))
            weights /= weights.sum()
            _kernels[key] = (weights[:, None] * weights[None, :])[None, None]
        channels = depth.shape[1]
        depth = F.conv2d(F.pad(depth, (size // 2,) * 4, mode="replicate"),
                         _kernels[key].expand(channels, 1, size, size), groups=channels)
    if additive_noise_std > 0:
        depth = depth + torch.randn_like(depth) * additive_noise_std
    depth = depth.clamp(0.1, 2.0)
    if min_z > 0:
        invalid |= depth < min_z
    if dropout_prob > 0:
        invalid |= torch.rand_like(depth) < dropout_prob
    return depth.masked_fill(invalid, 2.0)


def depth_sequence_from_history(history, newest_index, sequence_length, env_indices=None):
    """Select a chronological sequence ending at newest_index in a ring buffer."""
    if not 0 < sequence_length <= history.shape[0]:
        raise ValueError("Depth sequence length must fit the history buffer.")
    indices = (torch.arange(sequence_length, device=history.device)
               + newest_index - sequence_length + 1) % history.shape[0]
    if env_indices is not None:
        history = history.index_select(1, env_indices.to(history.device))
    return history.index_select(0, indices).permute(1, 0, 2, 3, 4).contiguous()
