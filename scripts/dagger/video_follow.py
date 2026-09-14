"""Select video subjects without changing commands or terrain assignments."""

from bisect import bisect_right
from itertools import accumulate


def rotating_video_env(columns, *, first_env, initial_env, clip_index,
                       num_columns, terrain_proportions=None):
    """Cycle available terrain groups, then cycle subjects inside each group.

    Curriculum terrains assign types deterministically by column. Randomly
    generated terrains can only be grouped by column, not inferred terrain type.
    """
    groups = {}
    if terrain_proportions:
        names, weights = zip(*terrain_proportions)
        total = sum(weights)
        boundaries = list(accumulate(weight / total for weight in weights))
    for env_id in range(first_env, len(columns)):
        col = columns[env_id]
        if terrain_proportions:
            group = names[bisect_right(boundaries, col / num_columns + .001)]
        else:
            group = f"column-{col}"
        groups.setdefault(group, []).append(env_id)
    if not groups:
        raise ValueError("No eligible environments for video rotation.")
    ordered = list(groups)
    initial_group = next((name for name, ids in groups.items() if initial_env in ids), ordered[0])
    start = ordered.index(initial_group)
    group = ordered[(start + clip_index) % len(ordered)]
    ids = groups[group]
    if clip_index == 0 and initial_env in ids:
        return initial_env, group
    return ids[(clip_index // len(ordered)) % len(ids)], group
