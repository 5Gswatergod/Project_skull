from __future__ import annotations

import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0
