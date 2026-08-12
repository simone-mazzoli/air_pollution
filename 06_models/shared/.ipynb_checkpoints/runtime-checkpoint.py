import os

import torch

from .config import CPU_INTEROP_THREADS, CPU_THREADS

_APPLIED = False


def apply_runtime_config():
    global _APPLIED
    if _APPLIED:
        return
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    _APPLIED = True


def runtime_summary():
    return (f"CPU threads: intra-op={torch.get_num_threads()} "
            f"inter-op={torch.get_num_interop_threads()} "
            f"(os.cpu_count={os.cpu_count()})")
