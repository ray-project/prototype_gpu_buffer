from typing import Tuple
import uuid


class GpuObjectRef:
    """Presents a reference to GPU buffer."""

    def __init__(self, id: uuid.UUID, group: str, src_rank: int, shape: Tuple, dtype):
        self.id = id
        self.group = group
        self.src_rank = src_rank
        self.shape = shape
        self.dtype = dtype
