"""Shared free-space checks for recording-adjacent writes, without loading JAX."""
from dataclasses import fields, is_dataclass
import math
import os
from pathlib import Path
import shutil


def require_space(path, *, pending_bytes=0, reserve_gib=None):
    reserve = float(os.environ.get("HILSERL_MIN_FREE_GIB", "8") if reserve_gib is None else reserve_gib)
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("HILSERL_MIN_FREE_GIB must be finite and nonnegative")
    if type(pending_bytes) is not int or pending_bytes < 0:
        raise ValueError("pending_bytes must be a nonnegative integer")
    directory = Path(path).resolve()
    while not directory.is_dir():
        if directory == directory.parent:
            raise OSError(f"Cannot determine storage filesystem: {path}")
        directory = directory.parent
    free = shutil.disk_usage(directory).free
    required = int(reserve * 2**30) + pending_bytes
    if free < required:
        raise OSError(f"Storage reserve reached: {free / 2**30:.2f} GiB available; "
                      f"need {pending_bytes / 2**20:.1f} MiB for this write plus {reserve:g} GiB reserve")


def array_storage_bytes(value):
    """Conservative array payload size; use shape metadata without GPU transfers."""
    size = getattr(value, "nbytes", None)
    if size is not None:
        return int(size)
    if isinstance(value, dict) or hasattr(value, "values"):
        return sum(array_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(array_storage_bytes(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(array_storage_bytes(getattr(value, field.name)) for field in fields(value))
    return 0


class CapacityCheckedWriter:
    """Check compressed blocks before forwarding them to the actual file."""
    def __init__(self, raw, path, *, reserve_gib=None):
        self.raw, self.path, self.reserve_gib = raw, path, reserve_gib

    def write(self, data):
        require_space(self.path, pending_bytes=len(data) + 64 * 1024, reserve_gib=self.reserve_gib)
        return self.raw.write(data)

    def __getattr__(self, name):
        return getattr(self.raw, name)
