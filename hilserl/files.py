"""Small file primitives shared by recording and the dependency-light console."""
import json
import os
from pathlib import Path
import time


def stamp():
    return {"monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns()}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default
