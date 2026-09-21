from dataclasses import dataclass
import ast
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hilserl import storage_space


def test_checks_reserve_and_pending_payload_before_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HILSERL_MIN_FREE_GIB", "1")
    monkeypatch.setattr(storage_space.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**30 + 64 * 1024 + 4))
    raw = io.BytesIO()
    writer = storage_space.CapacityCheckedWriter(raw, tmp_path / "new" / "chunk")
    with pytest.raises(OSError, match="Storage reserve reached"):
        writer.write(b"12345")
    assert raw.getvalue() == b""
    assert writer.write(b"1234") == 4


@pytest.mark.parametrize("reserve", ["nan", "inf", "-1", "not-a-number"])
def test_invalid_reserve_cannot_disable_guard(tmp_path, monkeypatch, reserve):
    monkeypatch.setenv("HILSERL_MIN_FREE_GIB", reserve)
    with pytest.raises(ValueError):
        storage_space.require_space(tmp_path)


def test_array_estimate_uses_metadata_without_materializing_device_arrays():
    class DeviceArray:
        nbytes = 512
        def __array__(self):
            raise AssertionError("No device transfer for a space estimate")

    @dataclass
    class State:
        params: object
        opt_states: object
        apply_fn: object

    state = State({"weights": DeviceArray()}, (np.ones(3, dtype=np.float32), None), lambda: None)
    assert storage_space.array_storage_bytes(state) == 524


def test_checkpoint_backend_is_not_entered_without_payload_headroom(tmp_path, monkeypatch):
    tree = ast.parse((Path(__file__).parents[1] / "_run_actor.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_save_training_checkpoint")
    calls = []
    namespace = dict(os=os, checkpoints=SimpleNamespace(save_checkpoint=lambda *a, **k: calls.append((a, k))))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "checkpoint_guard", "exec"), namespace)
    monkeypatch.setenv("HILSERL_MIN_FREE_GIB", "1")
    monkeypatch.setattr(storage_space.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**30 + 64 * 2**20))
    with pytest.raises(OSError, match="Storage reserve reached"):
        namespace["_save_training_checkpoint"](tmp_path, {"params": np.ones(8, np.float32)}, 7)
    assert calls == []
    # A final save may spend the emergency reserve, but still requires its own
    # array payload and metadata estimate to fit. This avoids a pointless loss
    # of the last update when recording has just reached its low-water stop.
    namespace["_save_training_checkpoint"](tmp_path, {"params": np.ones(8, np.float32)}, 7, final=True)
    assert len(calls) == 1


def test_low_space_conversion_keeps_original(tmp_path, monkeypatch):
    from hilserl.replay_io import compress_existing_chunk, ReplayChunkError
    path = tmp_path / "buffer" / "transitions_7.pkl"
    path.parent.mkdir()
    path.write_bytes(b"the source stays byte-for-byte intact")
    monkeypatch.setenv("HILSERL_MIN_FREE_GIB", "1")
    monkeypatch.setattr(storage_space.shutil, "disk_usage", lambda directory: SimpleNamespace(free=2**30 - 1))
    with pytest.raises(ReplayChunkError, match="Storage reserve reached"):
        compress_existing_chunk(path, remove_source=True)
    assert path.read_bytes() == b"the source stays byte-for-byte intact"
    assert list(path.parent.iterdir()) == [path]


def test_actor_final_flush_can_use_reserve_without_skipping_space_checks(tmp_path, monkeypatch):
    from hilserl.replay_io import write_replay_chunk, load_replay_chunk
    monkeypatch.setenv("HILSERL_MIN_FREE_GIB", "8")
    monkeypatch.setattr(storage_space.shutil, "disk_usage", lambda directory: SimpleNamespace(free=2**20))
    path = tmp_path / "buffer" / "transitions_1.pkl.lz4"
    with pytest.raises(OSError, match="Storage reserve reached"):
        write_replay_chunk(path, [{"value": 7}])
    write_replay_chunk(path, [{"value": 7}], final=True)
    assert load_replay_chunk(path) == [{"value": 7}]
