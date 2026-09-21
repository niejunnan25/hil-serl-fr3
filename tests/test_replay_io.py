"""Exercise the production chunk format without JAX or any robot connection."""
import ast
import gzip
import hashlib
import os
from pathlib import Path
import pickle
import stat
from types import SimpleNamespace

import lz4.frame
import numpy as np
import pytest

from hilserl import replay_io


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actor_helpers():
    names = {"_next_dump_step", "_dump_pending_transitions", "_reload_online_buffers"}
    nodes = [node for node in ast.parse((ROOT / "_run_actor.py").read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "_run_actor.py"), "exec"), namespace)
    return SimpleNamespace(**namespace)


def transition(index=0):
    return {
        "observations": {
            "image": np.full((2, 8, 8, 3), index, dtype=np.uint8),
            "state": np.arange(19, dtype=np.float32),
        },
        "next_observations": {"image": np.full((2, 8, 8, 3), index + 1, dtype=np.uint8)},
        "actions": np.array([0.0, -0.0, np.inf, np.nan], dtype=np.float32),
        "infos": {"grasp_penalty": np.float32(-1.0), "raw_transition_id": str(index),
                  "nested": (b"exact bytes", [True, None])},
        "rewards": np.float32(index),
        "dones": np.bool_(False),
    }


def assert_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_exact(left, right)
    else:
        assert actual == expected


class Buffer:
    def __init__(self):
        self.items = []

    def insert(self, value):
        self.items.append(value)


def legacy_chunk(root, index=0, payload=None):
    directory = root / "checkpoints" / "buffer"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"transitions_{index}.pkl"
    path.write_bytes(pickle.dumps([transition(index)]) if payload is None else payload)
    return path


def test_actor_dump_and_resume_preserve_nested_arrays_and_types(actor_helpers, tmp_path):
    expected = [transition(0), transition(1)]
    saved = actor_helpers._dump_pending_transitions(tmp_path, "buffer", 100, expected)
    assert Path(saved).name == "transitions_100.pkl.lz4"
    assert not list((tmp_path / "buffer").glob("*.pkl"))
    frame = lz4.frame.get_frame_info(Path(saved).read_bytes())
    assert frame["content_checksum"] is True
    assert_exact(replay_io.load_replay_chunk(saved), expected)
    online, intervention = Buffer(), Buffer()
    assert actor_helpers._reload_online_buffers(tmp_path, online, intervention) == {"buffer": 2, "demo_buffer": 0}
    for actual, wanted in zip(online.items, expected):
        assert actual.pop("grasp_penalty") == -1.0
        assert_exact(actual, wanted)
    # Dumping must not mutate the pending in-memory data.
    assert "grasp_penalty" not in expected[0]


def test_mixed_format_reload_preserves_mtime_then_index_order(actor_helpers, tmp_path):
    directory = tmp_path / "buffer"
    directory.mkdir()
    saved = []
    for index, compressed, timestamp in [(15, False, 100), (2, True, 300), (4, True, 200), (3, False, 200)]:
        path = directory / f"transitions_{index}.pkl{'.lz4' if compressed else ''}"
        value = [{"source": index}]
        if compressed:
            replay_io.write_replay_chunk(path, value)
        else:
            path.write_bytes(pickle.dumps(value))
        os.utime(path, ns=(timestamp, timestamp))
        saved.append(path)
    (directory / "transitions_900.pkl.tmp").write_bytes(b"not committed")
    (directory / ".transitions_901.pkl.lz4.abc.partial").write_bytes(b"not committed")
    online = Buffer()
    actor_helpers._reload_online_buffers(tmp_path, online, Buffer())
    assert [item["source"] for item in online.items] == [15, 3, 4, 2]
    assert actor_helpers._next_dump_step(directory, 0) == 16
    assert actor_helpers._next_dump_step(directory, 100) == 100


@pytest.mark.parametrize("existing_suffix", [".pkl", ".pkl.lz4"])
def test_logical_chunk_collision_never_overwrites(tmp_path, existing_suffix):
    directory = tmp_path / "buffer"
    directory.mkdir()
    existing = directory / f"transitions_5{existing_suffix}"
    original = b"previous data"
    existing.write_bytes(original)
    with pytest.raises(replay_io.ReplayChunkError, match="already exists"):
        replay_io.write_replay_chunk(directory / "transitions_5.pkl.lz4", [])
    assert existing.read_bytes() == original


@pytest.mark.parametrize("other_name", ["transitions_0.pkl.lz4", "transitions_00.pkl"])
def test_duplicate_index_rejected_before_reload(actor_helpers, tmp_path, other_name):
    directory = tmp_path / "buffer"
    directory.mkdir()
    (directory / "transitions_0.pkl").write_bytes(pickle.dumps([{}]))
    (directory / other_name).write_bytes(b"duplicate must not load")
    online = Buffer()
    with pytest.raises(replay_io.ReplayChunkError, match="Duplicate replay chunk index"):
        actor_helpers._reload_online_buffers(tmp_path, online, Buffer())
    assert online.items == []


@pytest.mark.parametrize("damage", ["truncated_footer", "truncated_body", "checksum", "gzip", "trailing"])
def test_damaged_compressed_chunks_fail_closed(actor_helpers, tmp_path, damage):
    directory = tmp_path / "buffer"
    directory.mkdir()
    path = directory / "transitions_0.pkl.lz4"
    replay_io.write_replay_chunk(path, [transition()])
    data = path.read_bytes()
    if damage == "truncated_footer":
        data = data[:-1]
    elif damage == "truncated_body":
        data = data[:len(data) // 2]
    elif damage == "checksum":
        data = data[:-1] + bytes([data[-1] ^ 1])
    elif damage == "gzip":
        data = gzip.compress(pickle.dumps([transition()]))
    else:
        data += b"trailing garbage"
    path.write_bytes(data)
    online = Buffer()
    with pytest.raises(replay_io.ReplayChunkError, match="Cannot load persisted replay chunk"):
        actor_helpers._reload_online_buffers(tmp_path, online, Buffer())
    assert online.items == []


def test_unsupported_compression_suffix_cannot_be_silently_omitted(tmp_path):
    path = tmp_path / "transitions_1.pkl.gz"
    path.write_bytes(gzip.compress(pickle.dumps([{}])))
    with pytest.raises(replay_io.ReplayChunkError, match="Unrecognized"):
        replay_io.replay_chunks(tmp_path)


@pytest.mark.parametrize("compressed", [False, True])
def test_multiple_pickles_in_one_chunk_are_rejected(tmp_path, compressed):
    path = tmp_path / ("transitions_1.pkl.lz4" if compressed else "transitions_1.pkl")
    data = pickle.dumps([{}]) + pickle.dumps([{}])
    path.write_bytes(lz4.frame.compress(data, content_checksum=True) if compressed else data)
    with pytest.raises(replay_io.ReplayChunkError, match="Trailing data"):
        replay_io.load_replay_chunk(path)


def test_frame_without_checksum_is_rejected(tmp_path):
    path = tmp_path / "transitions_0.pkl.lz4"
    path.write_bytes(lz4.frame.compress(pickle.dumps([{}]), content_checksum=False))
    with pytest.raises(replay_io.ReplayChunkError, match="no content checksum"):
        replay_io.load_replay_chunk(path)


def test_conversion_is_byte_exact_without_deserialization_preserves_metadata_and_reenters(tmp_path, monkeypatch):
    original = pickle.dumps([transition()], protocol=4)
    path = legacy_chunk(tmp_path, payload=original)
    os.chmod(path, 0o640)
    os.utime(path, ns=(111111111, 987654321))
    monkeypatch.setattr(replay_io.pickle, "load", lambda *args: pytest.fail("conversion must not deserialize"))
    first = replay_io.compress_existing_chunk(path)
    target = Path(first["destination"])
    assert path.read_bytes() == original
    assert lz4.frame.decompress(target.read_bytes()) == original
    assert first["source_sha256"] == first["decompressed_sha256"] == hashlib.sha256(original).hexdigest()
    assert first["compressed_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert target.stat().st_mtime_ns == 987654321
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert first["source_removed"] is False
    with pytest.raises(replay_io.ReplayChunkError, match="Duplicate"):
        replay_io.replay_chunks(path.parent)
    second = replay_io.compress_existing_chunk(path, remove_source=True)
    assert second["source_removed"] is True
    assert second["reused_destination"] is True
    assert second["bytes_released"] == len(original)
    assert not path.exists()
    assert replay_io.replay_chunks(path.parent) == [target]
    third = replay_io.compress_existing_chunk(path, remove_source=True)
    assert third["bytes_released"] == 0
    assert third["source_removed"] is False
    assert third["source_present"] is False
    assert third["source_sha256"] is None
    assert third["decompressed_sha256"] == first["source_sha256"]
    assert target.stat().st_mtime_ns == 987654321


def test_conversion_and_release_report_actual_disk_savings(tmp_path):
    path = legacy_chunk(tmp_path, payload=b"not a pickle, exact source bytes" * 1000)
    original_size = path.stat().st_size
    result = replay_io.compress_existing_chunk(path, remove_source=True)
    assert result["before_bytes"] == original_size
    assert result["after_bytes"] == Path(result["destination"]).stat().st_size
    assert result["bytes_released"] == result["before_bytes"] - result["after_bytes"] > 0
    assert not path.exists()


@pytest.mark.parametrize("failure", ["write", "fsync", "commit", "commit_directory_fsync"])
def test_conversion_io_failure_keeps_original(tmp_path, monkeypatch, failure):
    path = legacy_chunk(tmp_path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    if failure == "write":
        monkeypatch.setattr(replay_io, "_compressed_writer", fail)
    elif failure == "fsync":
        monkeypatch.setattr(replay_io.os, "fsync", fail)
    elif failure == "commit_directory_fsync":
        monkeypatch.setattr(replay_io, "_sync_directory", fail)
    else:
        monkeypatch.setattr(replay_io, "_commit_new_file", fail)
    with pytest.raises(replay_io.ReplayChunkError):
        replay_io.compress_existing_chunk(path, remove_source=True)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("target_type", ["different", "truncated", "symlink"])
def test_existing_nonmatching_destination_retains_source(tmp_path, target_type):
    path = legacy_chunk(tmp_path)
    original = path.read_bytes()
    target = path.with_name(path.name + ".lz4")
    other = lz4.frame.compress(pickle.dumps([{"other": True}]), content_checksum=True)
    if target_type == "different":
        target.write_bytes(other)
    elif target_type == "truncated":
        target.write_bytes(other[:-2])
    else:
        outside = tmp_path / "outside.lz4"
        outside.write_bytes(other)
        target.symlink_to(outside)
    with pytest.raises(replay_io.ReplayChunkError):
        replay_io.compress_existing_chunk(path, remove_source=True)
    assert path.read_bytes() == original


def test_source_change_during_conversion_retains_source(tmp_path, monkeypatch):
    path = legacy_chunk(tmp_path)
    real_decode = replay_io._decoded_digest

    def mutate_then_decode(target):
        path.write_bytes(b"new data appended by a concurrent writer")
        return real_decode(target)

    monkeypatch.setattr(replay_io, "_decoded_digest", mutate_then_decode)
    with pytest.raises(replay_io.ReplayChunkError, match="source changed"):
        replay_io.compress_existing_chunk(path, remove_source=True)
    assert path.read_bytes() == b"new data appended by a concurrent writer"
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("relative", ["demo_dir/example_success.pkl", "exports/transitions_1.pkl", "buffer/example_success.pkl"])
def test_conversion_does_not_touch_original_demos_or_exports(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"protected bytes")
    with pytest.raises(replay_io.ReplayChunkError):
        replay_io.compress_existing_chunk(path, remove_source=True)
    assert path.read_bytes() == b"protected bytes"


def test_actor_dump_failure_keeps_pending_data(actor_helpers, tmp_path, monkeypatch):
    pending = [transition()]

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(replay_io, "_compressed_writer", fail)
    assert actor_helpers._dump_pending_transitions(tmp_path, "buffer", 0, pending) is None
    assert len(pending) == 1
    assert list((tmp_path / "buffer").iterdir()) == []


def test_atomic_commit_refuses_target_created_after_index_check(tmp_path, monkeypatch):
    path = tmp_path / "buffer" / "transitions_1.pkl.lz4"
    commit = replay_io._commit_new_file
    concurrent_data = b"chunk committed by the other writer"

    def race(temporary, target):
        target.write_bytes(concurrent_data)
        commit(temporary, target)

    monkeypatch.setattr(replay_io, "_commit_new_file", race)
    with pytest.raises(replay_io.ReplayChunkError):
        replay_io.write_replay_chunk(path, [transition()])
    assert path.read_bytes() == concurrent_data
    assert list(path.parent.iterdir()) == [path]


def test_failed_commit_does_not_leave_actor_retry_as_a_duplicate(actor_helpers, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(replay_io, "_sync_directory", fail)
    pending = [transition()]
    assert actor_helpers._dump_pending_transitions(tmp_path, "buffer", 0, pending) is None
    assert list((tmp_path / "buffer").iterdir()) == []
    assert len(pending) == 1
