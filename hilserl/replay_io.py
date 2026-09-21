"""Lossless, atomic storage for locally generated online replay chunks.

Only checkpoint ``buffer`` and ``demo_buffer`` chunks use this format. Original
demonstrations and exported recordings remain unchanged. Callers converting
existing chunks must hold the run's writer/launch lock and stop its writers.
Pickles are trusted local training artifacts, never an external input format.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import re
import stat
import tempfile

import lz4.frame
from hilserl.storage_space import CapacityCheckedWriter, require_space


_CHUNK_NAME = re.compile(r"transitions_([0-9]+)\.pkl(?:\.lz4)?\Z")
_BLOCK_BYTES = 1024 * 1024


class ReplayChunkError(RuntimeError):
    """A saved replay chunk cannot safely be written, resumed, or converted."""


def chunk_index(path):
    match = _CHUNK_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ReplayChunkError(f"Unrecognized persisted replay chunk: {path}")
    return int(match.group(1))


def replay_chunks(directory):
    """List both formats in original save order, rejecting ambiguous indices."""
    directory = Path(directory)
    if not directory.exists():
        return []
    indexed = {}
    for path in directory.iterdir():
        # Both our independent temporary files and old .pkl.tmp files are
        # uncommitted. They must never enter a resumed replay buffer.
        if path.name.startswith(".") or path.name.endswith((".tmp", ".partial")):
            continue
        if not (path.name.startswith("transitions_") or ".pkl" in path.name):
            continue
        index = chunk_index(path)
        if path.is_symlink() or not path.is_file():
            raise ReplayChunkError(f"Replay chunk must be a regular local file: {path}")
        if index in indexed:
            raise ReplayChunkError(
                f"Duplicate replay chunk index {index}: {indexed[index]} and {path}. "
                "Finish the verified compression conversion before resuming; "
                "never load both copies or delete an unverified source."
            )
        indexed[index] = path
    return sorted(indexed.values(), key=lambda item: (item.stat().st_mtime_ns, chunk_index(item)))


def next_chunk_index(directory, step):
    return max(int(step) if step is not None else 0,
               max((chunk_index(path) for path in replay_chunks(directory)), default=-1) + 1)


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_file(path):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    return descriptor, Path(temporary)


def _commit_new_file(temporary, path):
    # link() commits a complete same-filesystem file atomically and refuses to
    # overwrite a destination that appeared after chunk-index allocation.
    os.link(temporary, path)
    try:
        temporary.unlink()
        _sync_directory(path.parent)
    except BaseException:
        # A failed commit must not leave an apparently successful chunk that a
        # caller would duplicate while retrying its still-pending transitions.
        path.unlink(missing_ok=True)
        raise


def _compressed_writer(raw, path, *, reserve_gib=None):
    return lz4.frame.open(CapacityCheckedWriter(raw, path, reserve_gib=reserve_gib), mode="wb",
                          compression_level=0, content_checksum=True)


def write_replay_chunk(path, transitions, *, final=False):
    """Commit one new .pkl.lz4 chunk; never overwrite a logical chunk index."""
    path = Path(path)
    index = chunk_index(path)
    if not path.name.endswith(".pkl.lz4"):
        raise ReplayChunkError(f"New replay chunks require .pkl.lz4: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_space(path.parent, reserve_gib=0 if final else None)
    if any(chunk_index(saved) == index for saved in replay_chunks(path.parent)):
        raise ReplayChunkError(f"Replay chunk index {index} already exists in {path.parent}")
    descriptor, temporary = _temporary_file(path)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with _compressed_writer(raw, path, reserve_gib=0 if final else None) as output:
                pickle.dump(transitions, output, protocol=pickle.HIGHEST_PROTOCOL)
            raw.flush()
            os.fsync(raw.fileno())
        _commit_new_file(temporary, path)
        return str(path)
    except Exception as exc:
        raise ReplayChunkError(f"Cannot save replay chunk {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _check_frame_header(path):
    with Path(path).open("rb") as stream:
        header = stream.read(19)
    info = lz4.frame.get_frame_info(header)
    if not info.get("content_checksum"):
        raise ReplayChunkError(f"Compressed replay chunk has no content checksum: {path}")


def load_replay_chunk(path):
    """Load exactly one complete pickle; verify the compressed footer as well."""
    path = Path(path)
    chunk_index(path)
    try:
        if path.name.endswith(".lz4"):
            _check_frame_header(path)
            stream = lz4.frame.open(path, mode="rb")
        else:
            stream = path.open("rb")
        with stream:
            transitions = pickle.load(stream)
            # pickle.load may stop before LZ4's footer. Reading to EOF catches
            # truncated frames and checksum failures after the pickle STOP.
            if stream.read(1):
                raise ReplayChunkError(f"Trailing data after persisted replay pickle: {path}")
        return transitions
    except Exception as exc:
        raise ReplayChunkError(f"Cannot load persisted replay chunk {path}: {exc}") from exc


def _stream_digest(stream, output=None):
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(_BLOCK_BYTES)
        if not block:
            break
        digest.update(block)
        size += len(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest(), size


def _raw_digest(path):
    with Path(path).open("rb") as stream:
        return _stream_digest(stream)


def _decoded_digest(path):
    _check_frame_header(path)
    with lz4.frame.open(path, mode="rb") as stream:
        return _stream_digest(stream)


def _regular_stat(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ReplayChunkError(f"Replay conversion requires a regular file, not a symlink: {path}")
    return value


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _preserve_metadata(path, original):
    os.chmod(path, stat.S_IMODE(original.st_mode))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def compress_existing_chunk(path, *, remove_source=False):
    """Stream-convert a legacy chunk without deserializing or changing its bytes.

    The default leaves the source for review. ``remove_source=True`` removes it
    only after a durable destination has passed full decoded SHA-256/byte-count
    verification. An existing destination must match the source exactly. Calling
    again after successful removal validates the compressed frame and is a no-op.
    A source/destination pair is deliberately rejected by replay_chunks until
    this conversion is finalized. The caller must exclude concurrent writers.
    """
    path = Path(path)
    index = chunk_index(path)
    if not path.name.endswith(".pkl") or path.parent.name not in ("buffer", "demo_buffer"):
        raise ReplayChunkError(f"Only legacy checkpoint buffer/demo_buffer chunks may be converted: {path}")
    target = path.with_name(path.name + ".lz4")
    source_exists = path.exists() or path.is_symlink()
    target_exists = target.exists() or target.is_symlink()
    source_stat = _regular_stat(path) if source_exists else None
    target_stat = _regular_stat(target) if target_exists else None
    if source_stat is None and target_stat is None:
        raise ReplayChunkError(f"Replay conversion source and destination are missing: {path}")
    before_bytes = (source_stat.st_size if source_stat else 0) + (target_stat.st_size if target_stat else 0)
    temporary = None
    try:
        source_hash = None
        if source_stat is not None:
            if target_exists:
                source_hash, source_bytes = _raw_digest(path)
                decoded_hash, decoded_bytes = _decoded_digest(target)
            else:
                require_space(target.parent)
                descriptor, temporary = _temporary_file(target)
                with os.fdopen(descriptor, "wb") as raw:
                    with _compressed_writer(raw, target) as output, path.open("rb") as original:
                        source_hash, source_bytes = _stream_digest(original, output)
                    raw.flush()
                    os.fsync(raw.fileno())
                decoded_hash, decoded_bytes = _decoded_digest(temporary)
            if (decoded_hash, decoded_bytes) != (source_hash, source_bytes):
                raise ReplayChunkError(f"Compressed destination differs from source; original retained: {target}")
            if _identity(_regular_stat(path)) != _identity(source_stat):
                raise ReplayChunkError(f"Replay source changed during compression; original retained: {path}")
            _preserve_metadata(temporary if temporary is not None else target, source_stat)
            if temporary is not None:
                _commit_new_file(temporary, target)
            else:
                _sync_directory(target.parent)
        else:
            decoded_hash, decoded_bytes = _decoded_digest(target)
            source_bytes = decoded_bytes
        compressed_hash, compressed_bytes = _raw_digest(target)
        saved_stat = target.stat()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            temporary = None
        removed = False
        if remove_source and source_stat is not None:
            if _identity(_regular_stat(path)) != _identity(source_stat):
                raise ReplayChunkError(f"Replay source changed before removal; original retained: {path}")
            # The destination and its directory have already been fsynced. If a
            # crash resurrects the old directory entry, re-running this function
            # verifies and finishes it; the durable compressed bytes remain.
            path.unlink()
            removed = True
        after_bytes = compressed_bytes + (source_bytes if source_stat is not None and not removed else 0)
        return {
            "source": str(path), "destination": str(target), "index": index,
            "source_sha256": source_hash, "decompressed_sha256": decoded_hash,
            "compressed_sha256": compressed_hash, "uncompressed_bytes": source_bytes,
            "compressed_bytes": compressed_bytes, "before_bytes": before_bytes,
            "after_bytes": after_bytes, "bytes_released": before_bytes - after_bytes,
            "mtime_ns": saved_stat.st_mtime_ns, "mode": oct(stat.S_IMODE(saved_stat.st_mode)),
            "source_removed": removed, "source_present": source_stat is not None and not removed,
            "reused_destination": target_exists,
        }
    except Exception as exc:
        if isinstance(exc, ReplayChunkError):
            raise
        raise ReplayChunkError(f"Cannot compress replay chunk {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
