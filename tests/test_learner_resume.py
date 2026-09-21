"""Execute real continuation helpers with a small serialized CPU state backend."""
import ast
import copy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def flatten(value):
    if isinstance(value, SimpleNamespace):
        value = vars(value)
    if isinstance(value, dict):
        parts = [(key, flatten(value[key])) for key in sorted(value)]
        return [leaf for _, (leaves, _) in parts for leaf in leaves], tuple((key, tree) for key, (_, tree) in parts)
    return [value], "leaf"


@dataclass(frozen=True)
class Agent:
    state: object

    def replace(self, **changes):
        return replace(self, **changes)


@pytest.fixture
def helpers():
    calls = []

    def restore(directory, target, *, step):
        calls.append((directory, target, step))
        with (Path(directory) / f"checkpoint_{step}" / "state.pkl").open("rb") as stream:
            return pickle.load(stream)

    namespace = dict(os=os, pkl=pickle, np=np,
                     jax=SimpleNamespace(tree_util=SimpleNamespace(tree_flatten=flatten)),
                     checkpoints=SimpleNamespace(restore_checkpoint=restore))
    tree = ast.parse((ROOT / "_run_actor.py").read_text())
    names = {"_validate_training_directory", "_restore_training_checkpoint", "_reload_online_buffers", "_next_dump_step"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == names
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / "_run_actor.py"), "exec"), namespace)
    return SimpleNamespace(**namespace, calls=calls)


def state(value):
    return SimpleNamespace(params={"policy": np.full((2, 3), value, np.float32)},
                           target_params={"policy": np.full((2, 3), value + 1, np.float32)},
                           opt_states={"m": np.full((2, 3), value + 2, np.float32),
                                       "v": np.full((2, 3), value + 3, np.float32)},
                           rng=np.array([value, value + 4], np.uint32), step=np.array(value, np.int32))


def checkpoint(directory, step, value=None, *, committed=True):
    path = directory / f"checkpoint_{step}"
    path.mkdir()
    (path / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1} if committed else {}))
    if value is not None:
        (path / "state.pkl").write_bytes(pickle.dumps(value))
    return path


def test_explicit_resume_restores_optimizer_targets_rng_and_step(helpers, tmp_path):
    original = Agent(state(0))
    saved = state(25974)
    path = checkpoint(tmp_path, 25974, saved)
    before = (path / "state.pkl").read_bytes()
    restored = helpers._restore_training_checkpoint(original, tmp_path, 25974)
    assert restored is not original
    for expected, actual in zip(flatten(saved)[0], flatten(restored.state)[0]):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(original.state.params["policy"], np.zeros((2, 3)))
    assert helpers.calls[0][1] is original.state
    assert helpers.calls[0][2] == 25974
    assert (path / "state.pkl").read_bytes() == before


@pytest.mark.parametrize("change", ["shape", "dtype", "optimizer_missing", "state_missing"])
def test_incompatible_full_state_fails_before_training(helpers, tmp_path, change):
    saved = state(20)
    if change == "shape":
        saved.params["policy"] = np.zeros((7, 3), np.float32)
    elif change == "dtype":
        saved.opt_states["m"] = np.zeros((2, 3), np.float64)
    elif change == "optimizer_missing":
        del saved.opt_states["v"]
    else:
        del saved.rng
    checkpoint(tmp_path, 20, saved)
    with pytest.raises(RuntimeError, match="Cannot resume full training state"):
        helpers._restore_training_checkpoint(Agent(state(0)), tmp_path, 20)


def test_fresh_training_never_silently_resumes(helpers, tmp_path):
    original = Agent(state(0))
    assert helpers._restore_training_checkpoint(original, tmp_path, -1) is original
    checkpoint(tmp_path, 0, state(1))
    with pytest.raises(RuntimeError, match="Fresh training"):
        helpers._restore_training_checkpoint(original, tmp_path, -1)
    assert helpers.calls == []


@pytest.mark.parametrize("condition", ["missing", "uncommitted", "later_checkpoint"])
def test_resume_requires_committed_latest_step_without_overwrite(helpers, tmp_path, condition):
    if condition != "missing":
        checkpoint(tmp_path, 10, state(10), committed=condition != "uncommitted")
    if condition == "later_checkpoint":
        checkpoint(tmp_path, 20, state(20))
    with pytest.raises(RuntimeError):
        helpers._restore_training_checkpoint(Agent(state(0)), tmp_path, 10)
    assert helpers.calls == []


class Buffer:
    def __init__(self):
        self.items = [{"source": "configured_demo"}]

    def insert(self, item):
        self.items.append(copy.deepcopy(item))


def test_restarted_actor_low_numeric_chunks_follow_old_chunks_in_saved_time_order(helpers, tmp_path):
    for name in ("buffer", "demo_buffer"):
        (tmp_path / name).mkdir()
    for index in (10, 2, 3):
        target = tmp_path / "buffer" / f"transitions_{index}.pkl"
        target.write_bytes(pickle.dumps([{"source": index}]))
        saved_ns = (100, 200, 300)[(10, 2, 3).index(index)]
        os.utime(target, ns=(saved_ns, saved_ns))
    (tmp_path / "demo_buffer" / "transitions_7.pkl").write_bytes(pickle.dumps([{"source": "human", "infos": {"grasp_penalty": -1}}]))
    (tmp_path / "buffer" / "transitions_11.pkl.tmp").write_bytes(b"uncommitted")
    online, intervention = Buffer(), Buffer()
    counts = helpers._reload_online_buffers(tmp_path, online, intervention)
    assert counts == {"buffer": 3, "demo_buffer": 1}
    assert [item["source"] for item in online.items] == ["configured_demo", 10, 2, 3]
    assert intervention.items[0] == {"source": "configured_demo"}
    assert intervention.items[1]["grasp_penalty"] == -1
    assert all(item["grasp_penalty"] == 0 for item in online.items[1:])


def test_new_actor_chunk_index_follows_all_old_attempts_without_overwrite(helpers, tmp_path):
    (tmp_path / "transitions_155.pkl").write_bytes(b"saved actor one")
    (tmp_path / "transitions_3.pkl").write_bytes(b"legacy restarted actor")
    assert helpers._next_dump_step(tmp_path, 0) == 156
    assert helpers._next_dump_step(tmp_path, 300) == 300
    assert (tmp_path / "transitions_155.pkl").read_bytes() == b"saved actor one"


def test_corrupt_replay_chunk_fails_instead_of_silent_partial_resume(helpers, tmp_path):
    (tmp_path / "buffer").mkdir()
    (tmp_path / "buffer" / "transitions_0.pkl").write_bytes(pickle.dumps({"not": "transitions"}))
    with pytest.raises(RuntimeError, match="transition list"):
        helpers._reload_online_buffers(tmp_path, Buffer(), Buffer())


def test_replay_reload_observes_stop_between_transitions(helpers, tmp_path):
    (tmp_path / "buffer").mkdir()
    (tmp_path / "buffer" / "transitions_0.pkl").write_bytes(pickle.dumps([{}, {}, {}]))
    checks = []

    def stop():
        checks.append(1)
        if len(checks) == 3:
            raise RuntimeError("stop during reload")

    online = Buffer()
    with pytest.raises(RuntimeError, match="stop during reload"):
        helpers._reload_online_buffers(tmp_path, online, Buffer(), check_stop=stop)
    assert len(online.items) == 2
