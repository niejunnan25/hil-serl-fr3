"""Execute real NumPy replay insert/sample code without importing JAX or Flax."""
import ast
import collections
import copy
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import typing

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "upstream/hil-serl/serl_launcher/serl_launcher/data"


class FrozenMapping(dict):
    def unfreeze(self):
        return copy.deepcopy(dict(self))


@pytest.fixture
def replay_class():
    # Only dependency imports/container freezing are replaced. Dataset indexing,
    # allocation, ring insertion, continuity and random sampling are real code.
    namespace = dict(vars(typing), np=np, gym=gym, Box=gym.spaces.Box,
                     copy=copy, collections=collections, seeding=seeding, Lock=Lock,
                     DatasetDict=dict, DataType=object,
                     frozen_dict=SimpleNamespace(FrozenDict=FrozenMapping, freeze=FrozenMapping),
                     jax=SimpleNamespace(device_put=lambda value, *args: value))
    from abc import abstractmethod
    namespace["abstractmethod"] = abstractmethod
    paths = [(DATA / "dataset.py", None), (DATA / "replay_buffer.py", None),
             (DATA / "memory_efficient_replay_buffer.py", None),
             (ROOT / "upstream/agentlace/agentlace/data/data_store.py", {"DataStoreBase"}),
             (DATA / "data_store.py", {"MemoryEfficientReplayBufferDataStore"})]
    for path, names in paths:
        tree = ast.parse(path.read_text())
        body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and (names is None or node.name in names)]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["MemoryEfficientReplayBufferDataStore"]


def make_replay(replay_class, stack, capacity=256):
    observation_space = gym.spaces.Dict({
        "state": gym.spaces.Box(-1000, 1000, (1,), dtype=np.float32),
        **{key: gym.spaces.Box(0, 255, (stack, 2, 2, 3), dtype=np.uint8)
           for key in ("image", "wrist")},
    })
    replay = replay_class(observation_space, gym.spaces.Box(-1000, 1000, (1,), dtype=np.float32),
                          capacity=capacity, image_keys=("image", "wrist"))
    replay.seed(42)
    return replay


def transition(identifier, frame, stack, raw_id, *, done=False, derived=False):
    def observation(offset):
        return {"state": np.array([identifier + offset], np.float32),
                **{key: np.stack([np.full((2, 2, 3), (frame + i + offset + channel) % 256, np.uint8)
                                 for i in range(stack)])
                   for key, channel in (("image", 0), ("wrist", 30))}}
    return dict(observations=observation(0), next_observations=observation(1),
                actions=np.array([identifier], np.float32), rewards=np.float32(identifier / 100),
                dones=done, masks=np.float32(0 if done else 1),
                infos={"raw_transition_id": raw_id, "derived": derived})


def verify_samples(replay, expected, *, require_all=True):
    for packed in (False, True):
        samples = replay.sample(128, pack_obs_and_next_obs=packed)
        identifiers = samples["actions"][:, 0].astype(int)
        if require_all:
            assert set(identifiers) == set(expected)
        for index, identifier in enumerate(identifiers):
            original = expected[identifier]
            for key in ("image", "wrist"):
                if packed:
                    wanted = np.concatenate((original["observations"][key],
                                             original["next_observations"][key][-1:]), axis=0)
                    np.testing.assert_array_equal(samples["observations"][key][index], wanted)
                else:
                    np.testing.assert_array_equal(samples["observations"][key][index], original["observations"][key])
                    np.testing.assert_array_equal(samples["next_observations"][key][index], original["next_observations"][key])
            np.testing.assert_array_equal(samples["observations"]["state"][index], original["observations"]["state"])
            np.testing.assert_array_equal(samples["next_observations"]["state"][index], original["next_observations"]["state"])
            assert samples["rewards"][index] == original["rewards"]
            assert samples["dones"][index] == original["dones"]
            assert samples["masks"][index] == original["masks"]


@pytest.mark.parametrize("stack", [1, 3])
@pytest.mark.parametrize("case", ["continuous", "reset", "actor_restart", "sparse_intervention", "credit_replay", "unknown"])
def test_random_samples_keep_actual_images_across_all_collection_boundaries(replay_class, stack, case):
    replay = make_replay(replay_class, stack)
    if case == "continuous":
        description = [(0, "run/attempt-a/000001/000000"), (1, "run/attempt-a/000001/000001"),
                       (2, "run/attempt-a/000001/000002")]
    elif case == "reset":
        description = [(0, "run/attempt-a/000001/000000"), (1, "run/attempt-a/000001/000001"),
                       (90, "run/attempt-a/000002/000000")]
    elif case == "actor_restart":
        description = [(0, "run/attempt-a/000001/000000"), (90, "run/attempt-b/000001/000000")]
    elif case == "sparse_intervention":
        description = [(0, "run/attempt-a/000001/000000"), (8, "run/attempt-a/000001/000008"),
                       (12, "run/attempt-a/000001/000012")]
    elif case == "credit_replay":
        description = [(9, "run/attempt-a/000001/000009"), (9, "run/attempt-a/000001/000009"),
                       (3, "run/attempt-a/000001/000003"), (4, "run/attempt-a/000001/000004")]
    else:
        description = [(0, None), (90, None), (180, "bad-id")]
    expected = {i: transition(i, frame, stack, raw_id, derived=case == "credit_replay")
                for i, (frame, raw_id) in enumerate(description)}
    original = copy.deepcopy(expected)
    for sample in expected.values():
        replay.insert(sample)
    verify_samples(replay, expected)
    # Neither the input actions/targets nor their raw observations are mutated.
    for i in expected:
        for key in ("rewards", "dones", "masks"):
            assert expected[i][key] == original[i][key]
        for group in ("observations", "next_observations"):
            for key in ("image", "wrist"):
                np.testing.assert_array_equal(expected[i][group][key], original[i][group][key])


@pytest.mark.parametrize("stack", [1, 3])
@pytest.mark.parametrize("case", ["continuous", "independent", "mixed"])
def test_random_samples_preserve_images_when_ring_storage_wraps(replay_class, stack, case):
    replay = make_replay(replay_class, stack, capacity=17)
    expected = {}
    for i in range(40):
        frame = i if case == "continuous" else i * 3
        raw_id = f"run/attempt/episode/{i:06d}" if case == "continuous" else None
        if case == "mixed" and i % 4 in (0, 1):
            raw_id = f"run/attempt/episode-{i // 4}/{i % 4:06d}"
            frame = (i // 4) * 10 + i % 4
        sample = transition(i, frame, stack, raw_id)
        expected[i] = sample
        replay.insert(sample)
        # Check after every insertion so an invalid wrap slot cannot disappear
        # behind later writes before the regression inspects it.
        verify_samples(replay, expected, require_all=False)


def test_explicit_done_is_preserved_even_with_consecutive_raw_ids(replay_class):
    replay = make_replay(replay_class, 1)
    expected = {0: transition(0, 0, 1, "run/attempt/episode/0", done=True),
                1: transition(1, 90, 1, "run/attempt/episode/1")}
    for item in expected.values():
        replay.insert(item)
    verify_samples(replay, expected)
