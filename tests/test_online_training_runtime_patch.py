"""Regression checks for the FR3 online-training runtime patch.

The complete actor runtime lives on the FR3 desktop.  These tests inspect the
staged copy that is deployed to `/home/robot/serl_projects/hil-serl-fr3`.
They are intentionally no-motion and do not import ROS or contact the robot.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = (
    ROOT
    / ".planning/2026-06-13-v221-hybrid-teleop/runtime-patches"
    / "20260617-online-stability"
)


def _read(relative: str) -> str:
    path = PATCH_ROOT / relative
    assert path.exists(), f"missing staged runtime file: {path}"
    return path.read_text()


def _load_actor_relz_helpers():
    source = _read("_run_actor.py")
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "SERL_FLAT_RELZ_STATE_INDEX" in names:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"_as_array", "_extract_relz"}:
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, "<actor-relz-helpers>", "exec"), namespace)
    return namespace["_extract_relz"]


def _load_actor_success_credit_helpers():
    source = _read("../20260617-manual-success-credit/_run_actor.py")
    tree = ast.parse(source)
    helper_names = {
        "_env_bool",
        "_success_credit_horizon",
        "_success_credit_min_action_norm",
        "_transition_action_norm",
        "_remember_success_credit_transition",
        "_make_success_credit_transition",
        "_insert_success_credit_tail",
    }
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"copy": copy, "np": np, "os": os}
    exec(compile(module, "<actor-success-credit-helpers>", "exec"), namespace)
    return namespace


def test_franka_pose_command_does_not_clearerr_each_step():
    source = _read("upstream/hil-serl/serl_robot_infra/franka_env/envs/franka_env.py")
    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    send_pos = methods["_send_pos_command"]
    guarded_recover_calls = [
        node
        for node in send_pos.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "clearerr_on_pose"
        and any(
            isinstance(child, ast.Attribute) and child.attr == "_recover"
            for child in ast.walk(node)
        )
    ]

    assert guarded_recover_calls
    assert "FRANKA_CLEARERR_ON_POSE" in source
    assert "FRANKA_SAFETY_DQ_MAX" in source
    assert "[franka_safety_fatal]" in source


def test_reset_motion_is_strict_and_clear_move_has_enough_time():
    source = _read("experiments/plug_insertion/env.py")

    assert "RESET_STRICT" in source
    assert "raise RuntimeError" in source
    assert 'name="clear"' in source
    assert "timeout=8.0" in source
    assert "settle_deadline" in source
    assert "[reset_settle_wait]" in source
    assert "return pos_err, rot_err" in source


def test_xbox_intervention_logs_left_stick_final_action_and_uses_tuned_steps():
    source = _read("scripts/xbox_intervention.py")

    assert "XBOX_MAX_STEP = 0.008" in source
    assert "XBOX_INSERT_REACH = 0.004" in source
    assert "XBOX_ROT_STEP = 0.030" in source
    assert "lx=%+.2f ly=%+.2f" in source
    assert "act_xyz=" in source
    assert "source=rb" in source


def test_actor_run_script_uses_tuned_but_bounded_action_caps():
    run_script = _read("scripts/run_actor_phaseC.sh")
    franka_env = _read("upstream/hil-serl/serl_robot_infra/franka_env/envs/franka_env.py")

    assert 'FRANKA_ACTION_MAX_STEP:-0.0050' in run_script
    assert 'FRANKA_ACTION_MAX_ROT_STEP:-0.045' in run_script
    assert "_env_float(\"FRANKA_ACTION_MAX_STEP\", 0.0050)" in franka_env
    assert "_env_float(\"FRANKA_ACTION_MAX_ROT_STEP\", 0.045)" in franka_env


def test_actor_has_hard_safety_stop_and_obs_sanity_check():
    source = _read("_run_actor.py")

    assert "ACTOR_SAFETY_DQ_MAX" in source
    assert "ACTOR_SAFETY_FORCE_MAX" in source
    assert "ACTOR_RELZ_ABS_MAX" in source
    assert "_check_runtime_safety" in source
    assert "_extract_relz" in source
    assert "SERLObsWrapper may flatten" in source
    assert "only trust explicit rel-z keys" in source
    assert "SERL_FLAT_RELZ_STATE_INDEX = 6" in source
    assert "arr.size == 19" in source
    assert "tcp_pose[2] is world z" in source
    assert "RuntimeError" in source


def test_actor_relz_helper_reads_known_serl_flat_state_and_ignores_unknown_flat_shapes():
    extract_relz = _load_actor_relz_helpers()

    flat_state = np.arange(19, dtype=float)
    flat_state[2] = -2.8089
    flat_state[6] = -0.061

    assert extract_relz({"state": flat_state}) == -0.061
    assert extract_relz({"state": flat_state.reshape(1, 19)}) == -0.061
    assert extract_relz({"state": np.arange(25, dtype=float)}) is None
    assert extract_relz({"state": {"tcp_pose": [0.0, 0.0, -2.8, 0.0, 0.0, 0.0]}}) is None
    assert extract_relz({"state": flat_state, "reset_rel_z": -0.08}) == -0.08


def test_reward_reads_relative_z_from_flattened_tcp_pose_slot_not_force_slot():
    source = _read("experiments/plug_insertion/config.py")

    assert "REWARD_RELZ_STATE_INDEX = 6" in source
    assert "gymnasium.spaces.Dict sorts keys" in source
    assert "obs[\"state\"][0, 6]" in source
    assert "jnp.reshape(obs[\"state\"], (-1,))[2]" not in source


def test_manual_success_credit_backfills_recent_nonzero_human_actions(monkeypatch):
    helpers = _load_actor_success_credit_helpers()
    monkeypatch.setenv("ACTOR_SUCCESS_CREDIT_HORIZON", "3")
    monkeypatch.setenv("ACTOR_SUCCESS_CREDIT_MIN_ACTION_NORM", "0.05")
    monkeypatch.delenv("ACTOR_SUCCESS_CREDIT_ONLINE", raising=False)

    tail = []
    zero_human = {
        "observations": {"state": np.zeros(19, dtype=np.float32)},
        "actions": np.zeros(7, dtype=np.float32),
        "next_observations": {"state": np.ones(19, dtype=np.float32)},
        "rewards": 0.0,
        "masks": 1.0,
        "dones": False,
    }
    policy_action = {**zero_human, "actions": np.full(7, 0.5, dtype=np.float32)}
    human_a = {**zero_human, "actions": np.array([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)}
    human_b = {**zero_human, "actions": np.array([0, 0.3, 0, 0, 0, 0, 0], dtype=np.float32)}

    helpers["_remember_success_credit_transition"](tail, zero_human, is_human=True)
    helpers["_remember_success_credit_transition"](tail, policy_action, is_human=False)
    helpers["_remember_success_credit_transition"](tail, human_a, is_human=True)
    helpers["_remember_success_credit_transition"](tail, human_b, is_human=True)

    assert len(tail) == 2

    class Store:
        def __init__(self):
            self.items = []

        def insert(self, item):
            self.items.append(copy.deepcopy(item))

    intvn_store = Store()
    online_store = Store()
    demo_transitions = []
    online_transitions = []

    added = helpers["_insert_success_credit_tail"](
        tail,
        intvn_store,
        demo_transitions,
        data_store=online_store,
        transitions=online_transitions,
    )

    assert added == 2
    assert len(intvn_store.items) == 2
    assert len(demo_transitions) == 2
    assert online_store.items == []
    assert online_transitions == []
    for item in demo_transitions:
        assert item["rewards"] == 1.0
        assert item["dones"] is False
        assert item["masks"] == 1.0
        assert item["infos"]["manual_success_credit"] is True
        assert item["infos"]["source_action"] == "human"


def test_manual_success_credit_can_optionally_duplicate_to_online_buffer(monkeypatch):
    helpers = _load_actor_success_credit_helpers()
    monkeypatch.setenv("ACTOR_SUCCESS_CREDIT_ONLINE", "1")

    transition = {
        "observations": {"state": np.zeros(19, dtype=np.float32)},
        "actions": np.array([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "next_observations": {"state": np.ones(19, dtype=np.float32)},
        "rewards": 0.0,
        "masks": 1.0,
        "dones": False,
    }

    class Store:
        def __init__(self):
            self.items = []

        def insert(self, item):
            self.items.append(copy.deepcopy(item))

    intvn_store = Store()
    online_store = Store()
    demo_transitions = []
    online_transitions = []

    added = helpers["_insert_success_credit_tail"](
        [transition],
        intvn_store,
        demo_transitions,
        data_store=online_store,
        transitions=online_transitions,
    )

    assert added == 1
    assert len(intvn_store.items) == 1
    assert len(online_store.items) == 1
    assert len(online_transitions) == 1
    assert online_store.items[0]["rewards"] == 1.0


def test_actor_loop_forces_manual_success_terminal_into_demo_and_logs_backfill():
    source = _read("../20260617-manual-success-credit/_run_actor.py")

    assert "success_credit_tail = []" in source
    assert "force_success_demo = bool(done and reward and info.get(\"manual_success\"))" in source
    assert "if already_intervened or force_success_demo:" in source
    assert "_remember_success_credit_transition(" in source
    assert "_insert_success_credit_tail(" in source
    assert "[manual_success_credit] added" in source
    assert "\"manual_success\": bool(info.get(\"manual_success\"))" in source
    assert "\"source_action\": info.get(\"source_action\")" in source
    assert "\"step\": int(step)" in source
