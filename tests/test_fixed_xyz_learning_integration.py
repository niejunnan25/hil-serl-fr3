"""Exercise contract admission, the production Learner loop and model selection."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hilserl.config import Config
from hilserl.files import atomic_json
from hilserl.learning_replay import ContractReplayStore
from hilserl.processes import Manager
from test_learner_runtime_shutdown import Runtime


def transition(index=0):
    obs = {"state": np.zeros((1, 19), np.float32),
           "wrist_1": np.zeros((1, 128, 128, 3), np.uint8),
           "side_policy": np.zeros((1, 128, 128, 3), np.uint8)}
    return dict(observations=obs, next_observations=deepcopy(obs), actions=np.zeros(3, np.float32),
                rewards=0., dones=False, masks=1.,
                infos=dict(action_contract="fixed-xyz-v1", raw_transition_id=f"run/attempt/episode/{index:06d}"))


@pytest.mark.parametrize("defect", ["dimension", "nonfinite", "positive_nonterminal", "bootstrap_terminal",
                                    "no_human_verdict", "credit", "penalty", "missing_id", "image", "frame_shape"])
def test_bad_replay_is_rejected_before_native_store_mutation(defect):
    data = transition()
    if defect == "dimension": data["actions"] = np.zeros(7, np.float32)
    elif defect == "nonfinite": data["actions"][0] = np.nan
    elif defect == "positive_nonterminal": data["rewards"] = 1
    elif defect == "bootstrap_terminal": data["dones"] = True
    elif defect == "no_human_verdict": data.update(dones=True, masks=0)
    elif defect == "credit": data["infos"]["manual_success_credit"] = True
    elif defect == "penalty": data["grasp_penalty"] = -.02
    elif defect == "missing_id": data["infos"].pop("raw_transition_id")
    elif defect == "image": data["observations"]["wrist_1"] = np.zeros((128, 128, 3), np.uint8)
    else: data["observations"]["state"] = np.zeros((1, 25), np.float32)
    native = Mock()
    store = ContractReplayStore(native)
    with pytest.raises(ValueError): store.insert(data)
    native.insert.assert_not_called()
    assert store.transition_count == 0


def test_count_excludes_image_support_slots_and_duplicate_transport_payloads():
    native = Mock()
    store = ContractReplayStore(native)
    store.insert(transition())
    with pytest.raises(ValueError, match="Duplicate"):
        store.insert(transition())
    assert store.transition_count == native.insert.call_count == 1
    native.insert.side_effect = OSError("failed insert")
    with pytest.raises(OSError): store.insert(transition(1))
    assert store.transition_count == 1


def fixed_runtime(tmp_path, monkeypatch):
    # This suite fakes optimizer state; actual JAX trees are covered separately
    # by test_model_health and the native full-model update/restore smoke.
    import sys
    monkeypatch.setitem(sys.modules, "hilserl.model_health",
                        SimpleNamespace(require_finite_update=lambda *_: None))
    runtime = Runtime(tmp_path, monkeypatch)
    runtime.config.action_contract = "fixed-xyz-v1"
    runtime.config.training_starts = 100
    runtime.replay.transition_count = 0
    runtime.demo.transition_count = 1004
    runtime.replay_size = 2008
    return runtime


def test_fixed_runtime_waits_for_100_real_online_transitions(tmp_path, monkeypatch):
    runtime = fixed_runtime(tmp_path, monkeypatch)
    runtime.sleep_limit = 5
    def arriving():
        assert runtime.update_count == 0
        runtime.replay.transition_count = 99 if runtime.sleep_count == 1 else 100
    runtime.on_sleep = arriving
    runtime.on_update = lambda _: runtime.stop_file()
    runtime.run()
    assert runtime.sleep_count == 2 and runtime.update_count == 1
    files = list((runtime.control.run_dir / "metrics").glob("*.jsonl"))
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    final = records[-1]
    assert final["update_groups"] == final["critic_updates"] == 1
    assert final["unique_online_count"] == 100
    assert final["critic_utd"] == .01


def test_restored_online_data_does_not_bypass_actor_gate(tmp_path, monkeypatch):
    runtime = fixed_runtime(tmp_path, monkeypatch)
    runtime.replay.transition_count = 1500
    runtime.actor_alive = False
    runtime.on_sleep = runtime.stop_file
    runtime.run()
    assert runtime.update_count == 0
    assert runtime.status()["pause_code"] == "actor_not_alive"


def test_model_health_checks_unsampled_groups_before_committing(tmp_path, monkeypatch):
    import sys
    runtime = fixed_runtime(tmp_path, monkeypatch)
    runtime.replay.transition_count = 1500
    checks = []
    def check(state, _info):
        checks.append(state.params["completed_updates"])
        if len(checks) == 2:
            raise FloatingPointError("nonfinite model before commit")
    sys.modules["hilserl.model_health"].require_finite_update = check
    with pytest.raises(FloatingPointError): runtime.run()
    assert checks == [1, 2]
    assert runtime.status()["step"] == 0
    assert runtime.status()["phase"] == "fault"
    assert runtime.status()["checkpoint_saved"]


def test_metric_failure_preserves_last_committed_state_and_fault(tmp_path, monkeypatch):
    runtime = fixed_runtime(tmp_path, monkeypatch)
    runtime.replay.transition_count = 1500
    from hilserl.training_metrics import LearnerMetrics
    original = LearnerMetrics.write
    def invalid(self, **kwargs):
        kwargs["update_info"] = {"critic_loss": float("nan")}
        return original(self, **kwargs)
    monkeypatch.setattr(LearnerMetrics, "write", invalid)
    with pytest.raises(FloatingPointError): runtime.run()
    assert runtime.status()["phase"] == "fault"
    assert runtime.status()["step"] == -1
    assert not runtime.status().get("checkpoint_saved")


def test_unversioned_snapshot_resumes_and_evaluates_as_legacy(tmp_path):
    cfg = replace(Config(), root=tmp_path, data_dir="runs", action_contract="fixed-xyz-v1")
    manager = Manager(cfg)
    run = cfg.output_root / "old"
    run.mkdir(parents=True)
    snapshot = replace(cfg, action_contract="legacy-hybrid-7d-v1").snapshot()
    snapshot.pop("action_contract")
    snapshot.pop("seed_dataset_sha256")
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    atomic_json(run / "config.json", snapshot)
    atomic_json(run / "run.json", dict(mode="train", checkpoint=str(run/"checkpoints"), config_sha256=digest))
    checkpoint = run / "checkpoints/checkpoint_17"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    restored = manager._resume_details(checkpoint)[-1]
    evaluated = manager._evaluation_config(checkpoint.parent)
    assert restored.action_contract == evaluated.action_contract == "legacy-hybrid-7d-v1"
    assert restored.environment("learner", run_dir=run)["ACTOR_SUCCESS_CREDIT_HORIZON"] == "24"
    assert cfg.environment("learner", run_dir=run)["ACTOR_SUCCESS_CREDIT_HORIZON"] == "0"
    assert "action_contract" not in json.loads((run/"config.json").read_text())


def test_evaluation_launch_uses_selected_model_config_before_preflight(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path, data_dir="runs", min_free_gib=0, action_contract="fixed-xyz-v1")
    manager = Manager(cfg)
    legacy = replace(cfg, action_contract="legacy-hybrid-7d-v1", batch_size=128)
    run = manager._new_run("train", config=legacy)
    checkpoint = run / "checkpoints/checkpoint_20"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"selected")
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [])
    monkeypatch.setattr("hilserl.processes.process_start", lambda *_: "owner")
    manager.preflight = Mock(return_value={"ok": True})
    manager._spawn = Mock(return_value=SimpleNamespace(pid=123))
    manager._wait_actor_ready = Mock()
    manager.launch("eval", checkpoint=checkpoint)
    assert manager.preflight.call_args.kwargs["config"] == legacy
    assert manager._spawn.call_args.args[3] == legacy
    assert manager._spawn.call_args.kwargs["eval_step"] == 20


def test_model_source_changes_affect_process_fingerprint(tmp_path):
    manager = Manager(replace(Config(), root=tmp_path))
    directory = tmp_path / "upstream/hil-serl/serl_launcher/serl_launcher"
    for relative in ("agents/continuous/sac.py", "networks/actor_critic_nets.py", "vision/resnet_v1.py"):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
        before = manager._source_hashes()
        path.write_text("changed")
        assert before != manager._source_hashes()


def test_fixed_preflight_reports_validated_seed_counts(tmp_path, monkeypatch):
    from pathlib import Path
    import sys
    cfg = replace(Config(), root=tmp_path, min_free_gib=0, python=sys.executable,
                  action_contract="fixed-xyz-v1", seed_dataset_sha256="a"*64)
    (tmp_path/"_run_actor.py").touch()
    (tmp_path/"classifier_ckpt/checkpoint_1").mkdir(parents=True)
    monkeypatch.setattr("hilserl.seed_dataset.validate_seed_dataset", lambda *a, **kw:
                        {"counts": {"episodes": 13, "transitions": 1004}})
    monkeypatch.setattr("hilserl.processes.shutil.which", lambda _: "/test/ffmpeg")
    result = Manager(cfg).preflight()
    assert result["ok"]
    assert result["demo_count"] == 13 and result["demo_transition_count"] == 1004
