import json
from dataclasses import replace
from pathlib import Path

import pytest

from hilserl.config import Config
from hilserl.control import send_command
from hilserl.reward_provider import RewardSpec
from scripts.serve_robometer import model_manifest
from reward_fakes import SPEC


def configured(tmp_path):
    return Config(root=tmp_path, action_contract="fixed-xyz-v1", reward_mode="robometer-episode",
                  reward_model_id=SPEC.model_id)


def test_default_is_sparse_and_reward_environment_has_same_discount_and_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("HILSERL_REWARD_SPEC", "stale inherited setting")
    plain = Config(root=tmp_path)
    assert plain.reward_spec() is None
    assert "HILSERL_REWARD_SPEC" not in plain.environment("actor", run_dir=tmp_path / "run")
    cfg = replace(configured(tmp_path), discount=.998, reward_seed_cache="cache")
    cfg.validate()
    actor = cfg.environment("actor", run_dir=tmp_path / "run")
    learner = cfg.environment("learner", run_dir=tmp_path / "run")
    assert actor["HILSERL_REWARD_SPEC"] == learner["HILSERL_REWARD_SPEC"]
    spec = RewardSpec(**json.loads(actor["HILSERL_REWARD_SPEC"]))
    assert spec.gamma == float(actor["HILSERL_DISCOUNT"]) == .998
    assert Path(actor["HILSERL_REWARD_SEED_CACHE"]) == tmp_path / "cache"


@pytest.mark.parametrize("changes", [dict(reward_mode="unknown"), dict(reward_model_id="unversioned"),
    dict(reward_image_key="unknown"), dict(reward_max_frames=0), dict(reward_scale=float("nan")),
    dict(reward_url="not-a-url"), dict(action_contract="legacy-hybrid-7d-v1"), dict(reward_timeout_seconds=0)])
def test_bad_reward_configuration_fails_before_launch(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(configured(tmp_path), **changes).validate()


@pytest.mark.parametrize("phase", ["reward_pending", "waiting_reward", "draining_reward"])
def test_workbench_can_stop_during_every_reward_wait(tmp_path, phase):
    (tmp_path / "commands").mkdir()
    state = dict(attempt_id="actor", episode_id="ep", phase=phase, gate_id=None)
    (tmp_path / "status.json").write_text(json.dumps(state))
    assert send_command(tmp_path, "stop", expected=state)["command"] == "stop"
    with pytest.raises(ValueError):
        send_command(tmp_path, "continue", expected=state)


def test_model_identity_tracks_weight_and_config_contents(tmp_path):
    (tmp_path / "weights.safetensors").write_bytes(b"fixture weights, not a real model")
    (tmp_path / "config.json").write_text('{"version":1}')
    original = model_manifest(tmp_path)
    assert len(original["model_id"]) == 64
    (tmp_path / "config.json").write_text('{"version":2}')
    assert model_manifest(tmp_path)["model_id"] != original["model_id"]
