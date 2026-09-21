from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stale_training_launcher_refuses_even_with_old_ack():
    result = subprocess.run(["bash", str(ROOT/"scripts/run_actor.sh")],
                            env={**os.environ,"RUN_ACTOR_LEGACY_ACK":"1"},capture_output=True,text=True)
    assert result.returncode == 64
    assert "bin/hil-serl train" in result.stderr


@pytest.mark.parametrize("name,mode", [
    ("run_actor_vice.sh",["train"]),("run_actor_phaseC.sh",["train"]),
    ("run_actor_phaseC_serl_aligned.sh",["train"]),
    ("run_train_vice_5080.sh",["train","--learner-only"]),
    ("run_learner_phaseC_local5080.sh",["train","--learner-only"]),
    ("run_learner_phaseC_serl_aligned.sh",["train","--learner-only"]),
    ("run_learner.sh",["train","--learner-only"]),
    ("run_insert_demo_collection_serl_aligned.sh",["collect"]),
    ("demo_collect_xbox.sh",["collect"]),("deploy_policy.sh",["eval"]),
])
def test_legacy_entrypoints_forward_to_one_runtime_without_running_it(tmp_path,name,mode):
    (tmp_path/"scripts").mkdir();(tmp_path/"bin").mkdir()
    wrapper=tmp_path/"scripts"/name
    shutil.copyfile(ROOT/"scripts"/name,wrapper)
    target=tmp_path/"bin/hil-serl"
    target.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    target.chmod(0o755)
    result=subprocess.run(["bash",str(wrapper),"--help"],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout) == mode+["--help"]
