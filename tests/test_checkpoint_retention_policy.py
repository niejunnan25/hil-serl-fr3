"""Retention plans and deletion exercise only synthetic temporary models."""
import copy
import json
from pathlib import Path

import pytest

from hilserl.checkpoint_retention import plan_latest, apply_latest_plan, retain_after_save


def model(root, step, *, directory=True, committed=True):
    path = root / f"checkpoint_{step}"
    if directory:
        path.mkdir()
        (path/"weights").write_bytes(bytes([step % 256]) * 9)
        if committed:
            (path/"_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": step+1}))
    else:
        path.write_bytes(bytes([step % 256]) * 11)
    return path


def test_numeric_latest_five_preserve_replay_and_ignore_uncommitted_or_lookalike_names(tmp_path):
    root=tmp_path.resolve()
    for step in (2,10,20,21,40,100,101):model(root,step,directory=step!=10)
    model(root,102,committed=False)
    for name in ("buffer","demo_buffer","checkpoint_tmp","checkpoint_backup","checkpoint_001"):
        path=root/name;path.mkdir();(path/"keep").write_bytes(b"original")
    plan=plan_latest(root,5)
    assert [e["step"] for e in plan["keep"]]==[20,21,40,100,101]
    assert [e["step"] for e in plan["remove"]]==[2,10]
    assert plan["ignored"]==["checkpoint_102"]
    result=apply_latest_plan(plan)
    assert result["removed"]==["checkpoint_2","checkpoint_10"]
    assert (root/"checkpoint_101/weights").read_bytes()==bytes([101])*9
    assert (root/"checkpoint_102/weights").exists()
    for name in ("buffer","demo_buffer","checkpoint_tmp","checkpoint_backup","checkpoint_001"):
        assert (root/name/"keep").read_bytes()==b"original"
    assert not plan_latest(root,5)["remove"]
    events=[json.loads(line)["event"] for line in (root/"retention.jsonl").read_text().splitlines()]
    assert events==["planned","deleting","deleted","deleting","deleted"]


@pytest.mark.parametrize("change", ["new_save","edited_weight","removed_keep"])
def test_inventory_drift_prevents_any_delete(tmp_path,change):
    root=tmp_path.resolve()
    for step in range(7):model(root,step)
    plan=plan_latest(root)
    if change=="new_save":model(root,8)
    elif change=="edited_weight":(root/"checkpoint_1/weights").write_bytes(b"changed")
    else:(root/"checkpoint_6/_CHECKPOINT_METADATA").unlink()
    with pytest.raises(RuntimeError,match="inventory changed"):
        apply_latest_plan(plan)
    assert (root/"checkpoint_0/weights").exists()


def test_file_or_directory_symlink_never_deletes_external_data(tmp_path):
    root=tmp_path.resolve();outside=root/"external";outside.mkdir();(outside/"protected").write_bytes(b"x")
    (root/"checkpoint_1").symlink_to(outside,target_is_directory=True)
    for step in range(2,9):model(root,step)
    apply_latest_plan(plan_latest(root))
    assert (outside/"protected").read_bytes()==b"x"
    (root/"checkpoint_8/linked").symlink_to(outside/"protected")
    with pytest.raises(ValueError,match="link"):
        plan_latest(root)


def test_disabled_policy_keeps_everything_and_enabled_requires_latest_committed_save(tmp_path,monkeypatch):
    root=tmp_path.resolve()
    for step in range(8):model(root,step)
    monkeypatch.delenv("HILSERL_CHECKPOINT_KEEP",raising=False)
    assert retain_after_save(root,7) is None
    assert len(list(root.glob("checkpoint_*")))==8
    monkeypatch.setenv("HILSERL_CHECKPOINT_KEEP","5")
    with pytest.raises(RuntimeError,match="latest"):
        retain_after_save(root,6)
    assert len(list(root.glob("checkpoint_*")))==8
    result=retain_after_save(root,7)
    assert result["removed"]==["checkpoint_0","checkpoint_1","checkpoint_2"]


@pytest.mark.parametrize("value", ["-1","0.5","true","5.0","05"])
def test_invalid_policy_never_deletes(tmp_path,monkeypatch,value):
    monkeypatch.setenv("HILSERL_CHECKPOINT_KEEP",value)
    with pytest.raises(ValueError):retain_after_save(tmp_path.resolve(),0)


def test_evaluation_ranking_preserves_latest_recovery_inside_five_slots(tmp_path):
    from hilserl.checkpoint_retention import plan_retention
    root=tmp_path.resolve()
    for step in range(8):model(root,step)
    plan=plan_retention(root,5,ranked_steps=[0,1,2,3,4,5])
    assert [entry["step"] for entry in plan["keep"]]==[0,1,2,3,7]
    assert plan["rule"]=="evaluation_then_latest"
    pinned=plan_retention(root,5,ranked_steps=[0,1,2,3,4],protected_steps=[6])
    assert [entry["step"] for entry in pinned["keep"]]==[0,1,2,6,7]


def test_reader_pin_survives_parent_close_and_exec_without_a_sixth_model(tmp_path):
    import os,subprocess,sys
    from hilserl.checkpoint_retention import open_reader_guard,locked_retention_plan
    root=tmp_path.resolve()
    for step in range(7):model(root,step)
    fd=open_reader_guard(root,0)
    child=subprocess.Popen([sys.executable,"-c",
        "import os,sys; os.fstat(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read(1)",str(fd)],
        pass_fds=(fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    os.close(fd)
    try:
        assert child.stdout.readline().strip()=="ready"
        with locked_retention_plan(root,5) as plan:
            assert plan["protected_steps"]==[0]
            result=apply_latest_plan(plan)
        assert len(result["kept"])==5 and "checkpoint_0" in result["kept"]
        assert (root/"checkpoint_0/weights").exists()
    finally:
        child.communicate(input="x",timeout=5)
    assert child.returncode==0
    with locked_retention_plan(root,5) as plan:
        assert plan["protected_steps"]==[]


def test_retention_error_does_not_claim_new_model_save_failed(tmp_path,monkeypatch):
    import hilserl.checkpoint_retention as retention
    root=tmp_path.resolve()
    for step in range(7):model(root,step)
    monkeypatch.setenv("HILSERL_CHECKPOINT_KEEP","5")
    monkeypatch.setattr(retention.shutil,"rmtree",lambda *_: (_ for _ in ()).throw(OSError("disk failure")))
    result=retention.maintenance_after_save(root,6)
    assert result["status"]=="error"
    assert retention.committed(root/"checkpoint_6")
    report=json.loads((root/"retention_status.json").read_text())
    assert report["status"]=="error" and report["step"]==6


def test_mpa_file_and_sidecar_are_preserved_as_unsupported_format(tmp_path):
    root=tmp_path.resolve()
    model(root,0,directory=False)
    (root/"checkpoint_0_gda").mkdir();(root/"checkpoint_0_gda/payload").write_bytes(b"array")
    for step in range(1,8):model(root,step)
    plan=plan_latest(root)
    assert "checkpoint_0" in plan["ignored"]
    apply_latest_plan(plan)
    assert (root/"checkpoint_0").exists() and (root/"checkpoint_0_gda/payload").read_bytes()==b"array"
