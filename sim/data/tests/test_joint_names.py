"""A5: 三个 sim 文件必须用 fr3_joint, 不用 panda_joint.

Source of truth: sim/scenes/plug_scene.py line 122/127 已经用
'fr3_joint[1-7]' / 'fr3_finger_joint.*' — 修复 gello_replay.py 对齐。
"""
import os
import pathlib

REPO = pathlib.Path(
    os.environ.get(
        "HILSERL_FR3_ROOT",
        pathlib.Path(__file__).resolve().parents[3],
    )
)

# A5 三个目标文件（字面替换范围）
TARGET_FILES = (
    "sim/data/gello_replay.py",
    "sim/scenes/plug_scene.py",
    "sim/data/plug_reward_labeler.py",
)

def test_no_panda_joint_in_gello_replay():
    p = REPO / "sim/data/gello_replay.py"
    text = p.read_text()
    assert "panda_joint" not in text, (
        f"gello_replay.py 仍含 panda_joint; A5 必须 sed 替换为 fr3_joint"
    )
    assert "panda_finger_joint" not in text, (
        f"gello_replay.py 仍含 panda_finger_joint; A5 必须替换为 fr3_finger_joint"
    )

def test_no_panda_joint_in_plug_scene():
    p = REPO / "sim/scenes/plug_scene.py"
    text = p.read_text()
    assert "panda_joint" not in text
    assert "panda_finger_joint" not in text

def test_no_panda_joint_in_plug_reward_labeler():
    p = REPO / "sim/data/plug_reward_labeler.py"
    text = p.read_text()
    assert "panda_joint" not in text
    assert "panda_finger_joint" not in text

def test_fr3_joint_present_in_gello_replay():
    """A5 修复后 gello_replay.py 必须含 fr3_joint 字符串."""
    p = REPO / "sim/data/gello_replay.py"
    text = p.read_text()
    assert "fr3_joint" in text, (
        "gello_replay.py 修复后应含 fr3_joint 字符串"
    )
    assert "fr3_finger_joint" in text, (
        "gello_replay.py 修复后应含 fr3_finger_joint 字符串"
    )

def test_fr3_joint_present_in_plug_scene():
    """plug_scene.py 已是 fr3_joint (line 122/127), 此 test 锁死 regression."""
    p = REPO / "sim/scenes/plug_scene.py"
    text = p.read_text()
    assert "fr3_joint" in text
    assert "fr3_finger_joint" in text
