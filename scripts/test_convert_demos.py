"""TDD tests for convert_demos_to_serl19.py

Spec is empirically locked from RunSetup + franka_env source:
  TARGET observation_space (live wrapped env):
    state:           Box (1, 19) float32
    side_policy:     Box (1, 128, 128, 3) uint8
    wrist_1:         Box (1, 128, 128, 3) uint8
    side_classifier: Box (1, 128, 128, 3) uint8
  action_space: Box(-1,1,(7,),float32)

  DEMO source obs per transition:
    state:           (25,) f32  = pos[0:3], quat[3:7] (w,x,y,z scalar-first),
                                   vel[7:13], force[13:16], torque[16:19],
                                   gripper[19:25] tiled 6x
    images:          (3,128,128) u8 CHW  (side_policy / wrist_1 / side_classifier)
"""
import os
import glob
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import convert_demos_to_serl19 as C

# ---- locked authoritative target spec (from RunSetup) ----
STATE_SHAPE = (1, 19)
IMG_SHAPE = (1, 128, 128, 3)
IMG_KEYS = ["side_policy", "wrist_1", "side_classifier"]

DEMO_GLOB = "/home/robot/hilserl-fr3/demos/hybrid/*success*.pkl"


@pytest.fixture(scope="module")
def one_success_demo_path():
    paths = sorted(glob.glob(DEMO_GLOB))
    assert paths, f"no demos found at {DEMO_GLOB}"
    return paths[0]


@pytest.fixture(scope="module")
def raw_demo(one_success_demo_path):
    import pickle
    with open(one_success_demo_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def converted(raw_demo):
    # is_success=True because it's a *success*.pkl
    return C.convert_demo(raw_demo, is_success=True)


# ---------------------------------------------------------------------------
# T1: converted obs leaf shapes/dtypes EQUAL the RunSetup observation_space
# ---------------------------------------------------------------------------
def test_T1_obs_leaf_shapes_dtypes(converted):
    assert len(converted) > 0
    for tr in (converted[0], converted[len(converted) // 2], converted[-1]):
        for which in ("observations", "next_observations"):
            o = tr[which]
            assert set(o.keys()) == {"state", *IMG_KEYS}, o.keys()
            assert o["state"].shape == STATE_SHAPE, o["state"].shape
            assert o["state"].dtype == np.float32, o["state"].dtype
            for k in IMG_KEYS:
                assert o[k].shape == IMG_SHAPE, (k, o[k].shape)
                assert o[k].dtype == np.uint8, (k, o[k].dtype)
        a = tr["actions"]
        assert a.shape == (7,)
        assert a.dtype == np.float32


# ---------------------------------------------------------------------------
# T2: at the FIRST/reset transition, relative tcp_pose pos == [0,0,0]
#     and orientation euler ~ [0,0,0]
# ---------------------------------------------------------------------------
def test_T2_reset_pose_is_identity(converted):
    state0 = converted[0]["observations"]["state"][0]  # (19,)
    tcp_pose = state0[:6]  # pos(3) + euler(3)
    np.testing.assert_allclose(tcp_pose[:3], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(tcp_pose[3:6], [0, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# T3: a known quaternion -> expected euler (matches Quat2EulerWrapper:
#     R.from_quat(xyzw).as_euler('xyz'))
# ---------------------------------------------------------------------------
def test_T3_known_quat_to_euler():
    # 90 deg about z: scalar-last xyzw = [0,0,sin45,cos45]
    q_xyzw = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    euler = C.quat_xyzw_to_euler(q_xyzw)
    np.testing.assert_allclose(euler, [0, 0, np.pi / 2], atol=1e-6)
    # and the scalar-first reorder helper must map wxyz->xyzw correctly
    q_wxyz = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    np.testing.assert_allclose(C.quat_wxyz_to_xyzw(q_wxyz), q_xyzw, atol=1e-12)


# ---------------------------------------------------------------------------
# T4: action reproject is norm-preserving (pre-clip):
#     ||a_new[:3]|| == ||a_base[:3]||, ||a_new[3:6]|| == ||a_base[3:6]||,
#     gripper unchanged.
# ---------------------------------------------------------------------------
def test_T4_action_reproject_norm_preserving():
    rng = np.random.default_rng(0)
    a_base = rng.uniform(-0.5, 0.5, size=7).astype(np.float32)
    # arbitrary current pose quat (scalar-last)
    q_xyzw = R.from_euler("xyz", [0.3, -0.7, 1.1]).as_quat()
    a_new = C.reproject_action(a_base, q_xyzw)
    np.testing.assert_allclose(np.linalg.norm(a_new[:3]),
                               np.linalg.norm(a_base[:3]), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(a_new[3:6]),
                               np.linalg.norm(a_base[3:6]), rtol=1e-6)
    assert a_new[6] == a_base[6]


def test_T4b_action_clipped_in_converter(converted):
    for tr in converted:
        a = tr["actions"]
        assert np.all(a[:6] >= -1.0) and np.all(a[:6] <= 1.0)


# ---------------------------------------------------------------------------
# T5: de-tile + CHW->HWC produce correct shapes (and values transposed right)
# ---------------------------------------------------------------------------
def test_T5_image_transpose_and_chunk():
    chw = np.arange(3 * 128 * 128, dtype=np.uint8).reshape(3, 128, 128)
    out = C.chw_to_chunked_hwc(chw)
    assert out.shape == IMG_SHAPE
    assert out.dtype == np.uint8
    # value check: out[0, y, x, c] == chw[c, y, x]
    assert out[0, 5, 7, 2] == chw[2, 5, 7]
    assert out[0, 100, 33, 0] == chw[0, 100, 33]


def test_T5b_gripper_detile():
    # de-tile takes index [19] of the 25-vec
    s = np.zeros(25, dtype=np.float32)
    s[19:25] = 0.42
    g = C.detile_gripper(s)
    assert np.isclose(g, 0.42)


# ---------------------------------------------------------------------------
# T6: reward/done/infos set correctly
# ---------------------------------------------------------------------------
def test_T6_success_reward_done_infos(converted):
    # all but terminal: reward 0
    for tr in converted[:-1]:
        assert tr["rewards"] == 0.0
        assert tr["dones"] in (False, 0, 0.0)
        assert "infos" in tr
        assert tr["infos"]["grasp_penalty"] == 0.0
    # terminal: reward 1.0, done True
    tlast = converted[-1]
    assert tlast["rewards"] == 1.0
    assert bool(tlast["dones"]) is True
    assert tlast["infos"]["grasp_penalty"] == 0.0


def test_T6b_fail_demo_all_zero_reward(raw_demo):
    conv_fail = C.convert_demo(raw_demo, is_success=False)
    assert all(tr["rewards"] == 0.0 for tr in conv_fail)


# ---------------------------------------------------------------------------
# T7 (regression): tcp_vel is rotated by EACH transition's OWN same-step pose,
#   matching the live RelativeFrame (relative_env.py L51/L62 set transform_matrix
#   from the current obs pose; L77-78 do inv(transform_matrix) @ tcp_vel). NOT the
#   reset-frame pose. Catches the bug where convert_obs rotated every transition's
#   tcp_vel by the FIRST transition's 6x6 transform (5-24% systematic mismatch).
# ---------------------------------------------------------------------------
def test_T7_tcp_vel_uses_own_same_step_pose(raw_demo, converted):
    # reset-frame inverse = the OLD/buggy transform (built once from the first obs)
    s0 = np.asarray(raw_demo[0]["observations"]["state"], dtype=np.float64)
    q0 = C.quat_wxyz_to_xyzw(s0[C.SL_QUAT])
    vel_reset_inv = np.linalg.inv(C.construct_transform_matrix(q0))

    discriminating = 0  # transitions where own-pose result != reset-pose result
    for i, tr in enumerate(raw_demo):
        s = np.asarray(tr["observations"]["state"], dtype=np.float64)
        q = C.quat_wxyz_to_xyzw(s[C.SL_QUAT])
        raw_vel = s[C.SL_VEL]
        # ground truth: live wrapper rotates by THIS step's own same-step pose
        gt_vel = np.linalg.inv(C.construct_transform_matrix(q)) @ raw_vel
        conv_vel = np.asarray(converted[i]["observations"]["state"][0][6:12],
                              dtype=np.float64)
        np.testing.assert_allclose(
            conv_vel, gt_vel, atol=1e-5,
            err_msg=f"tcp_vel must use own same-step pose; mismatch at transition {i}")
        # measure how different the buggy reset-frame result would be
        reset_vel = vel_reset_inv @ raw_vel
        if not np.allclose(gt_vel, reset_vel, atol=1e-3):
            discriminating += 1
    # the demo must contain transitions whose orientation differs from reset,
    # else the test cannot distinguish the bug (reset-frame) from the fix (own-pose)
    assert discriminating > 0, "non-discriminating demo: no transition rotates away from reset"


# ---------------------------------------------------------------------------
# T8 (insert-only): insertion_start_index drops grasp+transport, lands the EE
#   within `margin` of the seated x (above the socket), and is strictly inside.
# ---------------------------------------------------------------------------
def test_T8_insertion_start_index(raw_demo):
    margin = 0.05
    k = C.insertion_start_index(raw_demo, margin)
    n = len(raw_demo)
    assert 0 < k < n, f"trim index {k} not strictly inside [0,{n})"
    xs = np.array([np.asarray(tr["observations"]["state"], float)[C.SL_POS][0]
                   for tr in raw_demo])
    seated_x = float(np.median(xs[-5:]))
    # at/after the trim point the EE stays within margin of the socket x ...
    assert np.all(xs[k:] >= seated_x - margin - 1e-9)
    # ... and the frame just before it is OUTSIDE (k is the true run start)
    assert xs[k - 1] < seated_x - margin
    # transport really was dropped (full demo starts far back in x)
    assert xs[0] < seated_x - margin


# ---------------------------------------------------------------------------
# T9 (insert-only): convert_demo(start=k) trims to demo[k:], the reset frame is
#   the NEW first frame (rel pos ~0), and reward/done land on the real terminal.
# ---------------------------------------------------------------------------
def test_T9_convert_demo_start_trims_and_resets(raw_demo):
    k = C.insertion_start_index(raw_demo, 0.05)
    conv = C.convert_demo(raw_demo, is_success=True, start=k)
    assert len(conv) == len(raw_demo) - k
    # reset is the new first frame -> its relative tcp_pose pos == 0
    np.testing.assert_allclose(conv[0]["observations"]["state"][0][:3], [0, 0, 0], atol=1e-6)
    # terminal still carries the success reward/done
    assert conv[-1]["rewards"] == 1.0 and bool(conv[-1]["dones"]) is True
    assert conv[-1]["masks"] == 0.0
    # a mid frame is NOT terminal
    assert conv[len(conv) // 2]["rewards"] == 0.0
