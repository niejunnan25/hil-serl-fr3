"""Regression: replay_pure_fk must normalize rpy by ACTION_SCALE[3]=0.1, not [1]=0.015.

Latent bug: gello_replay built ``action_scale = list(ACTION_SCALE)`` (the 7-element
contract tuple) and passed it to scripts/normalize_action.py, which uses a 3-element
[pos_scale, rpy_scale, gripper_scale] convention (xyz/scale[0], rpy/scale[1]). So rpy
was divided by ACTION_SCALE[1]=0.015 instead of ACTION_SCALE[3]=0.1 — ~6.7x too large,
then clipped to 1.0. xyz was fine (ACTION_SCALE[0]=0.015 is correct either way).

The existing test_state_25d.py only checks action shape/dtype/range, all of which the
clip satisfies, so it never caught this.
"""
import numpy as np

from sim.data import gello_replay

_POS_SCALE = 0.015   # ACTION_SCALE[0]
_RPY_SCALE = 0.1     # ACTION_SCALE[3]
_BUG_SCALE = 0.015   # ACTION_SCALE[1] — what the buggy code used for rpy


def _demo_with_rpy_delta():
    # Small wrist/elbow steps -> an EE rotation delta whose /0.1 stays unsaturated
    # (so the exact proportionality is checkable) but whose /0.015 saturates (the bug).
    q0 = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.7], dtype=np.float64)
    q1 = q0.copy()
    q1[5] += 0.03
    q1[3] += 0.01
    joint_poses = np.stack([q0, q1])
    demo = {
        "joint_poses": joint_poses,
        "gripper_states": np.array([0.5, 0.5], dtype=np.float64),  # nonzero -> no zero-norm filtering
        "timestamps": np.array([0.0, 1.0 / 30.0], dtype=np.float64),
    }
    return demo, joint_poses


def test_replay_pure_fk_rpy_uses_rpy_scale_not_pos_scale():
    demo, joint_poses = _demo_with_rpy_delta()
    deltas = gello_replay._local_trajectory_to_cartesian_deltas(joint_poses)

    transitions = gello_replay.replay_pure_fk(demo, max_frames=2)
    assert len(transitions) == 2, "gripper=0.5 should keep both actions non-zero-norm"

    d = deltas[1]  # second frame carries the nonzero rotation
    assert np.max(np.abs(d[3:6])) > 1e-4, "test demo must produce a nonzero rpy delta"

    action = np.asarray(transitions[1]["actions"], dtype=np.float64)

    # rpy MUST be divided by 0.1 (then clipped), NOT by 0.015.
    expected_rpy = np.clip(d[3:6] / _RPY_SCALE, -1.0, 1.0)
    np.testing.assert_allclose(action[3:6], expected_rpy, atol=1e-5)

    # xyz still uses 0.015 (unchanged / always correct).
    expected_xyz = np.clip(d[0:3] / _POS_SCALE, -1.0, 1.0)
    np.testing.assert_allclose(action[0:3], expected_xyz, atol=1e-5)

    # Guard documenting the regression: the old /0.015 rpy scaling must NOT match.
    buggy_rpy = np.clip(d[3:6] / _BUG_SCALE, -1.0, 1.0)
    assert not np.allclose(action[3:6], buggy_rpy, atol=1e-5), (
        "rpy channels are still divided by ACTION_SCALE[1]=0.015 (the bug)"
    )
