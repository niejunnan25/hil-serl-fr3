"""Regression for BUG-9 (2026-06-18 read-only review).

``experiments/plug_insertion/config.py`` does, at module import time::

    from scripts.xbox_intervention import XboxIntervention, HoldGripperWrapper  # :31
    ...
    env = HoldGripperWrapper(env)                                              # :304

The canonical / laptop + DESK_HIL copy of ``scripts/xbox_intervention.py``
(copy A) shipped ``XboxIntervention`` but NOT ``HoldGripperWrapper``, so any
laptop-direct or DESK_HIL actor start raised ``ImportError`` at
``get_environment()``. DESK_SERL only ran because of an uncommitted
working-tree copy. These tests pin the symbol's presence and behaviour so the
canonical tree stays internally consistent with ``config.py``.
"""

from __future__ import annotations

import numpy as np


def test_config_import_symbols_present():
    # Exactly the import performed by experiments/plug_insertion/config.py:31.
    from scripts.xbox_intervention import (  # noqa: F401
        HoldGripperWrapper,
        XboxIntervention,
    )

    assert HoldGripperWrapper is not None


def test_hold_gripper_wrapper_holds_gripper_channel():
    from scripts.xbox_intervention import HoldGripperWrapper

    # action() is env-independent (only reads self.hold_value and rewrites the
    # gripper channel), so build the instance without a real gym.Env.
    w = HoldGripperWrapper.__new__(HoldGripperWrapper)
    w.hold_value = 0.0

    out = w.action(np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0], dtype=np.float32))
    assert out[6] == 0.0  # gripper channel neutralised to no-op (|x|<0.5)
    assert np.allclose(out[:6], [0.1, -0.2, 0.3, 0.4, -0.5, 0.6])  # rest untouched


def test_hold_gripper_wrapper_passes_through_6d_action():
    from scripts.xbox_intervention import HoldGripperWrapper

    w = HoldGripperWrapper.__new__(HoldGripperWrapper)
    w.hold_value = 0.0

    # A 6-D (gripper-less) action has no channel 6 -> returned unchanged.
    out6 = w.action(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32))
    assert np.allclose(out6, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
