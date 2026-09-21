import numpy as np

from sim.safety import feasibility_checker as fc


def test_joint_velocity_limit_catches_fast_motion_below_step_delta_limit():
    assert hasattr(fc, "DEFAULT_CONTROL_HZ")
    assert hasattr(fc, "check_joint_velocity")

    q0 = np.zeros(7, dtype=np.float64)
    q1 = q0.copy()
    q1[0] = (fc.JOINT_VEL_LIMIT[0] / fc.DEFAULT_CONTROL_HZ) * 1.25
    q_seq = np.vstack([q0, q1])

    violations = fc.check_joint_velocity(q_seq, control_hz=fc.DEFAULT_CONTROL_HZ)

    assert [v.category for v in violations] == [fc.ViolationCategory.JOINT_VEL]
    assert violations[0].timestep == 1


def test_action_chunk_applies_joint_velocity_gate_by_default():
    action_chunk = np.zeros((2, 8), dtype=np.float64)
    action_chunk[0, 5] = 0.6
    action_chunk[1, 5] = 0.6 + (fc.JOINT_VEL_LIMIT[5] / fc.DEFAULT_CONTROL_HZ) * 1.25

    report = fc.check_action_chunk(action_chunk)

    assert fc.ViolationCategory.JOINT_VEL in report.categories()
