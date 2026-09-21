from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FR3_J6_LOWER = 0.5445


def test_canonical_fr3_limits_do_not_allow_old_panda_joint6_lower():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from fr3_joint_limits import FR3_LOWER_LIMITS, FR3_UPPER_LIMITS

    assert FR3_LOWER_LIMITS.shape == (7,)
    assert FR3_UPPER_LIMITS.shape == (7,)
    assert float(FR3_LOWER_LIMITS[5]) >= OFFICIAL_FR3_J6_LOWER
    assert float(FR3_LOWER_LIMITS[5]) < float(FR3_UPPER_LIMITS[5])


def test_sim_feasibility_limits_match_canonical_fr3_joint_limits():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from fr3_joint_limits import FR3_LOWER_LIMITS, FR3_UPPER_LIMITS
    from sim.safety.feasibility_checker import JOINT_POS_LOWER, JOINT_POS_UPPER

    np.testing.assert_allclose(JOINT_POS_LOWER, FR3_LOWER_LIMITS)
    np.testing.assert_allclose(JOINT_POS_UPPER, FR3_UPPER_LIMITS)


def test_no_script_keeps_the_old_joint6_lower_literal():
    old_limit_snippets = (
        "[-2.8, -1.66, -2.8, -2.97, -2.8, 0.08, -2.8]",
        "[-2.80, -1.66, -2.80, -2.97, -2.80, 0.08, -2.80]",
    )
    checked_roots = (REPO_ROOT / "scripts", REPO_ROOT / "sim" / "safety")

    offenders = []
    for root in checked_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(snippet in text for snippet in old_limit_snippets):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []
