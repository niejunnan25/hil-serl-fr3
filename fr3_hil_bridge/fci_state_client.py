from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROBOT_IP = "172.16.0.2"
LIBFRANKA_BUILD = "/home/robot/libfranka/build"
POLYMETIS_LIB = "/home/robot/miniconda3/envs/polymetis-local/lib"
LD_LIBRARY_PATH = f"{LIBFRANKA_BUILD}:{POLYMETIS_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
GRIPPER_PROBE = Path("/home/robot/serl_projects/hil-serl-fr3/artifacts/fci_gripper_state_only_state_read/read_gripper_state_only")


@dataclass(frozen=True)
class FciGripperStateResult:
    ok: bool
    status: str
    robot_ip: str
    fci_state: dict[str, Any]
    gripper_state: dict[str, Any]
    read_counts: dict[str, int]
    runtime_claims: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    return subprocess.run(args, text=True, capture_output=True, check=False, timeout=timeout, env=env)


def _json_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            rows.append(json.loads(stripped))
    return rows


def _finite_vector(values: Any, length: int) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"expected vector length {length}")
    out = [float(value) for value in values]
    if not all(math.isfinite(value) for value in out):
        raise ValueError("non-finite vector value")
    return out


def _matrix_to_pose(values: list[float]) -> list[float]:
    _finite_vector(values, 16)
    r00, r01, r02 = values[0], values[4], values[8]
    r10, r11, r12 = values[1], values[5], values[9]
    r20, r21, r22 = values[2], values[6], values[10]
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("bad quaternion norm")
    return [
        float(values[12]),
        float(values[13]),
        float(values[14]),
        qw / norm,
        qx / norm,
        qy / norm,
        qz / norm,
    ]


def fetch_fci_gripper_state(
    robot_ip: str = ROBOT_IP,
    fci_timeout_seconds: int = 3,
    gripper_reads: int = 3,
) -> FciGripperStateResult:
    if not GRIPPER_PROBE.exists():
        raise RuntimeError(f"GRIPPER_PROBE_MISSING {GRIPPER_PROBE}")

    fci = _run(["timeout", str(fci_timeout_seconds), "/usr/bin/echo_robot_state", robot_ip], timeout=fci_timeout_seconds + 6)
    if fci.returncode != 0:
        raise RuntimeError(f"FCI_STATE_READ_FAILED returncode={fci.returncode} stderr={fci.stderr[:500]}")
    fci_rows = _json_rows(fci.stdout)
    if not fci_rows:
        raise RuntimeError("FCI_STATE_READ_EMPTY")

    gripper = _run(["timeout", "8", str(GRIPPER_PROBE), robot_ip, str(gripper_reads), "20"], timeout=12)
    if gripper.returncode != 0:
        raise RuntimeError(f"GRIPPER_STATE_READ_FAILED returncode={gripper.returncode} stderr={gripper.stderr[:500]}")
    gripper_rows = _json_rows(gripper.stdout)
    if not gripper_rows:
        raise RuntimeError("GRIPPER_STATE_READ_EMPTY")

    fci_sample = fci_rows[-1]
    gripper_sample = gripper_rows[-1]
    return FciGripperStateResult(
        ok=True,
        status="FR3_FCI_GRIPPER_STATE_FETCHED",
        robot_ip=robot_ip,
        fci_state={
            "robot_mode": str(fci_sample.get("robot_mode", "")).upper(),
            "q": _finite_vector(fci_sample.get("q"), 7),
            "dq": _finite_vector(fci_sample.get("dq"), 7),
            "tcp_pose": _matrix_to_pose(fci_sample.get("O_T_EE")),
            "tcp_wrench": _finite_vector(fci_sample.get("O_F_ext_hat_K"), 6),
            "timestamps_ms": [int(row["time"]) for row in fci_rows[:20] if isinstance(row.get("time"), int)],
        },
        gripper_state={
            "width_m": float(gripper_sample["width_m"]),
            "max_width_m": float(gripper_sample["max_width_m"]),
            "is_grasped": bool(gripper_sample["is_grasped"]),
            "temperature_c": float(gripper_sample["temperature_c"]),
            "timestamps_ms": [int(row["time_ms"]) for row in gripper_rows[:20] if isinstance(row.get("time_ms"), int)],
        },
        read_counts={"fci": min(len(fci_rows), 20), "gripper": min(len(gripper_rows), 20)},
        runtime_claims={
            "fr3_connection": True,
            "fci_state_read": True,
            "gripper_state_read": True,
            "motion_command": False,
            "cartesian_command": False,
            "joint_command": False,
            "impedance_start": False,
            "gripper_command": False,
            "gripper_open": False,
            "gripper_close": False,
            "policy_rollout": False,
            "droid_mutation": False,
            "openpi_dependency": False,
        },
    )
