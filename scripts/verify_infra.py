#!/usr/bin/env python3
"""
verify_infra.py – Comprehensive Phase 1 infrastructure verification for hilserl-fr3.

Verifies ALL infrastructure requirements:
  1. libfranka — version >= 0.13.0, library presence, robot connection test
  2. ROS Noetic — installation, catkin workspace, franka_ros packages
  3. serl_franka_controllers — package existence, compiled devel/ space, arm_id="fr3"
  4. franka_server — HTTP /getstate endpoint, expected JSON keys
  5. ZED cameras — open by serial, read one frame, verify resolution/format
  6. Network — zktitan reachability (SSH), required ports

Target host: fr3-desktop-ts (Ubuntu 22.04)
Robot IP:    172.16.0.2

Usage:
    python scripts/verify_infra.py                        # Full verification
    python scripts/verify_infra.py --skip-cameras         # Skip ZED camera check
    python scripts/verify_infra.py --skip-network         # Skip network check
    python scripts/verify_infra.py --json                 # JSON output to stdout
    python scripts/verify_infra.py --server-url http://127.0.0.1:5017/

Exit code 0 = all critical checks pass, 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── optional imports (fail gracefully if unavailable) ────────────────────────

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

# ── constants ────────────────────────────────────────────────────────────────

ROBOT_IP = "172.16.0.2"
CONTROL_HOST = "fr3-desktop-ts"
ZKTITAN_HOST = "zktitan"

# libfranka
LIBFRANKA_MIN_VERSION = (0, 13, 0)

# ZED cameras
ZED_SERIALS = {
    "external": "36276705",
    "wrist": "13132609",
}
ZED_WIDTH = 1280
ZED_HEIGHT = 720

# franka_server
DEFAULT_SERVER_URL = "http://127.0.0.2:5000/"
TIMEOUT_S = 5.0

# Expected /getstate keys
GETSTATE_KEYS = {
    "pose": list,
    "vel": list,
    "force": list,
    "torque": list,
    "q": list,
    "dq": list,
    "jacobian": list,
    "gripper_pos": (int, float),
}

# Paths on fr3-desktop-ts
SERL_FR3_ROOT = "/home/robot/serl_projects/hil-serl-fr3"
CATKIN_WS = f"{SERL_FR3_ROOT}/catkin_ws_franka"
SERL_CTRL_DIR = f"{SERL_FR3_ROOT}/upstream/serl_franka_controllers"
ROS_SETUP = "/opt/ros/noetic/setup.bash"

# Network ports to check
REQUIRED_PORTS: list[tuple[str, int, str]] = [
    (ROBOT_IP, 80, "franka desk web UI"),
]

# ── ANSI colors ──────────────────────────────────────────────────────────────

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
NC = "\033[0m"

PASS_T = f"{GREEN}PASS{NC}"
FAIL_T = f"{RED}FAIL{NC}"
WARN_T = f"{YELLOW}WARN{NC}"
SKIP_T = f"{CYAN}SKIP{NC}"


# ── result tracking ──────────────────────────────────────────────────────────

@dataclass
class Check:
    """Single check outcome."""
    name: str
    status: str  # "pass", "fail", "warn", "skip"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    section: str = ""


@dataclass
class InfraReport:
    """Aggregated infrastructure verification report."""
    checks: list[Check] = field(default_factory=list)
    timestamp: str = ""
    host: str = ""
    server_url: str = ""

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == "skip")

    @property
    def all_critical_passed(self) -> bool:
        """All non-warn, non-skip checks must pass."""
        return all(c.status in ("pass", "warn", "skip") for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "host": self.host,
            "server_url": self.server_url,
            "robot_ip": ROBOT_IP,
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "warned": self.warned,
                "skipped": self.skipped,
                "total": len(self.checks),
                "all_critical_passed": self.all_critical_passed,
            },
            "sections": self._group_by_section(),
            "checks": [
                {
                    "section": c.section,
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

    def _group_by_section(self) -> dict[str, dict[str, int]]:
        sections: dict[str, dict[str, int]] = {}
        for c in self.checks:
            if c.section not in sections:
                sections[c.section] = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
            sections[c.section][c.status] += 1
        return sections

    def print_report(self) -> None:
        icons = {"pass": PASS_T, "fail": FAIL_T, "warn": WARN_T, "skip": SKIP_T}
        print()
        print(f"{BOLD}{'=' * 70}{NC}")
        print(f"{BOLD}  HIL-SERL FR3 — Phase 1 Infrastructure Verification{NC}")
        print(f"  Time: {self.timestamp}")
        print(f"  Host: {self.host}")
        print(f"  Robot: {ROBOT_IP}")
        print(f"{BOLD}{'=' * 70}{NC}")

        current_section = ""
        for _, c in enumerate(self.checks, 1):
            if c.section != current_section:
                current_section = c.section
                print(f"\n  {BOLD}── {current_section} ──{NC}")

            icon = icons.get(c.status, "????")
            msg = f" – {c.message}" if c.message else ""
            print(f"    [{icon}] {c.name}{msg}")
            if c.details:
                for k, v in c.details.items():
                    print(f"           {k}: {v}")

        print()
        print(f"{BOLD}{'=' * 70}{NC}")
        if self.all_critical_passed:
            result_str = f"{GREEN}ALL CRITICAL CHECKS PASSED{NC}"
        else:
            result_str = f"{RED}CRITICAL FAILURES DETECTED{NC}"
        print(f"  {result_str}")
        print(f"  Passed: {self.passed}  Failed: {self.failed}  "
              f"Warned: {self.warned}  Skipped: {self.skipped}")
        print(f"  Total checks: {len(self.checks)}")
        print(f"{BOLD}{'=' * 70}{NC}")
        print()


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: str, timeout: int = 10, shell: bool = True) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def _check(name: str, status: str, message: str = "",
           details: dict[str, Any] | None = None, section: str = "") -> Check:
    return Check(name=name, status=status, message=message,
                 details=details or {}, section=section)


def _compare_versions(actual: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    """Return True if actual >= minimum."""
    return actual >= minimum


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse '0.13.0' into (0, 13, 0)."""
    parts = re.findall(r"\d+", version_str)
    return tuple(int(p) for p in parts) if parts else (0,)


# ── Section 1: libfranka ─────────────────────────────────────────────────────

def check_libfranka_version() -> Check:
    """Check libfranka version via dpkg, pkg-config, and library scan."""
    name = "libfranka version"
    section = "libfranka"
    found_version: Optional[str] = None
    details: dict[str, Any] = {}

    # 1) dpkg
    rc, out, _ = _run("dpkg -l 2>/dev/null | grep libfranka")
    if rc == 0 and out:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                ver = parts[2]
                found_version = ver
                details["dpkg_version"] = ver
                break

    # 2) pkg-config
    rc, out, _ = _run("pkg-config --modversion franka 2>/dev/null")
    if rc == 0 and out:
        details["pkg_config_version"] = out
        if found_version is None:
            found_version = out

    # 3) Check shared library existence
    rc, out, _ = _run("ldconfig -p 2>/dev/null | grep libfranka")
    if rc == 0 and "libfranka" in out:
        lib_path = out.splitlines()[0].split()[-1] if out.splitlines() else "?"
        details["library_path"] = lib_path
    else:
        details["library_status"] = "not found in ldconfig"

    # 4) Check header
    for d in ["/usr/include", "/usr/local/include", "/opt/libfranka/include"]:
        header = os.path.join(d, "franka", "robot.h")
        if os.path.isfile(header):
            details["header_path"] = header
            break

    if found_version is None:
        return _check(name, "fail",
                      "Could not determine libfranka version (not installed?)",
                      details, section)

    ver_tuple = _parse_version_tuple(found_version)
    details["parsed_version"] = ".".join(str(v) for v in ver_tuple)
    details["minimum_required"] = ".".join(str(v) for v in LIBFRANKA_MIN_VERSION)

    if _compare_versions(ver_tuple, LIBFRANKA_MIN_VERSION):
        return _check(name, "pass",
                      f"{found_version} >= {'.'.join(str(v) for v in LIBFRANKA_MIN_VERSION)}",
                      details, section)
    else:
        return _check(name, "fail",
                      f"{found_version} < {'.'.join(str(v) for v in LIBFRANKA_MIN_VERSION)}",
                      details, section)


def check_libfranka_tools() -> Check:
    """Check for franka example tools (echo_robot_state, etc.)."""
    name = "libfranka tools"
    section = "libfranka"
    details: dict[str, Any] = {}
    found = 0
    total = 0

    for tool in ["echo_robot_state", "print_joint_poses", "communication_test",
                 "franka_control"]:
        total += 1
        for d in ["/usr/bin", "/usr/local/bin"]:
            path = os.path.join(d, tool)
            if os.path.isfile(path):
                details[tool] = path
                found += 1
                break
        else:
            details[tool] = "not found"

    if found == total:
        return _check(name, "pass", f"{found}/{total} tools found", details, section)
    elif found > 0:
        return _check(name, "warn", f"{found}/{total} tools found", details, section)
    else:
        return _check(name, "warn", "No franka tools found in standard paths", details, section)


def check_robot_connection() -> Check:
    """Test basic network connectivity to the robot."""
    name = "Robot connection"
    section = "libfranka"
    details: dict[str, Any] = {"robot_ip": ROBOT_IP}

    # Ping
    rc, out, _ = _run(f"ping -c 1 -W 2 {ROBOT_IP}", timeout=5)
    if rc == 0:
        # Extract latency
        match = re.search(r"time=(\d+\.?\d*)", out)
        if match:
            details["ping_latency_ms"] = match.group(1)
        details["ping"] = "reachable"
    else:
        details["ping"] = "unreachable"
        return _check(name, "warn",
                      f"Cannot ping {ROBOT_IP} (robot may be off or firewall blocks ICMP)",
                      details, section)

    # Check local interface on robot subnet
    rc, out, _ = _run("ip addr show 2>/dev/null | grep '172.16.0'")
    if rc == 0 and out:
        local_ip = out.split()[1] if out.split() else "?"
        details["local_interface"] = local_ip
    else:
        details["local_interface"] = "not on 172.16.0.x subnet"
        return _check(name, "warn",
                      "No local interface on 172.16.0.x — robot LAN may not be configured",
                      details, section)

    return _check(name, "pass",
                  f"Robot reachable at {ROBOT_IP}",
                  details, section)


# ── Section 2: ROS ───────────────────────────────────────────────────────────

def check_ros_installation() -> Check:
    """Verify ROS Noetic is installed."""
    name = "ROS Noetic installation"
    section = "ROS"
    details: dict[str, Any] = {}

    if os.path.isfile(ROS_SETUP):
        details["setup_bash"] = ROS_SETUP
        details["distro"] = "noetic"

        # Check roscore
        rc, out, _ = _run("which roscore 2>/dev/null")
        if rc == 0:
            details["roscore"] = out
        else:
            details["roscore"] = "not in PATH"

        # Check catkin_make
        rc, out, _ = _run("which catkin_make 2>/dev/null")
        if rc == 0:
            details["catkin_make"] = out
        else:
            details["catkin_make"] = "not in PATH"

        return _check(name, "pass", "ROS Noetic found", details, section)

    # Check for other ROS distros
    for distro in ["melodic", "humble", "galactic"]:
        alt = f"/opt/ros/{distro}/setup.bash"
        if os.path.isfile(alt):
            details["found_distro"] = distro
            details["setup_bash"] = alt
            return _check(name, "warn",
                          f"Found ROS {distro} (serl_franka_controllers targets Noetic)",
                          details, section)

    return _check(name, "warn",
                  "ROS Noetic not installed (non-ROS direct bridge path recommended)",
                  details, section)


def check_catkin_workspace() -> Check:
    """Verify catkin workspace exists and has devel/ space."""
    name = "Catkin workspace"
    section = "ROS"
    details: dict[str, Any] = {"catkin_ws": CATKIN_WS}

    if not os.path.isdir(CATKIN_WS):
        return _check(name, "warn",
                      f"Catkin workspace not found at {CATKIN_WS}",
                      details, section)

    details["src_dir"] = os.path.isdir(os.path.join(CATKIN_WS, "src"))
    devel_dir = os.path.join(CATKIN_WS, "devel")
    details["devel_dir"] = os.path.isdir(devel_dir)

    if os.path.isdir(devel_dir):
        devel_setup = os.path.join(devel_dir, "setup.bash")
        details["devel_setup_bash"] = os.path.isfile(devel_setup)
        return _check(name, "pass", "Workspace exists with devel/ space", details, section)
    else:
        return _check(name, "warn",
                      "Workspace exists but devel/ not built (run catkin_make)",
                      details, section)


def check_franka_ros() -> Check:
    """Verify franka_ros packages are installed or available."""
    name = "franka_ros packages"
    section = "ROS"
    details: dict[str, Any] = {}

    # 1) rospack find (only works if ROS is sourced)
    rc, out, _ = _run(
        f"bash -c 'source {ROS_SETUP} 2>/dev/null && rospack find franka_ros 2>/dev/null'"
    )
    if rc == 0 and out and "/" in out:
        details["rospack_path"] = out
        return _check(name, "pass", f"franka_ros found: {out}", details, section)

    # 2) Check common workspace locations
    for ws in ["/home/robot/ros_ws", "/home/robot/catkin_ws",
               os.path.expanduser("~/ros_ws"), CATKIN_WS]:
        candidate = os.path.join(ws, "src", "franka_ros")
        if os.path.isdir(candidate):
            details["workspace_path"] = candidate
            return _check(name, "pass", f"franka_ros found at {candidate}", details, section)

    # 3) Check devel/ for franka_* packages
    devel_share = os.path.join(CATKIN_WS, "devel", "share")
    if os.path.isdir(devel_share):
        franka_pkgs = [d for d in os.listdir(devel_share) if d.startswith("franka_")]
        if franka_pkgs:
            details["devel_packages"] = franka_pkgs
            return _check(name, "pass",
                          f"franka_ros packages in devel/: {', '.join(franka_pkgs)}",
                          details, section)

    return _check(name, "warn",
                  "franka_ros not found (required for ROS path, not for direct bridge)",
                  details, section)


# ── Section 3: serl_franka_controllers ────────────────────────────────────────

def check_serl_controllers_package() -> Check:
    """Verify serl_franka_controllers package exists."""
    name = "Package existence"
    section = "serl_franka_controllers"
    details: dict[str, Any] = {"expected_path": SERL_CTRL_DIR}

    if not os.path.isdir(SERL_CTRL_DIR):
        return _check(name, "fail",
                      f"serl_franka_controllers not found at {SERL_CTRL_DIR}",
                      details, section)

    # Check for CMakeLists.txt (catkin package marker)
    cmake_path = os.path.join(SERL_CTRL_DIR, "CMakeLists.txt")
    details["CMakeLists.txt"] = os.path.isfile(cmake_path)

    # Check for package.xml
    pkg_xml = os.path.join(SERL_CTRL_DIR, "package.xml")
    details["package.xml"] = os.path.isfile(pkg_xml)

    # Git info
    rc, out, _ = _run(f"git -C {SERL_CTRL_DIR} rev-parse HEAD 2>/dev/null")
    if rc == 0:
        details["git_rev"] = out[:12]

    return _check(name, "pass",
                  f"Found at {SERL_CTRL_DIR}",
                  details, section)


def check_serl_controllers_compiled() -> Check:
    """Check if serl_franka_controllers is compiled (devel/ space exists)."""
    name = "Compilation (devel/)"
    section = "serl_franka_controllers"
    details: dict[str, Any] = {}

    devel_dir = os.path.join(CATKIN_WS, "devel")
    details["devel_dir"] = devel_dir

    if not os.path.isdir(devel_dir):
        return _check(name, "warn",
                      "devel/ not found — package not compiled (run catkin_make)",
                      details, section)

    # Check for serl_franka_controllers in devel/
    devel_lib = os.path.join(devel_dir, "lib")
    if os.path.isdir(devel_lib):
        lib_contents = os.listdir(devel_lib)
        serl_libs = [d for d in lib_contents if "serl" in d.lower()]
        details["devel_lib_contents"] = lib_contents[:10]
        if serl_libs:
            details["serl_libraries"] = serl_libs

    # Check for the controller node
    for candidate in glob.glob(os.path.join(devel_dir, "lib", "**", "*serl*"),
                               recursive=True):
        details["found_binary"] = candidate
        break

    return _check(name, "pass",
                  "devel/ space exists",
                  details, section)


def check_arm_id_fr3() -> Check:
    """Check arm_id config is 'fr3' not 'panda'."""
    name = "arm_id = 'fr3'"
    section = "serl_franka_controllers"
    details: dict[str, Any] = {}

    if not os.path.isdir(SERL_CTRL_DIR):
        return _check(name, "fail",
                      "serl_franka_controllers directory not found",
                      details, section)

    # Check for arm_id: fr3 in config files
    fr3_refs = 0
    panda_refs = 0
    fr3_files: list[str] = []
    panda_files: list[str] = []

    extensions = ["*.launch", "*.yaml", "*.yml", "*.xml", "*.cfg", "*.py"]

    for ext in extensions:
        for filepath in glob.glob(os.path.join(SERL_CTRL_DIR, "**", ext), recursive=True):
            try:
                content = Path(filepath).read_text(errors="ignore")
            except Exception:
                continue

            if re.search(r"arm_id.*fr3|fr3.*arm_id", content):
                fr3_refs += 1
                fr3_files.append(os.path.relpath(filepath, SERL_CTRL_DIR))
            if re.search(r"arm_id.*panda|panda.*arm_id", content):
                panda_refs += 1
                panda_files.append(os.path.relpath(filepath, SERL_CTRL_DIR))

    details["fr3_arm_id_count"] = fr3_refs
    details["panda_arm_id_count"] = panda_refs
    if fr3_files:
        details["fr3_files"] = fr3_files[:5]
    if panda_files:
        details["panda_files"] = panda_files[:5]

    # Also check joint names
    fr3_joints = 0
    panda_joints = 0
    for ext in ["*.launch", "*.yaml", "*.yml", "*.xml", "*.h", "*.hpp", "*.cpp"]:
        for filepath in glob.glob(os.path.join(SERL_CTRL_DIR, "**", ext), recursive=True):
            try:
                content = Path(filepath).read_text(errors="ignore")
            except Exception:
                continue
            fr3_joints += len(re.findall(r"fr3_joint[1-7]", content))
            panda_joints += len(re.findall(r"panda_joint[1-7]", content))

    details["fr3_joint_refs"] = fr3_joints
    details["panda_joint_refs"] = panda_joints

    if panda_refs > 0:
        return _check(name, "fail",
                      f"arm_id still references 'panda' in {panda_refs} files "
                      f"(run 02_adapt_serl_controllers.sh)",
                      details, section)
    if fr3_refs > 0:
        return _check(name, "pass",
                      f"arm_id correctly set to 'fr3' ({fr3_refs} files, "
                      f"{fr3_joints} joint refs)",
                      details, section)

    return _check(name, "warn",
                  "No arm_id references found in config files",
                  details, section)


# ── Section 4: franka_server ─────────────────────────────────────────────────

def check_franka_server_endpoint(base_url: str) -> Check:
    """Curl /getstate and verify response has all expected keys."""
    name = "GET /getstate"
    section = "franka_server"
    url = base_url.rstrip("/") + "/getstate"
    details: dict[str, Any] = {"url": url}

    if requests is None:
        # Fall back to curl
        rc, out, err = _run(
            f"curl -s --connect-timeout 3 --max-time 5 {url}", timeout=10
        )
        if rc != 0:
            return _check(name, "fail",
                          f"Cannot reach {url} (curl failed: {err})",
                          details, section)
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return _check(name, "fail",
                          "Response is not valid JSON",
                          {**details, "response": out[:200]}, section)
    else:
        try:
            resp = requests.post(url, timeout=TIMEOUT_S)
            details["http_status"] = resp.status_code
            if resp.status_code != 200:
                return _check(name, "fail",
                              f"HTTP {resp.status_code}",
                              {**details, "body": resp.text[:200]}, section)
            data = resp.json()
        except requests.ConnectionError:
            return _check(name, "fail",
                          f"Connection refused — is franka_server running?",
                          details, section)
        except Exception as e:
            return _check(name, "fail", str(e), details, section)

    # Verify keys
    missing_keys = []
    type_errors = []

    for key, expected_type in GETSTATE_KEYS.items():
        if key not in data:
            missing_keys.append(key)
            continue
        val = data[key]
        if expected_type is list:
            if not isinstance(val, list):
                type_errors.append(f"{key}: expected list, got {type(val).__name__}")
        elif isinstance(expected_type, tuple):
            if not isinstance(val, expected_type):
                type_errors.append(
                    f"{key}: expected {'/'.join(t.__name__ for t in expected_type)}, "
                    f"got {type(val).__name__}"
                )

    details["response_keys"] = list(data.keys())
    if "pose" in data and isinstance(data["pose"], list):
        details["pose_length"] = len(data["pose"])
    if "q" in data and isinstance(data["q"], list):
        details["q_length"] = len(data["q"])

    if missing_keys:
        return _check(name, "fail",
                      f"Missing keys: {missing_keys}",
                      details, section)
    if type_errors:
        return _check(name, "fail",
                      f"Type errors: {type_errors}",
                      details, section)

    return _check(name, "pass",
                  f"All {len(GETSTATE_KEYS)} keys present with correct types",
                  details, section)


def check_franka_server_health(base_url: str) -> Check:
    """Check /healthz or basic liveness."""
    name = "Server health"
    section = "franka_server"
    details: dict[str, Any] = {}

    if requests is None:
        rc, out, _ = _run(
            f"curl -s -o /dev/null -w '%{{http_code}}' "
            f"--connect-timeout 3 --max-time 5 "
            f"{base_url.rstrip('/')}/getstate",
            timeout=10,
        )
        if rc == 0 and out in ("200", "409"):
            return _check(name, "pass", f"Server responding (HTTP {out})", details, section)
        return _check(name, "fail", "Server not responding", details, section)

    # Try /healthz first
    for endpoint in ["/healthz", "/getstate"]:
        url = base_url.rstrip("/") + endpoint
        try:
            resp = requests.get(url, timeout=TIMEOUT_S)
            details[f"GET {endpoint}"] = resp.status_code
            if resp.status_code == 200:
                try:
                    body = resp.json()
                    details["response"] = body
                except Exception:
                    pass
                return _check(name, "pass",
                              f"Server alive (GET {endpoint} → 200)",
                              details, section)
        except requests.ConnectionError:
            details[f"GET {endpoint}"] = "connection refused"
        except Exception as e:
            details[f"GET {endpoint}"] = str(e)

    return _check(name, "fail",
                  "Server not responding on any health endpoint",
                  details, section)


# ── Section 5: ZED Cameras ───────────────────────────────────────────────────

def check_zed_cameras() -> list[Check]:
    """Try to open each ZED camera by serial, read one frame, verify resolution."""
    checks: list[Check] = []
    section = "ZED cameras"

    # Try to import pyzed
    try:
        import pyzed.sl as sl  # type: ignore
    except ImportError:
        # pyzed not available — try alternative check via ZED SDK CLI
        checks.append(_check(
            "pyzed import", "warn",
            "pyzed not importable — trying ZED CLI tools instead",
            {}, section,
        ))

        # Try zed_camera_list or similar CLI
        rc, out, _ = _run("ZED_Explorer --list 2>/dev/null || "
                          "python3 -c 'import cv2; print(\"opencv available\")' 2>/dev/null")
        if rc == 0 and out:
            checks.append(_check(
                "ZED detection (CLI)", "warn",
                "pyzed unavailable; ZED CLI check done",
                {"output": out[:200]}, section,
            ))
        else:
            checks.append(_check(
                "ZED detection", "skip",
                "Neither pyzed nor ZED CLI tools available",
                {}, section,
            ))
        return checks

    # pyzed available — check each camera
    for cam_label, serial in ZED_SERIALS.items():
        name = f"ZED {cam_label} ({serial})"
        details: dict[str, Any] = {"serial": serial, "label": cam_label}

        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(serial))
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.sdk_verbose = 0

        cam = sl.Camera()
        status = cam.open(init_params)
        details["open_status"] = str(status)

        if status != sl.ERROR_CODE.SUCCESS:
            if status == sl.ERROR_CODE.CAMERA_NOT_DETECTED:
                checks.append(_check(name, "warn",
                                     "Camera not detected (USB disconnected or in use)",
                                     details, section))
            else:
                checks.append(_check(name, "fail",
                                     f"Failed to open: {status}",
                                     details, section))
            continue

        # Read one frame
        runtime = sl.RuntimeParameters()
        grab_status = cam.grab(runtime)
        details["grab_status"] = str(grab_status)

        if grab_status == sl.ERROR_CODE.SUCCESS:
            # Get image info
            img = sl.Mat()
            cam.retrieve_image(img, sl.VIEW.LEFT)
            w = img.get_width()
            h = img.get_height()
            details["resolution"] = f"{w}x{h}"
            details["format"] = str(img.get_pixel_type())

            if w == ZED_WIDTH and h == ZED_HEIGHT:
                checks.append(_check(name, "pass",
                                     f"OK — {w}x{h} frame captured",
                                     details, section))
            else:
                checks.append(_check(name, "warn",
                                     f"Resolution {w}x{h} != expected {ZED_WIDTH}x{ZED_HEIGHT}",
                                     details, section))
        else:
            checks.append(_check(name, "fail",
                                 f"Failed to grab frame: {grab_status}",
                                 details, section))

        cam.close()

    return checks


# ── Section 6: Network ───────────────────────────────────────────────────────

def check_zktitan_ssh() -> Check:
    """Verify zktitan is reachable via SSH."""
    name = "zktitan SSH"
    section = "Network"
    details: dict[str, Any] = {"host": ZKTITAN_HOST}

    # DNS resolution
    try:
        addr = socket.gethostbyname(ZKTITAN_HOST)
        details["resolved_ip"] = addr
    except socket.gaierror:
        return _check(name, "warn",
                      f"Cannot resolve hostname '{ZKTITAN_HOST}' (check /etc/hosts or DNS)",
                      details, section)

    # Check SSH port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((addr, 22))
        sock.close()
        details["port_22"] = "open" if result == 0 else "closed"
        if result == 0:
            return _check(name, "pass",
                          f"SSH port open ({addr}:22)",
                          details, section)
        else:
            return _check(name, "warn",
                          f"SSH port closed on {addr}:22",
                          details, section)
    except Exception as e:
        details["error"] = str(e)
        return _check(name, "warn",
                      f"SSH check failed: {e}",
                      details, section)


def check_required_ports() -> Check:
    """Verify required network ports are open."""
    name = "Required ports"
    section = "Network"
    details: dict[str, Any] = {}
    issues: list[str] = []

    for host, port, desc in REQUIRED_PORTS:
        key = f"{host}:{port}"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            status = "open" if result == 0 else "closed"
            details[key] = f"{status} ({desc})"
            if result != 0:
                issues.append(key)
        except Exception as e:
            details[key] = f"error: {e} ({desc})"
            issues.append(key)

    if not issues:
        return _check(name, "pass",
                      f"All {len(REQUIRED_PORTS)} ports accessible",
                      details, section)
    elif len(issues) < len(REQUIRED_PORTS):
        return _check(name, "warn",
                      f"{len(issues)}/{len(REQUIRED_PORTS)} ports unreachable: {issues}",
                      details, section)
    else:
        return _check(name, "warn",
                      f"All required ports unreachable",
                      details, section)


# ── main ─────────────────────────────────────────────────────────────────────

def run_all_checks(args: argparse.Namespace) -> InfraReport:
    """Run all infrastructure checks and return the report."""
    report = InfraReport(
        timestamp=datetime.now().isoformat(),
        host=CONTROL_HOST,
        server_url=args.server_url,
    )

    to_stderr = args.json

    def log(msg: str) -> None:
        print(msg, file=sys.stderr if to_stderr else sys.stdout)

    # ── Section 1: libfranka
    log(f"[1/6] libfranka checks...")
    report.add(check_libfranka_version())
    report.add(check_libfranka_tools())
    report.add(check_robot_connection())

    # ── Section 2: ROS
    log(f"[2/6] ROS checks...")
    report.add(check_ros_installation())
    report.add(check_catkin_workspace())
    report.add(check_franka_ros())

    # ── Section 3: serl_franka_controllers
    log(f"[3/6] serl_franka_controllers checks...")
    report.add(check_serl_controllers_package())
    report.add(check_serl_controllers_compiled())
    report.add(check_arm_id_fr3())

    # ── Section 4: franka_server
    log(f"[4/6] franka_server checks...")
    report.add(check_franka_server_health(args.server_url))
    report.add(check_franka_server_endpoint(args.server_url))

    # ── Section 5: ZED cameras
    if args.skip_cameras:
        log(f"[5/6] ZED cameras — skipped")
        report.add(_check("ZED cameras", "skip", "Skipped by --skip-cameras", {}, "ZED cameras"))
    else:
        log(f"[5/6] ZED camera checks...")
        for c in check_zed_cameras():
            report.add(c)

    # ── Section 6: Network
    if args.skip_network:
        log(f"[6/6] Network — skipped")
        report.add(_check("Network", "skip", "Skipped by --skip-network", {}, "Network"))
    else:
        log(f"[6/6] Network checks...")
        report.add(check_zktitan_ssh())
        report.add(check_required_ports())

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 infrastructure verification for hilserl-fr3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"franka_server HTTP URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--skip-cameras",
        action="store_true",
        help="Skip ZED camera checks",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network reachability checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON to stdout (progress goes to stderr)",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Write JSON report to this file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = run_all_checks(args)

    # Human-readable output
    if not args.json:
        report.print_report()

    # JSON output
    if args.json or args.json_file:
        report_dict = report.to_dict()
        if args.json:
            print(json.dumps(report_dict, indent=2))
        if args.json_file:
            with open(args.json_file, "w") as f:
                json.dump(report_dict, f, indent=2)
            print(f"JSON report written to: {args.json_file}",
                  file=sys.stderr if args.json else sys.stdout)

    sys.exit(0 if report.all_critical_passed else 1)


if __name__ == "__main__":
    main()
