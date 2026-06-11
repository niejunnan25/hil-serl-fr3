"""Runtime check: ensure real-robot stack is idle before launching sim.

Called by:
  1. scripts/sim/phase0_smoke.py before any Isaac Lab API call
  2. scripts/sim/phase1_replay_lerobot.py (Phase 1+)
  3. any sim entry point that needs the safety gate

The intent is enforcement of spec §5.1 constraint 8/12:
  sim 启动前必须 verify pgrep run_server.py / franka_panda_client / launch_robot 为空,
  以及 ss -ltn :4242/:50051/:50052 未监听.

Prevents sim from starting while a real robot stack is running
(portable check, not droid-specific).
"""
from __future__ import annotations

import socket
import subprocess
from typing import Iterable


# Real-robot processes that must NOT be alive while sim runs
_FORBIDDEN_PROCESS_PATTERNS: tuple[str, ...] = (
    "run_server.py",
    "franka_panda_client",
    "franka_hand_client",
    "launch_robot",
    "launch_gripper",
    "polymetis.*build/run_server",
    "pi05_main_async",
)

# Real-robot ports that must NOT be listening
_FORBIDDEN_PORTS: tuple[int, ...] = (4242, 50051, 50052)


class RealStackBusy(RuntimeError):
    """Raised when real-robot stack is detected as alive/listening."""


def _self_pid_chain() -> set[int]:
    """Return self PID + parent + grandparent PIDs to exclude from pgrep matches.

    R-PHASE0-1: when this checker is called from a shell whose cmdline contains
    polymetis-pattern strings (e.g. nested ssh script monitoring polymetis),
    pgrep matches the caller's own shell. Filtering self-chain prevents false
    positives without weakening detection of real polymetis processes.
    """
    import os
    pids: set[int] = {os.getpid()}
    pid = os.getppid()
    while pid > 1:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        pid = int(line.split()[1])
                        break
                else:
                    break
        except (OSError, ValueError):
            break
    return pids


def _is_pgrep_self_match(cmdline: str) -> bool:
    """Return True for pgrep reporting its own `pgrep -af <pattern>` command."""
    parts = cmdline.split()
    return bool(parts) and parts[0].endswith("pgrep")


def check_no_polymetis_processes(
    patterns: Iterable[str] = _FORBIDDEN_PROCESS_PATTERNS,
) -> None:
    """Use `pgrep -af <pattern>` to detect any forbidden process.

    Raises RealStackBusy if any matching process is found, EXCLUDING the
    current process and its ancestor shells (R-PHASE0-1: prevents pgrep
    matching the caller's own cmdline that happens to contain polymetis
    pattern strings — e.g. monitoring scripts).
    """
    self_chain = _self_pid_chain()
    busy: list[str] = []
    for pat in patterns:
        result = subprocess.run(
            ["pgrep", "-af", pat],
            capture_output=True, text=True, check=False,
        )
        # pgrep exit 0 = match found; exit 1 = no match (the idle case)
        if result.returncode == 0 and result.stdout.strip():
            real_matches = []
            for line in result.stdout.strip().split("\n"):
                # pgrep -af format: "PID cmdline"
                parts = line.split(None, 1)
                if not parts:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                if pid in self_chain:
                    continue   # skip self/ancestors
                cmdline = parts[1] if len(parts) > 1 else ""
                if _is_pgrep_self_match(cmdline):
                    continue   # skip the pgrep child process itself
                real_matches.append(line)
            if real_matches:
                busy.append(f"{pat}: " + "; ".join(real_matches))

    if busy:
        msg = "Real-robot processes alive (sim cannot start):\n  " + "\n  ".join(busy)
        raise RealStackBusy(msg)


def check_no_zerorpc_listening(ports: Iterable[int] = _FORBIDDEN_PORTS) -> None:
    """Use socket connect probe to detect anything listening on forbidden ports.

    Raises RealStackBusy if any port has a listener (connect succeeds).
    """
    listening: list[int] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                listening.append(port)
            except (ConnectionRefusedError, socket.timeout, OSError):
                # connection refused or no service = idle = good
                pass

    if listening:
        ports_str = ", ".join(str(p) for p in listening)
        raise RealStackBusy(
            f"Real-robot ports still listening on localhost: {ports_str}"
        )


def ensure_real_stack_idle() -> None:
    """Top-level safety gate; called by sim entry points."""
    check_no_polymetis_processes()
    check_no_zerorpc_listening()


if __name__ == "__main__":
    import sys
    try:
        ensure_real_stack_idle()
        print("OK: real-robot stack is idle, sim safe to start")
        sys.exit(0)
    except RealStackBusy as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        sys.exit(1)
