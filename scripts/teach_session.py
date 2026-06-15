"""teach_session.py — interactive FR3 demo-collection session (run ON the desktop).

The operator drives the WHOLE flow from the desktop terminal, so start/stop
signalling is in the terminal in front of them (no chat-latency desync):

  1. Connectivity check + (default) RESET the robot to HOME via /jointreset.
     The robot moves AUTONOMOUSLY — the script warns and waits for Enter first;
     operator must be clear of the arm with the E-stop in hand.
  2. Open devices + start the hybrid GELLO/Xbox recording. "▶ 示教现在开始" is
     printed when teaching truly begins (after home + device init).
  3. Operator performs the demo (GELLO grasp -> ☰ -> Xbox insert); Ctrl-C stops.
  4. Prompt 成功/失败/丢弃 + notes; label the saved demo + append to index.jsonl.

Usage (desktop):  ./scripts/teach.sh            # reset + record
                  ./scripts/teach.sh --no-reset # skip the home reset
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np  # noqa: E402

from hybrid_teleop import DEFAULT_LEADER_SCALE, drive_gello, reset_to_home, run  # noqa: E402
from relative_teleop import DEFAULT_MAX_STEP, _session, get_state  # noqa: E402


def save_home(path: str, pose, q=None) -> None:
    """Persist a home pose (7D) + optional joint config to json."""
    d = {"pose": [float(x) for x in np.asarray(pose).reshape(-1)[:7]]}
    if q is not None:
        d["q"] = [float(x) for x in np.asarray(q).reshape(-1)[:7]]
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_home(path: str):
    with open(path) as f:
        return np.asarray(json.load(f)["pose"], dtype=float)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def labeled_path(path: str, label: str) -> str:
    """Insert _<label> before the extension: demo_x.pkl -> demo_x_success.pkl."""
    root, ext = os.path.splitext(path)
    return f"{root}_{label}{ext}"


def make_index_record(res: dict, label: str, notes: str, ts: str) -> dict:
    """Build the index.jsonl record for one demo (pure)."""
    return {
        "ts": ts,
        "label": label,
        "notes": notes,
        "pkl": os.path.basename(labeled_path(res["pkl"], label)),
        "npz": os.path.basename(labeled_path(res["npz"], label)),
        "n_transitions": int(res["n_transitions"]),
        "elapsed_s": float(res["elapsed_s"]),
        "validate_ok": bool(res["validate_ok"]),
        "abort_reason": res.get("abort_reason", ""),
    }


def parse_label(s: str) -> str:
    """'s'/'success' -> 'success'; 'f'/'fail' -> 'fail'; 'd'/'discard' -> 'discard'."""
    s = (s or "").strip().lower()
    if s.startswith("s"):
        return "success"
    if s.startswith("f"):
        return "fail"
    if s.startswith("d"):
        return "discard"
    return "fail"  # default conservative


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------
def _confirm(prompt: str) -> bool:
    try:
        input(prompt)
        return True
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description="Interactive FR3 GELLO/Xbox demo session")
    p.add_argument("--server", default="http://172.16.0.1:5000/")
    p.add_argument("--out-dir", default="/home/robot/hilserl-fr3/demos/hybrid")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    p.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-reset", dest="reset", action="store_false",
                   help="skip the home reset")
    p.add_argument("--set-home", dest="set_home", action="store_true",
                   help="drive via GELLO, then Ctrl-C to capture the current pose as home")
    p.add_argument("--home-file", default=None,
                   help="home pose json (default <out-dir>/home.json)")
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    home_file = a.home_file or os.path.join(a.out_dir, "home.json")
    session = _session()

    print("=" * 64)
    print(" FR3 示教采集 (GELLO Route E + Xbox 混合)")
    print("=" * 64)
    try:
        s0 = get_state(session, a.server, timeout=5.0)
    except Exception as e:
        print(f"[ERR] 连不上 franka_server {a.server}: {type(e).__name__}: {e}")
        return 2
    print(f"已连接。当前位姿 currpos={np.round(np.asarray(s0['pose'])[:3], 4).tolist()}")

    # ---- set-home: position via GELLO, capture current pose as home ----
    if a.set_home:
        print("\n[SET-HOME] 用 GELLO 把机器人摆到你要的起始位姿（夹爪张开、插头上方等）。")
        print("  摆好后在本终端按 Ctrl-C，当前位姿就保存为 home。")
        if not _confirm("  确认区域安全、E-stop 在手，按 Enter 开始驱动（Ctrl-C 取消）..."):
            return 1
        try:
            pose, q = drive_gello(session, a.server, leader_scale=a.leader_scale,
                                  max_step=a.max_step, hz=a.hz)
        except Exception as e:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print(f"\n[ERR] set-home 驱动失败: {type(e).__name__}: {e}")
            return 3
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        save_home(home_file, pose, q)
        print(f"\n✓ home 已保存: {home_file}")
        print(f"  pose(xyz)={[round(float(x), 4) for x in pose[:3]]}  (full 7D + q stored)")
        print("  之后直接 ./scripts/teach.sh 就会先笛卡尔复位回这个 home 再开录。")
        return 0

    # ---- normal: cartesian reset to home, then record ----
    if a.reset:
        if not os.path.exists(home_file):
            print(f"\n[ERR] 没有 home 文件: {home_file}")
            print("  先设 home: ./scripts/teach.sh --set-home   或跳过复位: --no-reset")
            return 4
        home = load_home(home_file)
        print(f"\n⚠ 复位：机器人将**笛卡尔缓慢移动**回 home (xyz={[round(float(x), 3) for x in home[:3]]})。")
        print("  确认夹爪/路径上无障碍、E-stop 在手。")
        if not _confirm("  按 Enter 开始复位（Ctrl-C 取消）..."):
            return 1
        print("复位中（机器人移动）...", flush=True)
        try:
            reset_to_home(session, a.server, home, max_step=max(a.max_step, 0.006))
        except (Exception, KeyboardInterrupt) as e:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print(f"[ERR] 复位失败: {type(e).__name__}: {e}")
            return 3
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("✓ 复位完成，已回 home。")
    else:
        print("\n(跳过复位 --no-reset)")

    print("\n设备初始化中（GELLO/Xbox/ZED/FK，约 8-12s）...")
    print(">>> 初始化后会提示 [ALIGN]：**把 GELLO 摆到和机器人一样的 HOME 姿态**（照着机械臂摆，")
    print("    夹爪朝下、手臂形状对齐），按 Enter——这步是主从对齐，不对齐会乱动。")
    print(">>> 然后 '▶ 示教现在开始'：GELLO 移动跟随 + 捏夹爪闭合；☰(Menu) 切 Xbox（GELLO 失效）；")
    print("    Xbox 按住 RB + 摇杆精插，RT 闭/LT 张；再按 ☰ 切回 GELLO。")
    print("    完成后在本终端按 Ctrl-C 结束示教。")
    print("-" * 64, flush=True)

    res = run(a.server, a.hz, a.duration, a.out_dir, a.max_step,
              a.leader_scale, a.fps, dry_run=False, align_prompt=True)

    # run() installed its own SIGINT handler; restore default for the prompts.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if res is None:
        print("\n[结束] 数据太少，未保存。")
        return 0

    print("\n" + "=" * 64)
    print(f" 本条示教: {res['n_transitions']} transitions, {res['elapsed_s']}s, "
          f"validate={'PASS' if res['validate_ok'] else 'FAIL'}")
    try:
        label = parse_label(input(" 结果 成功(s)/失败(f)/丢弃(d)? "))
    except (EOFError, KeyboardInterrupt):
        label = "fail"
        print("\n(默认记为 fail)")

    if label == "discard":
        for k in ("pkl", "npz"):
            try:
                os.remove(res[k])
            except OSError:
                pass
        print("✓ 已丢弃本条。")
        return 0

    try:
        notes = input(" 备注(可选，回车跳过): ").strip()
    except (EOFError, KeyboardInterrupt):
        notes = ""

    ts = os.path.splitext(os.path.basename(res["pkl"]))[0]
    for k in ("pkl", "npz"):
        try:
            os.rename(res[k], labeled_path(res[k], label))
        except OSError as e:
            print(f"[WARN] rename {k} failed: {e}")
    rec = make_index_record(res, label, notes, ts)
    index_path = os.path.join(a.out_dir, "index.jsonl")
    with open(index_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_ok = _count_label(index_path, "success")
    print(f"✓ 已记录: {label}  -> {rec['pkl']}")
    print(f"  索引: {index_path}  (累计 success={n_ok})")
    return 0


def _count_label(index_path: str, label: str) -> int:
    n = 0
    try:
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("label") == label:
                        n += 1
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return n


if __name__ == "__main__":
    sys.exit(main())
