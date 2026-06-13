#!/usr/bin/env python3
"""Full-scene IsaacLab 5.1.0 viewer: droid FR3 + droid table + CN plug + CN socket.

This renders, in ONE InteractiveScene, the proven droid FR3-vegetable-grasp
assets (the FR3 arm spawned via build_fr3_cfg and the procedural black table +
robot mount from primitive_scene) PLUS the CN plug and CN socket USD meshes,
lit by the droid dome+sun lights, framed by a pinhole Camera over the workspace.
Z-up (IsaacLab default).

Why this combines two recipes:
  * STRUCTURE  (FR3 + table + lights + home-pose pin) follows the WORKING
    droid phase3_scene_preview.py: AppLauncher -> SimulationContext ->
    make_scene_cfg(dataclasses.make_dataclass + @configclass) -> InteractiveScene
    -> ensure_fr3_gripper_collision_proxies -> reset -> write FR3 home joints.
  * CAPTURE  (headless ground-truth PNG) reuses the WORKING plug_scene_viewer.py:
    a Camera sensor at /World/RenderCamera, set_world_poses_from_view, settle
    frames + sim.render(), camera.data.output["rgb"] RGBA -> drop alpha -> PNG,
    RGB_MEAN + SAVED, and os._exit(0) to dodge the RTX-5080 close() hang.

PATCH (adversarial review, 2026-06-13):
  * The plug/socket are now spawned BEFORE sim.reset(), matching the proven
    plug_scene_viewer.py ordering. InteractiveScene(Cfg()) creates
    /World/envs/env_0 in its constructor (the single-env clone happens before
    reset), so authoring the static visual props into that namespace pre-reset
    guarantees they are present for the first annotator/render pass. Authoring
    fresh prims after the timeline is PLAYing risked them missing the first
    capture frame.
  * Per the plug/socket USD README the meshes are authored Y-up with the panel
    face / body height along USD-Y and the pins / insertion direction along
    USD-Z. A +90deg-about-X (Y->Z) puts the LARGER dimension vertical, so the
    socket (80x80x25 mm) stands ~80 mm tall (NOT 25 mm). Centering it at
    z=0.0125 would bury ~67 mm of it under the tabletop (z=0) and it would read
    as a thin sliver or vanish. Both parts are therefore raised so their
    authored center sits clearly ABOVE the table (plug z=0.03, socket z=0.05).
    The EXACT resting pose (which face down, and whether to flip the quat sign)
    must still be dialed on the first headless RGB -- but the parts are
    guaranteed visible above the table on this single expensive run instead of
    risking a sunk/invisible socket.

Two modes (argparse, separate from AppLauncher args):

  * DEFAULT (headless RGB capture, ground truth):
        builds the scene, steps several frames so the RTX denoiser settles,
        flushes with sim.render(), reads camera.data.output["rgb"], drops alpha,
        saves a PNG to --out. Prints RGB_MEAN=<float> and SAVED=<path>.

  * --gui (windowed inspection):
        opens an Isaac Sim window, frames the workspace with set_camera_view,
        holds the FR3 home pose, keeps the app alive for --seconds.

Run on the GPU host (orchestrator runs this, not the author):

    # headless ground-truth PNG (default)
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    XDG_RUNTIME_DIR=/run/user/1000 XDG_SESSION_TYPE=x11 \
    /home/robot/IsaacLab/isaaclab.sh -p plug_fullscene_viewer.py

    # GUI window
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    XDG_RUNTIME_DIR=/run/user/1000 XDG_SESSION_TYPE=x11 \
    /home/robot/IsaacLab/isaaclab.sh -p plug_fullscene_viewer.py --gui --seconds 120
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from typing import Any

# Line-buffer stdout so the orchestrator sees VIEWER_READY / RGB_MEAN / SAVED
# promptly even when piped.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Defaults (verified facts on fr3-desktop-ts).
# ---------------------------------------------------------------------------
DEFAULT_PLUG_USD = "/home/robot/plug_insertion_sim/sim-scene/cn_two_pin_plug.usd"
DEFAULT_SOCKET_USD = "/home/robot/plug_insertion_sim/sim-scene/cn_three_pin_socket.usd"
DEFAULT_OUT_PNG = "/home/robot/plug_insertion_sim/sim-scene/_fullscene_capture.png"

# Camera framing: FR3 at origin; the 270mm BULL strip lies along world X,
# centered at x=0.45 on the table top (z=0); the 2-pin plug sits in front of it.
# Elevated 3/4 view so the 6 outlets on the strip's top face are visible.
CAM_EYE = (0.52, 0.13, 0.15)
CAM_TARGET = (0.45, 0.0, 0.02)

# 2-pin plug (still cn_two_pin_plug USD): in front of the strip, resting on table.
# -90 deg about X (authored Y-up): pins point UP so they are visible.
PLUG_TRANSLATION = (0.45, 0.15, 0.03)
PLUG_ORIENT = (0.70710678, -0.70710678, 0.0, 0.0)

# --- 公牛 GN-109K 六口排插 (built from colored primitives; no pxr/USD file) ---
# Grounded on: (a) live ZED 2i photo of the real unit (white, rounded, compact,
# side cord) and (b) BULL GN-109K spec: 6 outlets, body 204 x 92 x 29 mm, white,
# master(总控) switch. The 92mm width => 2 rows x 3 columns. Per user's unit:
# top row = 3 三口 (3-hole/三极 品字), bottom row = 3 双口 (2-hole/两极).
# Built Z-up, bottom on table top z=0, long axis (204mm) along world X.
STRIP_BASE = (0.45, 0.0, 0.0)
_STRIP_BODY = (0.204, 0.092, 0.029)        # GN-109K L(X) x W(Y) x H(Z) meters
_ROW_Y = (0.023, -0.023)                   # top row (三口), bottom row (双口)
_COL_X = (-0.058, 0.0, 0.058)              # 3 columns
_PATCH = (0.050, 0.040, 0.0015)            # per-outlet faceplate recess
_STRIP_SLOT = (0.0015, 0.0063, 0.006)
_STRIP_SLOT_SPACING = 0.0127
_GND = (0.0018, 0.0070)                     # 三极 ground hole (taller)
_C_BODY = (0.92, 0.92, 0.93); _C_PATCH = (0.72, 0.72, 0.76); _C_SLOT = (0.03, 0.03, 0.04)
_C_SWITCH = (0.85, 0.10, 0.10); _C_CORD = (0.05, 0.05, 0.05); _C_METAL = (0.60, 0.60, 0.62)

# Plug/socket spawn directly into the cloned single-env namespace. env_0 is
# created by the InteractiveScene constructor (the single-env clone happens
# BEFORE reset), so the props are authored after InteractiveScene(Cfg()) and
# BEFORE sim.reset() -- matching the proven plug_scene_viewer.py spawn ordering.
ENV_ROOT = "/World/envs/env_0"


def spawn_six_outlet_strip(sim_utils, parent_path, base=STRIP_BASE):
    """Build a BULL-style CN 6-outlet power strip from colored cuboids.

    3 two-pin (双口) + 3 three-pin (三口) outlets, red rocker switch, own 3-pin
    wall cord. Z-up, bottom resting on table top z=0, long axis along world X.
    Returns the list of spawned prim paths. No pxr / no USD file needed.
    """
    bx, by, bz = base
    L, W, H = _STRIP_BODY
    paths = []

    def box(name, center, size, color):
        cfg = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        p = f"{parent_path}/{name}"
        cfg.func(p, cfg, translation=(bx + center[0], by + center[1], bz + center[2]))
        paths.append(p)

    box("Strip_Body", (0.0, 0.0, H / 2.0), (L, W, H), _C_BODY)
    sd = _STRIP_SLOT[2]
    sz = H - sd / 2.0 + 0.0008                      # dark slots flush w/ top, recessed down
    sp = _STRIP_SLOT_SPACING
    idx = 0
    # All 6 outlets are universal 五孔 (per user): 三极 品字 (ground on top + two
    # splayed L/N flats) ABOVE 两极 (two vertical L/N flats) = 5 holes each.
    for ry in _ROW_Y:
        for cx in _COL_X:
            box(f"Strip_Patch_{idx}", (cx, ry, H + 0.00075), _PATCH, _C_PATCH)
            box(f"Strip_O{idx}_E", (cx, ry + 0.012, sz), (_GND[0], _GND[1], sd), _C_SLOT)  # ground E (top)
            for dx in (-0.0072, 0.0072):                                                    # 三极 L/N (splayed)
                box(f"Strip_O{idx}_t{'L' if dx < 0 else 'R'}", (cx + dx, ry + 0.004, sz), _STRIP_SLOT, _C_SLOT)
            for dx in (-sp / 2, sp / 2):                                                    # 两极 L/N (vertical)
                box(f"Strip_O{idx}_b{'L' if dx < 0 else 'R'}", (cx + dx, ry - 0.010, sz), _STRIP_SLOT, _C_SLOT)
            idx += 1
    # red 总控 master switch near +X end
    box("Strip_Switch", (0.090, 0.0, H + 0.004), (0.014, 0.030, 0.008), _C_SWITCH)
    # tail cord (out the -X end) ending in the strip's OWN 3-pin (三脚) plug. THIS
    # plug is the manipulated object: the FR3 re-inserts it into one of the strip's
    # own 五孔 outlets (de-energized self-loop = safe for real RL). Resting on the
    # table, blades pointing +X toward the strip (insertion direction).
    box("Strip_Cord", (-0.128, 0.0, 0.006), (0.052, 0.009, 0.009), _C_CORD)
    px = -0.182                                    # 三脚 plug body center (local X)
    box("Strip_Plug_Body", (px, 0.0, 0.013), (0.034, 0.027, 0.026), _C_BODY)
    box("Strip_Plug_PinE", (px + 0.025, 0.0, 0.020), (0.016, 0.0032, 0.0016), _C_METAL)   # ground (top, longer)
    for dy in (-0.0064, 0.0064):                                                          # L/N blades (splayed)
        box(f"Strip_Plug_Pin{'L' if dy < 0 else 'R'}", (px + 0.024, dy, 0.0085), (0.014, 0.0016, 0.0050), _C_METAL)
    return paths


def parse_cli_args() -> argparse.Namespace:
    """Parse this script's own CLI args (separate from AppLauncher args)."""
    parser = argparse.ArgumentParser(
        description="Full-scene FR3 + table + plug + socket IsaacLab viewer.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open a windowed Isaac Sim viewport and keep it alive "
        "(default: headless RGB capture to --out).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_PNG,
        help=f"headless capture PNG output path (default: {DEFAULT_OUT_PNG}).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help="GUI mode: max seconds to keep the window alive (default: 120).",
    )
    parser.add_argument(
        "--plug-usd",
        default=DEFAULT_PLUG_USD,
        help="plug USD path.",
    )
    parser.add_argument(
        "--socket-usd",
        default=DEFAULT_SOCKET_USD,
        help="socket USD path.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="capture camera width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="capture camera height (default: 720).",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=40,
        help="headless: render frames before reading rgb so RTX settles "
        "(default: 40).",
    )
    parser.add_argument(
        "--rendering-mode",
        default="quality",
        choices=["performance", "balanced", "quality"],
    )
    # parse_known_args so any stray AppLauncher flags do not crash us.
    args, _unknown = parser.parse_known_args()
    return args


def build_app_argv(gui: bool, rendering_mode: str) -> list[str]:
    """Build the EXACT proven AppLauncher argv (droid phase3 / plug_scene recipe).

    --enable_cameras is MANDATORY even in headless, or camera.data.output["rgb"]
    never populates. --kit_args disables the DLSS denoiser + DLSS-G, the recipe
    proven to render non-black on this RTX 5080. --headless is inserted at the
    front when NOT --gui.
    """
    kit_args_str = (
        "--/rtx/denoiser/dlss/enabled=false "
        "--/rtx-transient/dlssg/enabled=false"
    )
    app_argv = [
        "--enable_cameras",
        "--rendering_mode", rendering_mode,
        "--kit_args", kit_args_str,
    ]
    if not gui:
        app_argv.insert(0, "--headless")
    return app_argv


def main() -> int:
    cli_args = parse_cli_args()

    # -----------------------------------------------------------------------
    # STEP 0: real-stack safety guard FIRST, before AppLauncher (droid idiom).
    # Aborts only if a live FR3 control stack is up; safe for this sim viewer.
    # NOTE: ensure_real_stack_idle() may internally call sys.exit(); SystemExit
    # is BaseException (NOT caught by 'except Exception'), so a deliberate abort
    # there propagates cleanly. RealStackBusy IS an Exception subclass, so a
    # busy real stack is swallowed and the sim viewer proceeds (acceptable: no
    # robot motion in this viewer). If the droid import path is unavailable, the
    # viewer still runs (sim-only).
    # -----------------------------------------------------------------------
    try:
        from droid.sim.safety.runtime_check import ensure_real_stack_idle
        ensure_real_stack_idle()
        print("[fullscene] ensure_real_stack_idle=OK (no live FR3 stack)", flush=True)
    except Exception as exc:
        print(f"[fullscene] ensure_real_stack_idle skipped ({exc})", flush=True)

    # -----------------------------------------------------------------------
    # STEP 1: launch the app FIRST, before any isaaclab.sim / .sensors import.
    # carb is only importable AFTER launch.
    # -----------------------------------------------------------------------
    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_argv = build_app_argv(gui=cli_args.gui, rendering_mode=cli_args.rendering_mode)
    app_args = app_parser.parse_args(app_argv)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    # Heavy imports AFTER app launch only.
    import pathlib
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils import configclass

    # droid reused assets (the droid package is importable in this env).
    from droid.sim.assets.lighting import build_distant_light, build_dome_light
    from droid.sim.assets.official_fr3_loader import (
        build_fr3_cfg,
        ensure_fr3_gripper_collision_proxies,
    )
    from droid.sim.assets.primitive_scene import (
        ROBOT_BASE_POS,
        build_robot_mount_cfg,
        build_table_cfg,
    )

    # -----------------------------------------------------------------------
    # STEP 2: build the InteractiveScene cfg dynamically (droid phase3 pattern).
    # FR3 (build_fr3_cfg at ROBOT_BASE_POS) + table + robot_mount + dome + sun.
    # The lambda cfg=cfg binding avoids the late-binding closure bug.
    # -----------------------------------------------------------------------
    def make_scene_cfg(scene_assets: dict[str, Any]) -> type[InteractiveSceneCfg]:
        fields: list[tuple[str, type, Any]] = [
            ("num_envs", int, dataclasses.field(default=1)),
            ("env_spacing", float, dataclasses.field(default=2.5)),
            ("replicate_physics", bool, dataclasses.field(default=True)),
            ("robot", Any, dataclasses.field(default_factory=lambda: build_fr3_cfg(
                prim_path="{ENV_REGEX_NS}/Fixed_Robot",
                base_pos=ROBOT_BASE_POS))),
        ]
        for name, cfg in scene_assets.items():
            fields.append((
                name,
                Any,
                dataclasses.field(default_factory=lambda cfg=cfg: cfg),
            ))
        fields.extend([
            ("dome_light", Any, dataclasses.field(default_factory=build_dome_light)),
            ("sun_light", Any, dataclasses.field(default_factory=build_distant_light)),
        ])
        return configclass(
            dataclasses.make_dataclass("Cfg", fields, bases=(InteractiveSceneCfg,))
        )

    # Workspace = table + robot mount ONLY (no vegetables/plate/checkerboard).
    # Each builder is independently guarded so one failure does not kill the rest.
    scene_assets: dict[str, Any] = {}
    for name, builder in (("table", build_table_cfg), ("robot_mount", build_robot_mount_cfg)):
        try:
            scene_assets[name] = builder()
            print(f"[fullscene] asset_cfg=OK {name}", flush=True)
        except Exception as exc:
            print(f"[fullscene] asset_cfg=FAILED {name} : {exc}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 3: SimulationContext (+ RenderCfg with DLSS off; in-code twin of the
    # --kit_args flags). render_interval=1 so each headless settle step renders.
    # -----------------------------------------------------------------------
    render_cfg = None
    try:
        from isaaclab.sim import RenderCfg
        render_cfg = RenderCfg(
            antialiasing_mode="DLAA",
            carb_settings={
                "rtx.denoiser.dlss.enabled": False,
                "rtx-transient.dlssg.enabled": False,
            },
        )
    except (ImportError, TypeError) as exc:
        print(f"[fullscene] RenderCfg unavailable ({exc}); relying on --kit_args.",
              flush=True)

    if render_cfg is not None:
        sim_cfg = SimulationCfg(dt=1.0 / 60.0, render_interval=1, render=render_cfg)
    else:
        sim_cfg = SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = SimulationContext(sim_cfg)

    # GUI viewport framing (no-op when --headless): aim at the workspace.
    sim.set_camera_view(eye=CAM_EYE, target=CAM_TARGET)

    # -----------------------------------------------------------------------
    # STEP 4: instantiate the scene, add gripper collision proxies BEFORE reset.
    # InteractiveScene(Cfg()) performs the single-env clone here, so
    # /World/envs/env_0 exists immediately after this line (before reset).
    # -----------------------------------------------------------------------
    Cfg = make_scene_cfg(scene_assets)
    scene = InteractiveScene(Cfg())
    try:
        gripper_collision_proxies = ensure_fr3_gripper_collision_proxies(
            f"{ENV_ROOT}/Fixed_Robot"
        )
        print(f"[fullscene] gripper_collision_proxies={len(gripper_collision_proxies)}",
              flush=True)
    except Exception as exc:
        print(f"[fullscene] gripper_collision_proxies=FAILED {exc}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 5: spawn plug + socket as static visual USD prims onto the table,
    # in front of the FR3. PATCH: this is done BEFORE sim.reset() (env_0 already
    # exists from the clone above) to match the proven plug_scene_viewer.py
    # ordering -- authoring prims after the timeline is PLAYing risked them
    # missing the first annotator/render pass. Each wrapped in try/except so one
    # missing/broken asset never aborts the render. Passing
    # translation=/orientation= OVERRIDES the USD's authored Y-up xform (proven
    # in plug_scene_viewer.py via spawn_from_usd).
    # -----------------------------------------------------------------------
    spawned: list[str] = []
    # Authored USD assets (generate_gn109k_usd.py): real GN-109K strip with
    # recessed 五孔 outlets + the strip's own 三脚 tail-cord plug. The plug is the
    # manipulated object, re-inserted into a 五孔 outlet (de-energized self-loop =
    # safe for real RL). Both USD are Z-up, bottom on table top z=0.
    STRIP_USD = "/home/robot/plug_insertion_sim/sim-scene/cn_gn109k_strip.usd"
    PLUG_USD = "/home/robot/plug_insertion_sim/sim-scene/cn_gn109k_plug.usd"
    try:
        sc = sim_utils.UsdFileCfg(usd_path=STRIP_USD)
        sc.func(f"{ENV_ROOT}/Strip", sc, translation=(0.45, 0.0, 0.0))
        spawned.append(f"{ENV_ROOT}/Strip")
        print(f"[fullscene] strip_usd=OK <- {STRIP_USD}", flush=True)
    except Exception as exc:
        print(f"[fullscene] strip_usd=FAILED {exc}", flush=True)
    try:
        pc = sim_utils.UsdFileCfg(usd_path=PLUG_USD)
        pc.func(f"{ENV_ROOT}/Plug", pc, translation=(0.30, 0.06, 0.0))
        spawned.append(f"{ENV_ROOT}/Plug")
        print(f"[fullscene] plug_usd=OK <- {PLUG_USD}", flush=True)
    except Exception as exc:
        print(f"[fullscene] plug_usd=FAILED {exc}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 6: create the camera sensor (PLAIN /World path -- single env, no
    # InteractiveScene clone for it). clipping (0.01, 1e6) so the small plug/
    # socket near z~0.05 are not clipped at eye ~1.5 m. Reused verbatim from
    # plug_scene_viewer.py.
    # -----------------------------------------------------------------------
    camera = None
    try:
        camera = Camera(cfg=CameraCfg(
            prim_path="/World/RenderCamera",
            update_period=0.0,
            height=cli_args.height,
            width=cli_args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1.0e6),
            ),
        ))
        print("[fullscene] camera=OK /World/RenderCamera", flush=True)
    except Exception as exc:
        print(f"[fullscene] camera=FAILED {exc}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 7: reset (timeline PLAY wires the camera annotators + articulation),
    # then pin the FR3 home pose so the arm holds its default configuration.
    # -----------------------------------------------------------------------
    sim.reset()

    home_joint_pos = None
    zero_joint_vel = None
    robot = None
    try:
        robot = scene["robot"]
        home_joint_pos = robot.data.default_joint_pos.clone()
        zero_joint_vel = torch.zeros_like(home_joint_pos)
        robot.write_joint_state_to_sim(home_joint_pos, zero_joint_vel)
        scene.write_data_to_sim()
        print("[fullscene] fr3_home_pose=OK (pinned)", flush=True)
    except Exception as exc:
        robot = None
        print(f"[fullscene] fr3_home_pose=FAILED {exc}", flush=True)

    print(f"VIEWER_READY spawned={spawned} gui={cli_args.gui}", flush=True)

    # =======================================================================
    # GUI MODE: frame the workspace, hold FR3 home, keep the window alive.
    # =======================================================================
    if cli_args.gui:
        sim.set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
        if camera is not None:
            try:
                eyes = torch.tensor([list(CAM_EYE)], device=sim.device,
                                    dtype=torch.float32)
                targets = torch.tensor([list(CAM_TARGET)], device=sim.device,
                                       dtype=torch.float32)
                camera.set_world_poses_from_view(eyes, targets)
            except Exception as exc:
                print(f"[fullscene] gui set_world_poses_from_view FAILED {exc}",
                      flush=True)
        deadline = time.time() + cli_args.seconds
        print(f"[fullscene] GUI alive for up to {cli_args.seconds:.0f}s "
              "(close the window to stop early).", flush=True)
        while simulation_app.is_running() and time.time() < deadline:
            try:
                if robot is not None and home_joint_pos is not None:
                    robot.write_joint_state_to_sim(home_joint_pos, zero_joint_vel)
                    scene.write_data_to_sim()
                sim.step(render=True)  # physics + render the viewport
                scene.update(dt=sim_cfg.dt)
                if camera is not None:
                    camera.update(dt=sim.get_physics_dt())
            except Exception as exc:
                print(f"[fullscene] gui_loop_error={exc}", flush=True)
                break
        simulation_app.close()
        print("[fullscene] closed", flush=True)
        return 0

    # =======================================================================
    # HEADLESS MODE (default): ground-truth RGB capture to PNG.
    # Capture body reused verbatim from plug_scene_viewer.py.
    # =======================================================================
    if camera is None:
        print("[fullscene] ERROR: camera not created; cannot capture. "
              "RGB_MEAN=0.00", flush=True)
        simulation_app.close()
        return 1

    # 1) AIM the camera (eye -> target). Both (N,3) torch tensors on sim.device,
    #    N == number of camera prims (1 here). MUST be after sim.reset().
    eyes = torch.tensor([list(CAM_EYE)], device=sim.device, dtype=torch.float32)
    targets = torch.tensor([list(CAM_TARGET)], device=sim.device, dtype=torch.float32)
    try:
        camera.set_world_poses_from_view(eyes, targets)  # Z-up OK
    except Exception as exc:
        print(f"[fullscene] set_world_poses_from_view FAILED {exc}", flush=True)

    # 2) STEP + RENDER several frames so the RTX denoiser / annotators converge.
    #    Hold the FR3 home pose each frame so the arm does not drift, and
    #    force_recompute so the annotator buffers refresh per frame.
    n_frames = max(8, int(cli_args.settle_frames))
    for _ in range(n_frames):
        try:
            if robot is not None and home_joint_pos is not None:
                robot.write_joint_state_to_sim(home_joint_pos, zero_joint_vel)
                scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_cfg.dt)
            camera.update(dt=sim.get_physics_dt(), force_recompute=True)
        except Exception as exc:
            print(f"[fullscene] settle_step_error={exc}", flush=True)
            break

    # 2b) Final explicit Hydra/fabric flush + buffer refresh before reading.
    try:
        sim.render()
        camera.update(dt=sim.get_physics_dt(), force_recompute=True)
    except Exception as exc:
        print(f"[fullscene] final_flush_error={exc}", flush=True)

    # 3) READ rgb, drop alpha, to uint8 numpy.
    rgb_mean = 0.0
    out_path = pathlib.Path(cli_args.out)
    saved = False
    try:
        rgb_t = camera.data.output["rgb"][0]  # (H,W,4) torch uint8
        arr = rgb_t.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]  # drop alpha before mean / save
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # 5) MEAN INTENSITY (non-black sanity check).
        rgb_mean = float(np.asarray(arr, dtype=np.float32).mean())

        # 4) SAVE PNG (PIL, fallback to imageio).
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(arr).save(str(out_path))
            saved = True
        except Exception as pil_exc:
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(out_path), arr)
                saved = True
            except Exception as iio_exc:
                print(f"[fullscene] PNG save FAILED pil={pil_exc} "
                      f"imageio={iio_exc}", flush=True)

        print(f"[fullscene] capture shape={arr.shape} dtype={arr.dtype}", flush=True)
    except Exception as exc:
        print(f"[fullscene] rgb_read FAILED {exc}", flush=True)

    # Required machine-readable lines for the orchestrator.
    print(f"RGB_MEAN={rgb_mean:.2f}", flush=True)
    if saved:
        print(f"SAVED={out_path}", flush=True)
    else:
        print(f"SAVED=NONE (intended {out_path})", flush=True)
    if rgb_mean <= 1.0:
        print("[fullscene] WARNING: mean ~0 -> still black. Add --settle-frames "
              "or raise dome/sun intensity.", flush=True)

    # 6) Clean exit. simulation_app.close() can HANG in headless Kit on this
    #    RTX 5080 box, so skip it and os._exit(0) -- the PNG is already on disk.
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
