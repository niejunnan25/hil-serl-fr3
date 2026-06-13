#!/usr/bin/env python3
"""Self-contained IsaacLab 5.1.0 viewer for the CN plug + socket preview.

This is a NO-MOTION visual/layout tool. Unlike the old
``plug_insertion_scene.py`` (which renders BLACK because its __main__ only does
SimulationContext()+reset()+step() on an EMPTY stage with no prims/lights/camera),
this script DIRECT-SPAWNS a ground plane, a DomeLight + DistantLight pair, the
plug and socket USD meshes, and a pinhole Camera aimed at the workspace.

Two modes (argparse):

  * DEFAULT (headless RGB capture, ground truth):
        builds the stage, steps several frames so the RTX denoiser/annotators
        settle, flushes Hydra textures with sim.render(), reads
        ``camera.data.output["rgb"]``, drops the alpha channel, and saves a PNG
        to --out. Prints ``RGB_MEAN=<float>`` and ``SAVED=<path>``. This avoids
        X11 window-capture artifacts.

  * --gui (windowed inspection):
        opens an Isaac Sim window, frames the workspace with set_camera_view,
        and keeps the app alive for --seconds (default 120) while stepping
        physics + rendering the viewport.

Z-up (IsaacLab default). The old script's "Y-up" claim is ignored: the plug /
socket USDs are referenced as visual prims and are visible regardless of their
authored up-axis.

Run on the GPU host (orchestrator runs this, not the author):

    # headless ground-truth PNG (default)
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    XDG_RUNTIME_DIR=/run/user/1000 XDG_SESSION_TYPE=x11 \
    /home/robot/IsaacLab/isaaclab.sh -p plug_scene_viewer.py

    # GUI window
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    XDG_RUNTIME_DIR=/run/user/1000 XDG_SESSION_TYPE=x11 \
    /home/robot/IsaacLab/isaaclab.sh -p plug_scene_viewer.py --gui --seconds 120
"""
from __future__ import annotations

import argparse
import sys
import time

# Line-buffer stdout so the orchestrator sees VIEWER_READY / RGB_MEAN / SAVED
# promptly even when piped.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Defaults (verified facts: these USDs exist and load on fr3-desktop-ts).
# ---------------------------------------------------------------------------
DEFAULT_PLUG_USD = "/home/robot/plug_insertion_sim/sim-scene/cn_two_pin_plug.usd"
DEFAULT_SOCKET_USD = "/home/robot/plug_insertion_sim/sim-scene/cn_three_pin_socket.usd"
DEFAULT_OUT_PNG = "/home/robot/plug_insertion_sim/sim-scene/_viewer_capture.png"

# Camera framing for the small plug/socket workspace (meters, Z-up).
# Parts are ~5-8 cm, so the eye is pulled in close (~0.5 m) to fill the frame.
CAM_EYE = (0.22, 0.17, 0.17)
CAM_TARGET = (0.0, 0.0, 0.03)

# Recommended part poses (Z-up world). Parts are small generated meshes:
#   plug  ~55 x 25 x 18 mm, socket panel ~80 x 80 x 25 mm.
# NOTE: passing translation= to the spawner OVERRIDES the USD's authored
# transform (verified in isaaclab.sim.spawners.from_files.spawn_from_usd),
# so these are absolute world translations.
PLUG_TRANSLATION = (0.0, 0.06, 0.03)
SOCKET_TRANSLATION = (0.0, -0.06, 0.02)

# The plug/socket USDs are authored Y-UP (`Y upAxis` in the USDC header) and
# the stage is Z-up. Per-object quats (w,x,y,z) chosen so each part's FEATURES
# face up toward the elevated camera (otherwise pins point into the ground and
# read as a plain box):
#   plug  : pins authored -Y; -90 deg about X maps -Y -> +Z  => pins point UP
#   socket: panel front authored +Y; +90 deg about X maps +Y -> +Z => slots UP
PLUG_ORIENT = (0.70710678, -0.70710678, 0.0, 0.0)
SOCKET_ORIENT = (0.70710678, 0.70710678, 0.0, 0.0)

# Lighting: the first capture (dome 1000 / sun 1400) blew out to mean 225 and
# hid the OmniPBR shading. Drop to a moderate level so mesh detail shows.
DOME_INTENSITY = 300.0
DOME_COLOR = (0.9, 0.9, 0.9)
SUN_INTENSITY = 600.0
SUN_COLOR = (1.0, 1.0, 0.95)
SUN_ANGLE_DEG = 0.5

# Local ground (the default GroundPlaneCfg fetches an Omniverse-S3 USD that is
# unreachable offline on this host). A flat cuboid gives a matte ground that
# the parts rest on and contrast against.
GROUND_SIZE = (1.0, 1.0, 0.02)
GROUND_COLOR = (0.25, 0.27, 0.30)


def parse_cli_args() -> argparse.Namespace:
    """Parse this script's own CLI args (separate from AppLauncher args)."""
    parser = argparse.ArgumentParser(
        description="Self-contained plug+socket IsaacLab viewer.",
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
    # parse_known_args so any stray AppLauncher flags do not crash us.
    args, _unknown = parser.parse_known_args()
    return args


def build_app_argv(gui: bool) -> list[str]:
    """Build the EXACT proven AppLauncher argv (droid phase3 recipe).

    --enable_cameras is MANDATORY even in headless or camera.data.output["rgb"]
    never populates. --kit_args disables the DLSS denoiser + DLSS-G, the
    R-PHASE0-6 recipe proven to render correctly (non-black) on this RTX 5080.
    """
    kit_args_str = (
        "--/rtx/denoiser/dlss/enabled=false "
        "--/rtx-transient/dlssg/enabled=false"
    )
    app_argv = [
        "--enable_cameras",
        "--rendering_mode", "quality",
        "--kit_args", kit_args_str,
    ]
    if not gui:
        # Headless capture: insert --headless at the front (droid idiom).
        app_argv.insert(0, "--headless")
    return app_argv


def main() -> int:
    cli_args = parse_cli_args()

    # -----------------------------------------------------------------------
    # STEP 0: launch the app FIRST, before any isaaclab.sim / .sensors import.
    # carb is only importable AFTER launch.
    # -----------------------------------------------------------------------
    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_argv = build_app_argv(gui=cli_args.gui)
    app_args = app_parser.parse_args(app_argv)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    # Heavy imports AFTER app launch only.
    import torch
    import numpy as np
    import pathlib

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sensors import Camera, CameraCfg

    # -----------------------------------------------------------------------
    # STEP 1: SimulationContext (+ RenderCfg with DLSS off, in-code twin of
    # the --kit_args flags; belt-and-suspenders, matching the proven runs).
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
        print(f"[viewer] RenderCfg unavailable ({exc}); relying on --kit_args.",
              flush=True)

    if render_cfg is not None:
        sim_cfg = SimulationCfg(dt=1.0 / 60.0, render_interval=1, render=render_cfg)
    else:
        sim_cfg = SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = SimulationContext(sim_cfg)

    # GUI viewport framing (no-op when --headless): aim at the workspace.
    sim.set_camera_view(eye=CAM_EYE, target=CAM_TARGET)

    # -----------------------------------------------------------------------
    # STEP 2 (PATH A): DIRECT SPAWN, no InteractiveScene.
    # USD auto-creates the /World ancestor; the spawners' @clone decorator
    # treats a non-regex parent path as a literal source path.
    # -----------------------------------------------------------------------
    # 2a. local ground (a flat cuboid). The default GroundPlaneCfg fetches an
    # Omniverse-S3 USD that is unreachable offline on this host (first run:
    # ground=FAILED), so spawn a matte cuboid the parts rest/contrast on.
    try:
        ground_cfg = sim_utils.CuboidCfg(
            size=GROUND_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GROUND_COLOR),
        )
        ground_cfg.func(
            "/World/Ground", ground_cfg,
            translation=(0.0, 0.0, -GROUND_SIZE[2] / 2.0),
        )
        print("[viewer] ground=OK /World/Ground (local cuboid)", flush=True)
    except Exception as exc:
        print(f"[viewer] ground=FAILED {exc}", flush=True)

    # 2b. lights — DomeLight (ambient floor) + DistantLight (directional sun).
    try:
        dome_cfg = sim_utils.DomeLightCfg(intensity=DOME_INTENSITY, color=DOME_COLOR)
        dome_cfg.func("/World/DomeLight", dome_cfg)
        print(f"[viewer] dome_light=OK intensity={DOME_INTENSITY}", flush=True)
    except Exception as exc:
        print(f"[viewer] dome_light=FAILED {exc}", flush=True)

    try:
        sun_cfg = sim_utils.DistantLightCfg(
            intensity=SUN_INTENSITY, color=SUN_COLOR, angle=SUN_ANGLE_DEG,
        )
        sun_cfg.func("/World/SunLight", sun_cfg)
        print(f"[viewer] sun_light=OK intensity={SUN_INTENSITY}", flush=True)
    except Exception as exc:
        print(f"[viewer] sun_light=FAILED {exc}", flush=True)

    # 2c. plug + socket USDs as static visual prims. Wrap each in try/except so
    # one missing/broken asset does not abort the whole preview.
    spawned = []
    plug_specs = [
        ("/World/Plug", cli_args.plug_usd, PLUG_TRANSLATION, PLUG_ORIENT),
        ("/World/Socket", cli_args.socket_usd, SOCKET_TRANSLATION, SOCKET_ORIENT),
    ]
    for prim_path, usd_path, translation, orientation in plug_specs:
        try:
            usd_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
            usd_cfg.func(prim_path, usd_cfg, translation=translation,
                         orientation=orientation)
            spawned.append(prim_path)
            print(f"[viewer] usd=OK {prim_path} <- {usd_path} "
                  f"translation={translation} orientation={orientation}", flush=True)
        except Exception as exc:
            print(f"[viewer] usd=FAILED {prim_path} <- {usd_path} : {exc}",
                  flush=True)

    # 2d. create the camera sensor. PLAIN /World path (NOT {ENV_REGEX_NS}) since
    # there is no InteractiveScene. clipping_range (0.01, 1e6) so the tiny parts
    # near z=0.03 with eye ~0.85 m are not clipped out.
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
        print("[viewer] camera=OK /World/RenderCamera", flush=True)
    except Exception as exc:
        print(f"[viewer] camera=FAILED {exc}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 3: play the simulator. MUST run before set_world_poses_from_view and
    # before reading camera.data.output (the timeline PLAY event fires the
    # SensorBase initialize callback that wires up the camera annotators).
    # -----------------------------------------------------------------------
    sim.reset()
    print(f"VIEWER_READY spawned={spawned} gui={cli_args.gui}", flush=True)

    # =======================================================================
    # GUI MODE: frame the workspace, keep the window alive for --seconds.
    # =======================================================================
    if cli_args.gui:
        sim.set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
        # Also aim the sensor camera so its prim is positioned sensibly.
        if camera is not None:
            try:
                eyes = torch.tensor([list(CAM_EYE)], device=sim.device,
                                    dtype=torch.float32)
                targets = torch.tensor([list(CAM_TARGET)], device=sim.device,
                                       dtype=torch.float32)
                camera.set_world_poses_from_view(eyes, targets)
            except Exception as exc:
                print(f"[viewer] gui set_world_poses_from_view FAILED {exc}",
                      flush=True)
        deadline = time.time() + cli_args.seconds
        print(f"[viewer] GUI alive for up to {cli_args.seconds:.0f}s "
              "(close the window to stop early).", flush=True)
        while simulation_app.is_running() and time.time() < deadline:
            try:
                sim.step(render=True)  # physics + render the viewport
                if camera is not None:
                    camera.update(dt=sim.get_physics_dt())
            except Exception as exc:
                print(f"[viewer] gui_loop_error={exc}", flush=True)
                break
        simulation_app.close()
        print("[viewer] closed", flush=True)
        return 0

    # =======================================================================
    # HEADLESS MODE (default): ground-truth RGB capture to PNG.
    # =======================================================================
    if camera is None:
        print("[viewer] ERROR: camera not created; cannot capture. "
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
        print(f"[viewer] set_world_poses_from_view FAILED {exc}", flush=True)

    # 2) STEP + RENDER several frames so the RTX denoiser / annotators converge.
    #    The first 1-2 frames can be black/partial (RTX warm-up). force_recompute
    #    on camera.update so the annotator buffers actually refresh per frame.
    n_frames = max(8, int(cli_args.settle_frames))
    for _ in range(n_frames):
        try:
            sim.step(render=True)
            camera.update(dt=sim.get_physics_dt(), force_recompute=True)
        except Exception as exc:
            print(f"[viewer] settle_step_error={exc}", flush=True)
            break

    # 2b) Final explicit Hydra/fabric flush + buffer refresh before reading.
    #     This is the proven droid idiom (sim_ctx.render() after the last step)
    #     and guards against reading a stale/partially-rendered final frame.
    try:
        sim.render()
        camera.update(dt=sim.get_physics_dt(), force_recompute=True)
    except Exception as exc:
        print(f"[viewer] final_flush_error={exc}", flush=True)

    # 3) READ rgb, drop alpha, to uint8 numpy.
    rgb_mean = 0.0
    out_path = pathlib.Path(cli_args.out)
    saved = False
    try:
        rgb_t = camera.data.output["rgb"][0]  # (H,W,4) torch uint8
        arr = rgb_t.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]  # drop alpha before mean / save
        # PIL needs C-contiguous; the alpha slice above is a view.
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # 5) MEAN INTENSITY (non-black sanity check). Expect ~40-160 here.
        rgb_mean = float(np.asarray(arr, dtype=np.float32).mean())

        # 4) SAVE PNG (PIL, fallback to imageio). Artifact-free path.
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
                print(f"[viewer] PNG save FAILED pil={pil_exc} "
                      f"imageio={iio_exc}", flush=True)

        print(f"[viewer] capture shape={arr.shape} dtype={arr.dtype}", flush=True)
    except Exception as exc:
        print(f"[viewer] rgb_read FAILED {exc}", flush=True)

    # Required machine-readable lines for the orchestrator.
    print(f"RGB_MEAN={rgb_mean:.2f}", flush=True)
    if saved:
        print(f"SAVED={out_path}", flush=True)
    else:
        print(f"SAVED=NONE (intended {out_path})", flush=True)
    if rgb_mean <= 1.0:
        print("[viewer] WARNING: mean ~0 -> still black. Add --settle-frames "
              "or raise DOME_INTENSITY.", flush=True)

    # 6) Clean exit. simulation_app.close() can HANG in headless Kit (observed
    #    on the first run: RC=124 from the outer timeout, AFTER the PNG saved),
    #    so skip it and os._exit(0) immediately — the PNG is already on disk.
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
