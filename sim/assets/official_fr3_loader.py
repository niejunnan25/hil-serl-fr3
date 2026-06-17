"""Build ArticulationCfg for the official Isaac Sim 5.1 Franka FR3 USD.

Per spec §3.1: v2.3.2 `isaaclab_assets` ships only `FRANKA_PANDA_CFG`,
没有 stock `FR3_CFG`. We wrap the FR3 USD by hand here.

USD path: local `droid/sim/assets/fr3.usd` (built via scripts/sim/build_fr3_usd.sh from
 franka_description repo; avoids Nucleus connectivity issues on fr3-desktop)
Joint prefix: `fr3_*`
Gripper joints: `fr3_finger_joint1` / `fr3_finger_joint2` (mimic; only need
to control joint1; finger range [0, 0.04] m each).

Fallback: stock `FRANKA_PANDA_CFG` (Panda joint prefix `panda_*`); recorded
in risk_log.md when used.

Note on ISAAC_NUCLEUS_DIR: imported from `isaaclab.utils.assets` which requires
`carb` and `omni.client` (Isaac Sim runtime). This module must only be imported
inside the Isaac Sim / Isaac Lab app context.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Canonical PD gains shared with sim/scenes/plug_scene.py (stable_sim profile).
from sim.assets.fr3_pd_gains import (
    ARM_STIFFNESS,
    ARM_DAMPING,
    ARM_EFFORT_LIMIT_SIM,
    GRIPPER_STIFFNESS,
    GRIPPER_DAMPING,
)

# Canonical repo-relative USD paths (single source of truth; no host paths).
# The repo ships sim/assets/fr3.usd and sim/assets/fr3_gripper_collision.usd,
# which are exactly the native + patched meshes this loader expects.
from .paths import (
    FR3_USD_PATH as _NATIVE_FR3_USD_PATH,
    FR3_GRIPPER_COLLISION_USD_PATH as _FR3_GRIPPER_COLLISION_USD_PATH,
)

# Default home pose — copied from spec §2.5 + handoff RobotEnv reset_joints
# RobotEnv.reset_joints  = [0, 0, 0, -π/2, 0, π/2, 0]
# (per the real RobotEnv.reset_joints array; see fr3-desktop handoff)
_FR3_DEFAULT_HOME = {
    "fr3_joint1": 0.0,
    "fr3_joint2": 0.0,
    "fr3_joint3": 0.0,
    "fr3_joint4": -1.5707963,   # -π/2
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.5707963,    # +π/2
    "fr3_joint7": 0.0,
    "fr3_finger_joint.*": 0.04,  # open
}

_FR3_GRIPPER_PROXY_LOCAL_CENTER = (0.0, 0.013, 0.027)
_FR3_GRIPPER_PROXY_LOCAL_SIZE = (0.048, 0.048, 0.056)
_FR3_GRIPPER_PROXY_LINKS = ("fr3_leftfinger", "fr3_rightfinger")
_FR3_GRIPPER_PROXY_STATIC_FRICTION = 2.2
_FR3_GRIPPER_PROXY_DYNAMIC_FRICTION = 1.8


def _create_or_update_gripper_physics_material(stage, robot_prim_path: str, PhysxSchema):
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Scope.Define(stage, f"{robot_prim_path}/Materials")
    material = UsdShade.Material.Define(
        stage,
        f"{robot_prim_path}/Materials/codex_gripper_high_friction_physics",
    )
    material_prim = material.GetPrim()
    UsdPhysics.MaterialAPI.Apply(material_prim)
    material_prim.CreateAttribute(
        "physics:staticFriction",
        Sdf.ValueTypeNames.Float,
    ).Set(_FR3_GRIPPER_PROXY_STATIC_FRICTION)
    material_prim.CreateAttribute(
        "physics:dynamicFriction",
        Sdf.ValueTypeNames.Float,
    ).Set(_FR3_GRIPPER_PROXY_DYNAMIC_FRICTION)
    material_prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(0.0)

    if PhysxSchema is not None:
        PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        material_prim.CreateAttribute(
            "physxMaterial:frictionCombineMode",
            Sdf.ValueTypeNames.Token,
        ).Set("max")
        material_prim.CreateAttribute(
            "physxMaterial:restitutionCombineMode",
            Sdf.ValueTypeNames.Token,
        ).Set("min")
    return material


def _author_gripper_collision_proxies(
    stage,
    robot_prim_path: str,
    *,
    allow_create: bool,
) -> list[str]:
    from pxr import Gf, UsdGeom, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:  # pragma: no cover - depends on Isaac Sim runtime.
        PhysxSchema = None

    proxy_paths: list[str] = []
    physics_material = None
    if allow_create:
        physics_material = _create_or_update_gripper_physics_material(
            stage,
            robot_prim_path,
            PhysxSchema,
        )

    for link_name in _FR3_GRIPPER_PROXY_LINKS:
        collisions_path = f"{robot_prim_path}/{link_name}/collisions"
        collisions_prim = stage.GetPrimAtPath(collisions_path)
        if not collisions_prim or not collisions_prim.IsValid():
            continue
        if allow_create and collisions_prim.IsInstance():
            collisions_prim.SetInstanceable(False)

        proxy_path = f"{collisions_path}/codex_finger_collision_proxy"
        existing = stage.GetPrimAtPath(proxy_path)
        if existing and existing.IsValid():
            proxy_paths.append(proxy_path)
            continue
        if not allow_create:
            continue

        cube = UsdGeom.Cube.Define(stage, proxy_path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube.GetPrim())
        xform.SetTranslate(Gf.Vec3d(*_FR3_GRIPPER_PROXY_LOCAL_CENTER))
        xform.SetScale(Gf.Vec3f(*_FR3_GRIPPER_PROXY_LOCAL_SIZE))
        imageable = UsdGeom.Imageable(cube.GetPrim())
        imageable.CreatePurposeAttr(UsdGeom.Tokens.proxy)
        imageable.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision_api.CreateCollisionEnabledAttr(True)
        if PhysxSchema is not None:
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            physx_collision.CreateContactOffsetAttr(0.005)
            physx_collision.CreateRestOffsetAttr(0.0)
        if physics_material is not None:
            from pxr import UsdShade

            material_api = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
            material_api.Bind(
                physics_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        proxy_paths.append(proxy_path)
    return proxy_paths


def _ensure_fr3_usd_with_gripper_collision() -> str:
    """Create a patched FR3 USD with finger colliders when pxr is available."""
    import os

    if not os.path.exists(_NATIVE_FR3_USD_PATH):
        return _NATIVE_FR3_USD_PATH
    if (
        os.path.exists(_FR3_GRIPPER_COLLISION_USD_PATH)
        and os.path.getmtime(_FR3_GRIPPER_COLLISION_USD_PATH) >= os.path.getmtime(_NATIVE_FR3_USD_PATH)
    ):
        return _FR3_GRIPPER_COLLISION_USD_PATH

    try:
        from pxr import Usd
    except ImportError:
        return _NATIVE_FR3_USD_PATH

    source_stage = Usd.Stage.Open(_NATIVE_FR3_USD_PATH)
    if source_stage is None:
        return _NATIVE_FR3_USD_PATH

    temp_usd_path = f"{_FR3_GRIPPER_COLLISION_USD_PATH}.tmp.usd"
    try:
        flat_layer = source_stage.Flatten()
        flat_layer.Export(temp_usd_path)
        stage = Usd.Stage.Open(temp_usd_path)
    except Exception:
        return _NATIVE_FR3_USD_PATH
    if stage is None:
        return _NATIVE_FR3_USD_PATH

    try:
        proxies = _author_gripper_collision_proxies(stage, "/fr3", allow_create=True)
    except Exception:
        return _NATIVE_FR3_USD_PATH
    if len(proxies) != len(_FR3_GRIPPER_PROXY_LINKS):
        return _NATIVE_FR3_USD_PATH
    stage.GetRootLayer().Export(_FR3_GRIPPER_COLLISION_USD_PATH)
    try:
        os.remove(temp_usd_path)
    except OSError:
        pass
    return _FR3_GRIPPER_COLLISION_USD_PATH


def _build_native_fr3_cfg(
    prim_path: str,
    base_pos: tuple[float, float, float],
    fixed_base: bool,
    actuator_profile: str,
) -> ArticulationCfg:
    """Native FR3 cfg using the official Isaac Sim 5.1 USD.

    Per spec §2.5 hard limits抄自 franka_hardware_left.yaml:
      Kq  = [40, 30, 50, 25, 35, 25, 10]
      Kqd = [4, 6, 5, 5, 3, 2, 1]
    """
    if actuator_profile == "stable_sim":
        disable_gravity = True
        arm_actuator = ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            effort_limit_sim=dict(ARM_EFFORT_LIMIT_SIM),
            stiffness=ARM_STIFFNESS,
            damping=ARM_DAMPING,
        )
    elif actuator_profile == "hardware_kq":
        disable_gravity = False
        arm_actuator = ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            stiffness={
                "fr3_joint1": 40.0, "fr3_joint2": 30.0, "fr3_joint3": 50.0,
                "fr3_joint4": 25.0, "fr3_joint5": 35.0, "fr3_joint6": 25.0,
                "fr3_joint7": 10.0,
            },
            damping={
                "fr3_joint1": 4.0, "fr3_joint2": 6.0, "fr3_joint3": 5.0,
                "fr3_joint4": 5.0, "fr3_joint5": 3.0, "fr3_joint6": 2.0,
                "fr3_joint7": 1.0,
            },
        )
    else:
        raise ValueError(
            f"unknown actuator_profile={actuator_profile!r}; "
            "expected 'stable_sim' or 'hardware_kq'"
        )

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ensure_fr3_usd_with_gripper_collision(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                fix_root_link=fixed_base,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base_pos,
            joint_pos=_FR3_DEFAULT_HOME,
        ),
        actuators={
            "arm": arm_actuator,
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                stiffness=GRIPPER_STIFFNESS,
                damping=GRIPPER_DAMPING,
            ),
        },
    )


def _build_panda_fallback_cfg(
    prim_path: str,
    base_pos: tuple[float, float, float],
    fixed_base: bool,
) -> ArticulationCfg:
    """Fallback to stock FRANKA_PANDA_CFG when native FR3 USD is unavailable.

    Recorded in risk_log.md when invoked. Per spec §3.1 Fallback 1.
    """
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # type: ignore[import-not-found]
    cfg = FRANKA_PANDA_CFG.replace(prim_path=prim_path)
    cfg.init_state.pos = base_pos
    if cfg.spawn is not None and cfg.spawn.articulation_props is not None:
        cfg.spawn.articulation_props.fix_root_link = fixed_base
    return cfg


def build_fr3_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    use_panda_fallback: bool = False,
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_base: bool = True,
    actuator_profile: str = "stable_sim",
) -> ArticulationCfg:
    """Construct the FR3 ArticulationCfg.

    Args:
        prim_path: USD prim path under the env namespace.
        use_panda_fallback: if True, return stock FRANKA_PANDA_CFG instead.
            ONLY use when the FR3 USD path is unavailable; record in risk_log.md.
        base_pos: FR3 root pose in the sim world. Defaults to the tabletop
            mounting plane used by the primitive scene.
        fixed_base: whether to create/enable the fixed joint from world to
            the articulation root. Real FR3 setup is bolted to the table.
        actuator_profile: "stable_sim" uses high Isaac PD gains so the arm
            does not sag or lie down during sim control; it follows Isaac Lab's
            high-PD Franka convention by disabling robot-link gravity.
            "hardware_kq" preserves the historical franka_hardware_left.yaml
            gains and is useful only when intentionally studying low-gain sim
            behavior.

    Returns:
        ArticulationCfg ready for InteractiveScene.

    Raises:
        ImportError if isaaclab/isaaclab_assets is not installed.
    """
    if use_panda_fallback:
        return _build_panda_fallback_cfg(prim_path, base_pos, fixed_base)
    return _build_native_fr3_cfg(prim_path, base_pos, fixed_base, actuator_profile)


def ensure_fr3_gripper_collision_proxies(
    robot_prim_path: str = "/World/envs/env_0/Fixed_Robot",
) -> list[str]:
    """Add missing FR3 finger collision cuboids to an already spawned stage.

    The generated FR3 USD currently contains rigid finger links but empty
    ``/collisions`` children. That makes policy-driven gripper closure visible
    without giving PhysX a finger surface to push tabletop objects. These
    invisible cuboids are sized from the finger visual local bounds and live
    under each finger's rigid body, so they follow the prismatic finger joints.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("no active USD stage; call after Isaac Lab creates the scene")
    return _author_gripper_collision_proxies(stage, robot_prim_path, allow_create=False)


__all__ = ["build_fr3_cfg", "ensure_fr3_gripper_collision_proxies"]
