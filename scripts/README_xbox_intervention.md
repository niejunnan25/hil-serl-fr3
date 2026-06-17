# xbox_intervention.py — copies & single source of truth

There are three DISTINCT `xbox_intervention.py` contents in this repo. They are
NOT redundant duplicates and MUST NOT be textually merged: they implement two
different wrapper APIs for two different runtimes.

## The three copies

| Copy | Path | API | Status / canonical for |
|------|------|-----|------------------------|
| A (Phase A, hub-based) | `scripts/xbox_intervention.py` | `XboxIntervention(env, hub, ...)`; exports `DEADZONE=0.15`, `SCALE_FINE=0.3`, `SCALE_COARSE=1.0`, `GRIPPER_IDX`; 3mm/step cap inside wrapper | CANONICAL for local teleop tooling. Imported by `scripts/teleop_arbiter.py`, `scripts/record_hybrid_demos.py`, `scripts/relative_teleop.py`; pinned by `tests/test_xbox_intervention_contract.py` and `tests/test_teleop_arbiter.py`. |
| B (evidence yaw-fix) | `.planning/2026-06-13-v221-hybrid-teleop/evidence/orientation-yaw-fix-20260616/src/serl_projects/hil-serl-fr3/scripts/xbox_intervention.py` | `XboxIntervention(env, gripper_action_value=0.0)` + `HoldGripperWrapper`; `XBOX_ROT_STEP=0.05`, `XBOX_MAX_STEP=0.02`, `XBOX_INSERT_REACH=0.01`; rot uses `action_scale[1]` | HISTORICAL evidence snapshot. Superseded by copy C. Do not deploy. |
| C (runtime-patches online-stability) | `.planning/2026-06-13-v221-hybrid-teleop/runtime-patches/20260617-online-stability/scripts/xbox_intervention.py` | same API as B, but bounded constants `XBOX_ROT_STEP=0.030`, `XBOX_MAX_STEP=0.008`, `XBOX_INSERT_REACH=0.004`; rot uses `action_scale[3]`; richer `[XboxDBG] source=rb ...` log | CANONICAL for the FR3-desktop ACTOR runtime. This is the copy the live actor runs. Pinned by `tests/test_online_training_runtime_patch.py` and (after Item 20) `tests/test_actor_xbox_intervention.py`. |

A fourth file, `.planning/.../evidence/orientation-yaw-fix-20260616/src/hilserl-fr3/scripts/xbox_intervention.py`, is byte-identical to copy A (an evidence snapshot of the local script) and carries no independent meaning.

## Why they are not merged

- Copy A is hub-injected (a `TeleopDeviceHub` is passed in) and is consumed by the local import graph (`import xbox_intervention`). Copy C lazily creates its own pygame hub and is deployed headless to the robot desktop.
- Copy A exports the calibration constants (`DEADZONE`, `SCALE_FINE`) that `record_hybrid_demos.py` imports; copy C exports `normalize_action`/`ACTION_SCALE`/`HoldGripperWrapper` instead. Their public surfaces do not overlap.
- Copy C lives under `.planning/.../runtime-patches/` as a staged deploy artifact. Editing it is HIGH risk (it changes what the live actor runs) and is out of scope for a refactor — change it only via the deliberate runtime-patch process with explicit approval.

## Rule of thumb

- Touching local teleop / demo collection -> edit copy A (`scripts/xbox_intervention.py`).
- Touching the actor's takeover behavior -> change copy C through the runtime-patch deploy process, and update `tests/test_online_training_runtime_patch.py` constants in lockstep.
- Copy B is frozen evidence; do not edit or import it.
