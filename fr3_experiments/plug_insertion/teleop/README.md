# Plug-ZED Xbox Teleop

This package contains no-motion Xbox teleoperation and demo-recording
scaffolds for the isolated non-DROID HIL-SERL FR3 workspace.

Allowed by default:

```text
xbox input probe
SDL controller backend import/probe
action preview
mock HIL-SERL demo transition writing
state-only actor dry run
LeRobot-compatible sidecar export
demo collection preflight
schema validation
```

Not allowed by default:

```text
live FR3 motion
gripper command
DROID/OpenPI/RobotEnv fallback
policy rollout
real demo collection
remote training
```

Run the validator from the workspace root:

```bash
source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh
python -m fr3_experiments.plug_insertion.teleop.validate_xbox_teleop \
  --artifact-root artifacts/plug_zed_xbox_teleop_no_motion
```

Physical controller probe before real teleop:

```bash
source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh
python -m fr3_experiments.plug_insertion.teleop.xbox_probe \
  --backend sdl \
  --duration-sec 120
```

Physical-controller preflight before generating a live packet:

```bash
source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh
python -m fr3_experiments.plug_insertion.teleop.xbox_demo_collection_preflight \
  --artifact-root artifacts/plug_zed_xbox_demo_collection_preflight/physical_probe \
  --probe-duration-sec 120 \
  --require-controller
```

Real FR3 motion and real demo collection still require the current task-local
live execution packet and Human physical facts. The Xbox layer only emits
bounded normalized 7D HIL-SERL actions; it is never a raw joystick-to-FCI path.
