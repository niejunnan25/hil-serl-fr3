# FR3 Control Bridge Options

Owner: Codex primary, Human approval required before live control

The HIL-SERL algorithm layer is now separated from DROID. Real FR3 control still
needs a lower-layer bridge decision.

Option A: Mock-first HIL contract

- current path;
- validates HIL env shape and task modularity without hardware;
- no ROS, DROID, Desk/FCI, or gripper command startup.

Option B: Direct non-DROID FR3 bridge

- build a small HIL-compatible Gym/env adapter around a direct FR3 control
  process;
- bounded API: `get_state`, `reset_to_joint_pose`, bounded Cartesian or joint
  delta `step`, gripper move/open/close, force/torque state, safety state;
- good modularity, more controller engineering work.

Option C: Upstream-compatible isolated ROS/controller bridge

- use `serl_franka_controllers` and upstream `franka_server.py` semantics;
- current host has `libfranka` but no visible ROS1/catkin toolchain;
- do not install ROS directly on host as a default first step;
- evaluate a separate Ubuntu 20.04/Noetic container or otherwise isolated ROS
  workspace only after a new approval gate.

Recommended sequence:

1. Finish no-motion validation.
2. Decide whether fine insertion first needs force/torque-rich direct control
   or upstream ROS compatibility.
3. Produce a separate live-control plan with rollback, E-stop, workspace-clear,
   initial-pose, controller-limit, and one-step smoke gates.
