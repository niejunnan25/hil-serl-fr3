# Live Robot Gate

Owner: Codex primary, Human physical approval required

Nothing in this workspace authorizes live FR3 motion.

Before any live-control command:

- Human confirms E-stop supervision.
- Human confirms workspace clear.
- Human confirms robot idle or known initial pose.
- Human approves the exact script/command and maximum motion envelope.
- Codex records rollback and stop procedure.
- Codex verifies no DROID process or port will be touched.
- First smoke uses the smallest possible non-task motion or state-only probe,
  depending on the approved controller path.

Blocked until separate approval:

- `franka_server.py`
- `roscore`, `roslaunch`, `catkin_make`, `catkin build`, `colcon build`
- Desk/FCI unlock or control transitions
- real gripper commands
- real camera recording tied to robot execution
- Polymetis or any other live controller process
