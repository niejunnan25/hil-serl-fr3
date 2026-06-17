# FR3 HIL Bridge

This directory is the project-owned bridge layer between HIL-SERL and future
FR3 control backends.

Current contents:

- `mock_server/`: localhost-only no-motion contract mock for validation.
- `direct_franka_design/`: reserved for a future non-DROID direct bridge design.
- `ros_controller_design/`: reserved for a future isolated upstream-compatible
  ROS/controller design.

Do not place upstream source modifications here. Keep backend experiments as
small adapters with explicit safety gates.
