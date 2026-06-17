# No-Motion Validation

Owner: Codex

Run:

```bash
source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh
python /home/robot/serl_projects/hil-serl-fr3/tools/no_motion_validate.py
```

The validator checks:

- pinned package versions and `pip check`;
- importability of `jax`, `flax`, `optax`, `gymnasium`, `serl_launcher`,
  `franka_env`, and `agentlace`;
- JAX device visibility;
- upstream example package import wiring;
- headless import behavior with `PYNPUT_BACKEND=dummy` recorded in the report;
- a localhost-only HIL robot HTTP contract mock on `127.0.0.1:5017`;
- `FrankaEnv(fake_env=True)` construction through the mock `/getstate` endpoint;
- rejection of command-like mock endpoints such as `/pose`;
- experiment folder shape for `plug_insertion`, `usb_pickup_insertion`, and
  `ram_insertion`.

Expected result:

```text
NO_MOTION_VALIDATION_OK
```

Failure policy:

- If port `5017` is occupied, stop and report the conflict.
- If JAX only sees CPU, record it as a training-performance issue, not a robot
  safety issue.
- If any import or package constraint fails, fix the isolated env only.
- Do not start any real robot server as a fallback.
