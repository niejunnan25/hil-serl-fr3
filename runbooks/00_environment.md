# Environment

Owner: Codex

Root:

```bash
/home/robot/serl_projects/hil-serl-fr3
```

Activate:

```bash
source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh
```

Pinned upstream checkouts:

- `upstream/hil-serl`: `c32939bccb65f3b8c43a9f9add3d322d4ab0264a`
- `upstream/serl`: `1fa2af7496be042e43d36120a7c8d1b71f1bfb59`
- `upstream/serl_franka_controllers`: `1f140ef0d8e3fc443569c193d3ede1856e50d521`
- `upstream/agentlace`: `cf2c337c5e3694cdbfc14831b239bd657bc4894d`

Python environment:

- conda env: `hilserl-fr3`
- path: `/home/robot/miniconda3/envs/hilserl-fr3`
- constraints: `env/hilserl_constraints.txt`
- full installed lock snapshot: `env/pip_freeze.lock`

JAX CUDA note:

- The env sets `CUDA_ROOT` to its own `nvidia/cuda_nvcc` wheel path during
  activation.
- This is required because JAX can otherwise crash while importing the
  namespace-style `nvidia.cuda_nvcc` package (first observed on the older
  `0.4.35` line; the env now pins `0.6.2` / jaxlib `0.6.2` on both the desktop
  actor env and the zktitan learner env, and retains the workaround).
- Verified expected no-motion result: `jax.devices()` includes `CudaDevice(id=0)`.

HIL robot infra import note:

- Upstream `serl_robot_infra/setup.py` exposes `robot_servers` through editable
  install, but `franka_env/` is a namespace-style directory without a top-level
  `__init__.py`.
- The activation script therefore adds
  `upstream/hil-serl/serl_robot_infra` to `PYTHONPATH`.
- This is an overlay path fix, not an upstream source change.

Non-interference rules:

- Do not write under `/home/robot/droid`.
- Do not reuse DROID conda envs.
- Do not source global ROS workspaces from this setup.
- Do not bind non-localhost servers during no-motion validation.
- Do not start `franka_server.py`, ROS, Polymetis, Franka controllers, Desk/FCI,
  or any real gripper/camera/motion path from this workspace without a separate
  live robot approval gate.
