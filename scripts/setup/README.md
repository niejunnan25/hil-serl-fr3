# FR3 Setup Guide — Phase 1

Setting up the SERL/HIL-SERL stack for the Franka FR3 robot.

## Prerequisites

- fr3-desktop-ts (Ubuntu 22.04, realtime kernel)
- FR3 at 172.16.0.2, control host at 172.16.0.4
- libfranka >= 0.13.0 (FR3 requires FI3)
- conda env hilserl-fr3 (Python 3.10 + JAX)

## Steps

### Step 1: Check environment

```
ssh fr3-desktop-ts
bash scripts/setup/01_check_libfranka.sh
```

This checks:
- libfranka version >= 0.13.0
- Robot network reachability
- ROS Noetic availability
- franka_ros and serl_franka_controllers presence
- conda env hilserl-fr3 and JAX

### Step 2: Adapt controllers

```
bash scripts/setup/02_adapt_serl_controllers.sh
```

This clones serl_franka_controllers and patches:
- arm_id: panda -> fr3
- Joint names: panda_joint1-7 -> fr3_joint1-7
- Link names: panda_link0-8 -> fr3_link0-8

If ROS Noetic is available, it will also catkin build.

### Step 3: Start franka server

```
bash scripts/setup/03_start_franka_server.sh start
```

Commands: start | stop | restart | status | logs | health

Default port: 5000 (upstream SERL default, matches config.py)

## Two control paths

**ROS path** (upstream compatible):
Needs ROS Noetic + franka_ros + catkin build of serl_franka_controllers

**Direct libfranka bridge** (recommended):
Non-ROS, lower coupling, lives in fr3_hil_bridge/. Recommended for this project.

## Important notes

- Do NOT run on real robot without safety approval
- Port 5000 is default (5000=server, override with SERL_FRANKA_PORT)
- These scripts do not modify DROID or other envs
- Always have E-stop reachable when testing

## Troubleshooting

| Problem | Solution |
|---------|----------|
| libfranka < 0.13.0 | Install 0.14.1+ via dpkg |
| Robot unreachable | Check cable, IP config on enp6s0 |
| ROS not found | Use direct bridge (non-ROS) |
| Port 5000 occupied | Check with ss -ltnp grep 5000 |
| franka_server won't start | Check tail -f artifacts/logs/franka_server.log |

## File index

- `01_check_libfranka.sh` — environment check
- `02_adapt_serl_controllers.sh` — clone + patch for FR3
- `03_start_franka_server.sh` — server lifecycle
- `README.md` — this file
