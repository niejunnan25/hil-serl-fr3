# HIL-SERL 插插头

日常训练、评估、Xbox 示教、回放和导出统一使用 `bin/hil-serl`。桌面工作台用 `bin/hil-serl console --background --open` 打开。

代码版本、GitHub 同步与上游补丁恢复见 [Git 管理说明](docs/git-management.md)。新克隆先运行 `python3 scripts/manage_upstream.py restore`；示教、模型和录像需单独恢复。

详见 [工作台使用与数据契约](docs/unified-workbench.md) 和 [真机运行门](runbooks/03_live_robot_gate.md)。默认每条上限 190 步；人工 0/1 最终裁决；全程双路视频与有效 episode 分开归档；新训练使用新目录。每轮最多保留 5 个模型 checkpoint：最新续训点必留，其余按独立评估成绩优先、无成绩按较新版本补齐，详见 [checkpoint 保留规则](docs/checkpoint-retention.md)。

Actor 等待、退出或反馈中断时 Learner 自动暂停，采集和在线数据恢复后接着原 step 更新。可单独手动暂停 Learner；停止保存后用“续训 Learner”恢复本轮完整训练状态。运行中的 Actor 遇到状态接口 503 会保留进程，等待新的复位确认；底层恢复与复位仍由明确按钮操作触发。

新建训练默认使用 `fixed-xyz-v1`：只学习 XYZ，旋转和夹爪继续锁定；示范使用从原始记录核验的 13 条全人工成功轨迹。Policy 两路画面采用 `insert-roi160-v1`，从 1280×720 原帧按任务区域裁切后输入 160×160；详情见 [相机裁切与输入契约](docs/camera-input.md)。旧 checkpoint 的续训与评估按原 run 配置选择旧模型。版本边界、奖励和训练指标见 [XYZ 训练说明](docs/fixed-xyz-training.md)。

以下为保留的项目背景与底层组件资料；日常启动方式以上述统一入口为准。

---

# HIL-SERL FR3 Desktop Workspace

This workspace is an isolated HIL-SERL/SERL environment for FR3 experiments on
`fr3-desktop-ts`.

It is intentionally separate from DROID:

- root: `/home/robot/serl_projects/hil-serl-fr3`
- conda env: `/home/robot/miniconda3/envs/hilserl-fr3`
- robot/mock port: `5017`
- optional ROS namespace port for later design: `11317`
- learner/actor port base: `54817`

Current status:

- no live robot control is enabled here;
- no DROID files, envs, processes, or ports are used;
- upstream repos are pinned under `upstream/`;
- project-owned overlays live outside upstream repos.

Use:

```bash
source /home/robot/serl_projects/hil-serl-fr3/env/activation.sh
python /home/robot/serl_projects/hil-serl-fr3/tools/no_motion_validate.py
```

Plug no-motion runner dry-runs:

```bash
/home/robot/serl_projects/hil-serl-fr3/env/headless_runner.sh \
  python /home/robot/serl_projects/hil-serl-fr3/tools/run_plug_learner.py --dry-run --headless
/home/robot/serl_projects/hil-serl-fr3/env/headless_runner.sh \
  python /home/robot/serl_projects/hil-serl-fr3/tools/run_plug_actor.py --dry-run --headless
```

These wrappers import `fr3_experiments.plug_insertion.config.TrainConfig`
directly. They do not require upstream `CONFIG_MAPPING["plug_insertion"]`, and
dry-run mode does not call `env.reset()`, `env.step()`, FCI, cameras, policy
server, or gripper commands. The `headless_runner.sh` helper is no-motion only;
live actor launch still requires a separate intervention-device decision and
approval gate.

Live FR3 motion, ROS controller launch, Desk/FCI actions, gripper commands, real
camera recording, or hardware server startup require a separate approval gate.

---

## sim/ Module Structure (IsaacLab Integration)

```
sim/
├── assets/
│   ├── fr3.usd                          # FR3 robot USD (1452 bytes)
│   ├── fr3_gripper_collision.usd        # Gripper collision USD
│   ├── official_fr3_loader.py           # ArticulationCfg builder
│   └── lighting.py                      # Scene lighting
├── kinematics/
│   └── fr3_fk.py                        # Pure-Python FK (DH params)
├── data/
│   ├── gello_replay.py                  # GELLO replay
│   ├── sim_replay_pipeline.py           # Sim replay pipeline
│   ├── plug_reward_labeler.py           # Reward labeling
│   └── demo_relative_replay.py          # Demo relative replay
├── obs/
│   ├── camera_wrapper.py                # Camera observation wrapper
│   └── obs_dict.py                      # Observation dictionary
├── safety/
│   ├── feasibility_checker.py           # Feasibility checks
│   └── runtime_check.py                 # Runtime safety checks
├── scenes/
│   └── __init__.py
└── transforms/
    └── delta_actions.py                 # Delta action transforms
```

### Verification Status (2026-06-10)

| Component | Status | Details |
|-----------|--------|---------|
| fr3_fk.py | PASS | Pure-Python FK; home pose [0.5545, -0.0, 0.5211], det(R)=1.0 |
| official_fr3_loader.py | PASS (syntax) | AST OK (21 top-level nodes); requires Isaac Sim runtime for pxr |
| fr3.usd | PASS | Exists, 1452 bytes |
| Package structure | PASS | All 7 subpackages have __init__.py |
| GELLO pipeline | PASS | fk_converter.forward_kinematics imports OK (hilserl-fr3 env) |

> **Note**: `official_fr3_loader.py` imports `isaaclab.sim` and `pxr` which require
> Isaac Sim runtime (SimulationApp context). It cannot be imported in a bare
> Python shell — only inside an IsaacLab simulation script or via `isaaclab.sh -p`.
