# RoboMeter episode 间隔计奖：开发验证记录

日期：2026-09-21。分支：`codex/robometer-episode-gap`。开发前基线：`0fea2b4b4db97bced20ccb4677c949cdd0117b99`。

工作目录：`/home/robot/serl_projects/hil-serl-fr3/artifacts/worktrees/robometer-episode-gap`。
本地文档与源码镜像：`/Users/n/Documents/ChatGPT/HIL-SERL/development/robometer-episode-gap`。

## 结论

episode 暂存、暂停屏障、RoboMeter 原生推理适配器、因果前缀批处理、完整入库回执、demo 奖励缓存、恢复、工作台状态已实现。相关运行逻辑通过 CPU 测试，包含真实本地 HTTP 和 Agentlace REQ/REP 通信，以及原生 replay 的插入和采样。

**最终可运行回归范围：1,089 passed，2 deselected，79 warnings，93.87 秒。** 下方明确列出两个缺外部源码的模块及两条已复现的基线失败；不把这次结果称作无条件全库通过。

功能测试文件包含 33 个测试用例；最后修订后的功能、原生 wrapper 和录制流程定向回归共 55 项通过（6.86 秒）。工作台 JavaScript 通过 `node --check`；`git diff --check` 通过；`scripts/manage_upstream.py check` 确认 hil-serl、serl、agentlace、serl_franka_controllers 四个依赖均与独立 worktree 的锁定版本/补丁一致。

本次没有启动真实机器人、训练任务或 4B 奖励模型，没有修改正在使用的 main 工作区配置，也没有停止或重启现有服务。main 自身已有未提交修改，保持其原有状态；开发提交只进入本分支。

## 关键验证

| 场景 | 验证内容 |
| --- | --- |
| 异步重叠 | `run_episodes` 真实循环中，奖励模型被人为阻塞时复位仍能完成；奖励入库确认前不发下一轮动作；最后一条正常排空 |
| 暂停屏障 | 执行真实 Learner 函数，以模拟优化器验证完整 critic/actor 更新组结束并执行完成同步后才确认暂停；暂停时接收线程仍工作；不跨间隔预取 |
| 观测刷新 | 使用原生 RelativeFrame、Quat2Euler、SERLObs、Chunking 和外层统计 wrapper；等待结束后读取最新状态/图像，保留复位原点、更新当前变换、不下发动作或累计环境步数 |
| 状态故障 | 新鲜状态读取失败时停止该读路径，不发送动作；Stop 在奖励等待时关闭设备流程并保留有标签数据 |
| 因果批处理 | 实际 HTTP NPZ 请求内各 sample 只包含该时刻及以前图像；batch size 改变不改变模拟结果；拒绝错误模型 ID、非法分数与顺序 |
| 入库一致性 | 真实 Agentlace socket 完成 begin/commit/release；旧 streaming 路由拒绝非空数据；丢失 commit/release 回执后重试不重复插入 |
| 部分故障与重启 | 第二个 buffer 写入失败后锁住训练；新 Learner 从完整日志重建；Actor 恢复 pending 不重发历史机器人命令 |
| 去重 | 相同原始步骤换 episode 身份再次提交，在写事务日志之前被拒绝 |
| 数据与奖励 | 原生 replay 抽样核对两路图像、状态、动作、奖励和 mask；负奖励失败不误判成功；奖励版本/像素哈希/任务原始数据不符拒绝 |
| 示教缓存 | 独立缓存准备、加载和版本拒绝，逐条核对原始 seed 图像与动作；不修改原示教集 |
| 配置和工作台 | 默认 sparse；Actor/Learner 的奖励契约和 SAC discount 一致；错误配置启动前失败；奖励等待阶段可 Stop |

## 回归命令与排除原因

在上述远端 worktree 执行：

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:upstream/agentlace:upstream/hil-serl/serl_launcher:upstream/hil-serl/serl_robot_infra:upstream/hil-serl/examples \
/home/robot/miniconda3/envs/hilserl-fr3/bin/python -m pytest tests -q \
  --ignore=tests/test_controller_freshness.py \
  --ignore=tests/test_controller_lifecycle_helper.py \
  -k "not test_sample_reset_pose_respects_random_flag and not test_current_profile_uses_the_intended_demo_set" \
  --disable-warnings --maxfail=3
```

未运行的两个模块依赖测试写死的外部路径，当前机器没有该来源。没有用不相干源码替代：

- `test_controller_freshness.py`：`/Users/tacyvan/Code/hilserl-reset-gripper-20260912/controller-patch/robot_servers`，可通过 `HILSERL_CONTROLLER_PATCH_ROOT` 指定真实对应来源。
- `test_controller_lifecycle_helper.py`：`/Users/tacyvan/Code/hilserl-controller-recovery-20260912/controller_lifecycle.py`。

以下两个断言在未修改的 `0fea2b4` 源码归档内也失败，复现日志保留；本次不修改无关的现场 reset 或示教配置来迎合旧断言：

- `test_formal_training_mainline.py::test_sample_reset_pose_respects_random_flag`：测试期待随机 yaw，基线当前已经使用固定竖直姿态，yaw 不随机。
- `test_formal_training_mainline.py::test_current_profile_uses_the_intended_demo_set`：测试期待 `demos/fixed_xyz_roi160_v1_20260914`，基线当前配置是 `demos/fixed_xyz_front_v2_command20_20260921`。

较早的整库运行还发现配置入口过早引入 NumPy，本次已通过延迟导入修复，相关轻量 CLI 回归包含在最后通过结果中。

日志保存在远端 worktree 根目录：`artifacts-reward-final-tests.log`、`artifacts-reward-refresh-tests.log`、`artifacts-reward-baseline.log`；本地副本位于 `artifacts/reward-verification/`。日志是开发证据，不进入训练数据集。

## 尚待部署验证

这次没有加载真实 RoboMeter 权重。模型 backend 适配来自已核查的 SEAL-0 实现，但本次 socket 测试使用可控的模拟进度模型。因此尚不能保证特定 GPU 上的一条 190 步 episode 能在 5 秒或 7 秒内完成计奖，也没有测同卡 JAX/PyTorch 显存或真实网络传输性能。

真实模型接入时按 [使用说明](robometer-episode-gap.md) 固定模型内容 ID、单独生成相同版本的 demo 奖励缓存，并使用独立配置/端口/输出目录。worktree 只隔离代码和文件，使用同一机器人仍需安排设备交接。本功能默认关闭，没有给当前运行实验热切换奖励。

“复位 5 秒、奖励连同提交 7 秒、额外等待约 2 秒”是解释调度的示例。额外等待、人工准备、复位均计墙钟；复位和等待不产生训练 transition。该功能未新增 30 分钟自动截止、连续十次成功自动停止，或跨 Actor 进程独立录像。
