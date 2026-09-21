# RoboMeter 与当前 FR3 工作区的版本整合

完成日期：2026-09-22。工作分支：`codex/robometer-episode-gap`。

## 来源与 Git 管理

先读取指定对话 `01a0c1e1-24e8-7f51-941c-c0eae701947b`（《连接 pku5080 检查状态》），再以实际远端工作区为准核对代码。对话中最后的修复包括降低拔出高度、单次手柄提醒、复位稳定性校验、两路新帧校验及异常停控。

读取时 main 为 `ed67521`，有 31 个修改或未跟踪的源码、配置、测试和文档文件；GitHub origin/main 没有更新。当前现场还包含统一 6 mm 到位容差及轴向误差 clip 降到 8 mm 的变更，对应弹簧项上限为 20 N；它们比原对话的部分文字描述更新。

将现场文件原样保存为 main 提交 **`01ed79999f8b7d136c72a24d9ab8a38670e05655`**。提交前后逐文件 SHA-256 相同，main 工作区干净；没有 stash、删除、覆盖这些未提交文件，也没有停止/重启服务或执行机器人动作。

备份与证据：`/home/robot/serl_projects/hil-serl-fr3/artifacts/git-sync/20260921T155252Z/`，包括原始 index、状态、二进制 diff、31 文件源码 tar、前后 SHA 与提交回执。

奖励分支原 HEAD 为 `9742013`。由于两条历史已分叉，本次采用保留双方提交的 merge 整合 `01ed799`，不是覆盖文件或强制重置。解决了 5 个文件的冲突：

- 环境运动、配置和轴向限制测试保留现场版本。
- `_run_actor.py` 保留同版本 demo 奖励缓存与整条 episode 提交链路。
- 配置默认保持 sparse，新 RoboMeter 模式明确启用后才生效。

## 整合修复

仅做文本合并会产生一个实际问题：reset 原本在返回前震动并取新帧，而旧奖励分支在 reset 返回后继续等待奖励。于是操作者可能提前收到开始提醒，第一帧也可能在长等待后变旧。

现在把奖励屏障放进 reset 生命周期的“运动验证完成”和“开始提醒”之间：

```text
本条结束 → 人工标签 / 后台奖励任务
               ├─ Actor：两段复位运动、到位与稳定验证 ─┐
               └─ Worker：等待 Learner 排空、计奖、完整入库 ┤
                                                        ↓
                    两边就绪 → 确认释放 → 单次震动 0.8 秒
                             → 状态检查、两路新帧 → 下一条采集
```

等待时复用当前 `_check_reset_hold`，只读检查稳定姿态、关节速度、力、夹爪和反馈时效。不发送 pose、gripper 或 recovery 命令。Stop、奖励错误或测量异常会阻止提醒和采集；已有奖励数据按原 pending / journal 机制保留。

通过标准 Gym `reset(options=...)` 传递临时回调；原生包装层测试证明它可穿过 RecordEpisodeStatistics、动作包装、RelativeFrame、Quat2Euler、SERLObs 和 Chunking。回调不会写入配置或录制证据；环境若忽略回调则禁止采集。删除旧的 reset 后单独刷新旁路，沿用现场已有的两路新帧时间边界、2 秒超时、0.25 秒最大帧龄和状态检查。

回执释放后，Actor 仍处于提醒/检查阶段，Learner 继续暂停，直到 Actor 真正开始下一条采集才恢复参数更新。最后一条仍单独排空，不强迫机器人再复位。

修正两处历史测试断言：随机复位只改平移、保持竖直姿态；正式 seed 采用当前 `fixed_xyz_front_v2_command20_20260921` 与 `insert-front-roi160-v2`。同步更正文档中的旧 4 mm 描述，运动代码保持现场当前 6 mm 条件。

## 验证结果

- 定向回归：**169 passed，9.68 秒**。
- 最终整库可运行范围：**1,166 passed，79 warnings，102.18 秒**。没有 deselected，也不再排除旧的两个基线断言。
- 新增 **12 项整合用例**，覆盖真实 reset + 奖励事务的并行时序、入库前不提醒、等待期间只读检查、7 种异常、原生 wrapper 传参及新观测、忽略屏障时禁止采集、提醒与等图期间 Learner 仍暂停。
- `node --check hilserl/web/app.js` 与 Git whitespace 检查通过。
- 四个独立依赖（hil-serl、serl、agentlace、serl_franka_controllers）通过锁定版本与补丁完整性检查。

最终测试命令（在隔离 worktree 内）：

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:upstream/agentlace:upstream/hil-serl/serl_launcher:upstream/hil-serl/serl_robot_infra:upstream/hil-serl/examples \
/home/robot/miniconda3/envs/hilserl-fr3/bin/python -m pytest tests -q \
  --ignore=tests/test_controller_freshness.py \
  --ignore=tests/test_controller_lifecycle_helper.py \
  --disable-warnings --maxfail=3
```

两个未运行模块仍依赖工作区之外指定的控制器 patch / lifecycle 源码。仓库内的上游原版控制器不是那份被测源码，本次没有拿它替代以声称通过。具体路径见 [首次验证记录](robometer-episode-gap-verification.md)。

日志：远端 worktree 根目录 `artifacts-reward-sync-focused.log` 与 `artifacts-reward-sync-full.log`；本地副本位于 `artifacts/reward-verification/`。

## 部署边界

此版本完成代码整合和离线验证，仍未启动真实 RoboMeter 4B 服务、真机训练或震动；默认配置没有启用新奖励模式。真实模型需按 [启用说明](robometer-episode-gap.md) 固定内容 ID、准备同版本 demo 奖励缓存，并测量吞吐和显存。暂停 JAX 更新不释放其常驻显存。

例如运动与到位验证 5 秒、奖励连同入库 7 秒，新增奖励等待约 2 秒；之后还有一次 0.8 秒提醒及实际等帧时间。这是调度示例，不是 GPU 实测。复位、提醒、等待均计墙钟，不产生训练样本。
