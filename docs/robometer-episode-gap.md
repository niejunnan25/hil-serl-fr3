# RoboMeter episode 间隔计奖

2026-09-22 更新。隔离分支 `codex/robometer-episode-gap` 已整合现场源码快照 `01ed799`，包括当前复位、单次提醒、新帧检查和异常停控。默认仍为 `reward_mode=sparse`。本文描述实际代码行为。CPU、模拟设备、实际本地 HTTP/Agentlace 通信和原生 replay 采样已有测试；尚未加载真实 4B 权重或启动真机训练，7 秒计奖是调度示例，不是实测吞吐。

## 执行流程

| 阶段 | Actor / 机器人 | 奖励 worker | Learner |
| --- | --- | --- | --- |
| 采集第 n 条 | 原有策略/人工接管；持续保存原始 NPZ、视频 | 暂存本条真实 transition | 在有效采集期用已入库的历史数据更新 |
| 结束采集 | 停止采集，等待人工标签 | 申请本 episode 的暂停屏障 | 完成本次完整更新组，并等待对应 GPU 运算完成 |
| episode 间隔 | 人工准备、原有复位和到位验证 | 收到暂停确认后批量计分；末步任务奖励等待人工标签 | 不开始新更新，网络接收线程保持运行 |
| 计奖完成 | 已复位则等待 | 保存分数、奖励和来源；提交完整 episode | 接收并保存完整事务，写入 online 和 demo 子集，返回回执 |
| 入库与运动均完成 | 确认入库并释放屏障；震动一次 0.8 秒，检查状态和两路新帧，复位才返回成功并开始下一条 | 收到释放确认，结束本条工作 | 提醒、检查期间继续暂停；看到健康 Actor 开始下一条后恢复更新 |

这里的并行是：机器人复位在 Actor 主线程，模型推理与提交在独立 worker；worker 不调用机器人 API。采集与 Learner 之间仍保持原有的异步策略发布方式。

```text
示例：共同起点，复位运动与到位检查 5 秒，计奖＋保存＋提交 7 秒
时间          0              5      7          7.8+
机器人        [复位与验证      ][等2秒][震动、取得新帧] → 下一条采集
奖励 worker   [计分、保存、提交、确认  ] → 完成
Learner       [不开始新优化器更新                   ] → 恢复更新
接收线程      [保持运行                            ] → 保持运行
录像与墙钟    [连续记录                            ] → 连续记录
```

实际额外等待还包括暂停排空、CPU 图像处理、传输与确认耗时，按时间戳记录。复位/人工等待/计奖可以重叠，不能把这些区间简单相加。环境步数仅在有效 `env.step()` 后增加；额外等待计入从工作台启动在线训练开始的墙钟时间。

复位运动完成后，奖励等待仍属于 reset 生命周期：持续只读检查已确认的稳定姿态、关节速度、力、夹爪及状态时效；异常时禁止进入采集。等待结束后才发出一次 0.8 秒提醒，再取得共同时间边界之后的两路新帧。之后经正常 reset wrapper 路径建立本条相对坐标原点、转换欧拉角、展开状态和堆叠图像。等待、提醒和等图不下发动作、不再次复位、不增加训练步数，也不运行成功分类器。

应用层通过 Gym 标准 `reset(options=...)` 传入临时 `hilserl_ready_barrier` 回调。它不进入配置文件、录像元数据或 reset 证据；如果环境忽略回调，Actor 会在采集前拒绝继续。删除旧版 reset 返回后单独刷新的旁路，避免绕过现场新增的新帧与稳定性检查。

首次采集不要求先有在线 replay：先拿到初始策略、验证模型服务身份和首次复位，再采第一条。Learner 仍受 `training_starts`（当前 100 条真实 online transition）约束。正常结束运行时专门处理最后一条，不依赖下一次复位。用户 Stop 优先结束设备流程，未完成的奖励任务留待恢复。

## 奖励和数据契约

- 当前 XYZ actor、策略图像、裁剪、动作控制、人工接管和复位动作保持原样。
- 第一版仅支持 `fixed-xyz-v1`。RoboMeter 默认看 `side_policy` 的 RGB policy observation，也可配置 `wrist_1`；一次选一个视角。
- 每个计分样本只含起点到该时刻的历史图像，最多 8 帧。客户端先构造这些独立前缀，再组成 mini-batch。服务不保存跨请求的 episode 历史，重试不会混入其他轨迹。
- 图像保持现有 HWC uint8 内容，不照搬 LIBERO 的 180° 翻转。模型内部仍用 RoboMeter processor。
- 相邻步骤仅在前一步 `next_obs` 图像与后一步 `obs` 完全相同时复用边界分数；否则按实际图像分别计分。复位图像不作为最后一步的 `next_obs`。

```text
r_train[t] = r_task[t] + scale * (gamma * mask[t] * Phi(next_obs[t]) - Phi(obs[t]))
```

`gamma` 来自和 SAC 相同的 `discount` 配置，默认 0.98。`r_task` 是人工任务结果，保留当前固定 XYZ 的终止/mask 语义。普通步的任务奖励为零，人工结束的最后一步 mask 为零。

**成功标签独立于模型奖励。** 失败末步可能有负奖励；人工成功在不同 scale 下也可能得到非正的塑形后奖励。demo 子集仍由人工接管/独立成功标签决定，不按奖励是否非零决定。RoboMeter 不决定任务 success、终止或“最近 10 次成功”的评估标准。

reward contract 固定模型内容 ID、任务文本、视角、图像 profile、前缀规则、折扣和 scale。online 与预先准备的 demo 缓存必须匹配此版本。缓存仍校验原始示教集 SHA、每个原始 observation/action 和 transition ID。模型分数、奖励公式、像素哈希、数据类型和维度在 replay 入库前校验。

## 暂停、入库与恢复

`EpisodeCommitServer` 与完整 Learner update group 共用一把许可锁。请求暂停后，已开始的组允许结束；`jax.block_until_ready` 完成后才回报无 in-flight 更新。新模式使用组内按需采样与传输，关闭跨间隔的预取。

Agentlace 新增应用层请求类型 `episode-reward-v1`，操作为 `hello / begin / status / commit / release`。请求绑定 run、当前 Actor attempt、gap 和 Learner generation。worker 拥有自己的 REQ socket，不跨线程共享 Actor 的 socket。旧的逐步 datastore 路由在此模式下拒绝非空数据。

完整 episode 在提交前经过验证并写入持久日志，然后写 online 与应进入 demo 的子集；两者成功后才返回完整回执。重复提交返回同一回执，丢失 release 回执也可重试。相同原始步骤换一个 episode 身份再提交会被拒绝。

如果两个 buffer 之间发生部分写入，服务锁住后续更新，不宣称已经提交。重新启动 Learner 时，从完整持久日志重建两个 buffer，包含上一次已保存但尚未回执的事务，生成新 generation 的回执。旧 RAM 的回执不能直接证明新 RAM 已就绪。

Actor 重启时，在首次复位之前恢复已有人类标签的 pending episode；使用缓存分数时仍校验图像与奖励版本。未完成的环境 transition、没有最终人工标签的记录保留并列出原因，不伪造标签，也不重新发送机器人动作。模型超时、无效分数或版本错误会暂停这条链路，不用零奖励兜底训练。

存储位置：

| 路径（相对于 run） | 内容 |
| --- | --- |
| `recordings/<attempt>/episodes/` 与录像目录 | 原有原始 NPZ、人工标签、相机索引和连续录像 |
| `recordings/<attempt>/reward/status.json` | 当前 worker 阶段、计分时间、入库回执、额外等待 |
| `checkpoints/reward_pending/<episode-key>/` | 待完成数据；完成后保留 `entry.json`、`scores.json`、`ready.json`、回执等小文件 |
| `checkpoints/episode_replay/*.pkl` | 完整、校验过的训练 episode 事务，供 Learner 重启重建 replay |
| 独立 `reward_seed_cache` | 带同一奖励版本的示教缓存，不改写原示教集 |

完成回执后删除 pending 中重复的整幅图像 payload，避免永久保存多份相同的相机数组。原始记录和接收端完整事务继续保留。工作台展示计奖阶段与额外等待；在计奖/入库/收尾阶段仍可 Stop。

## 启用方式

代码不会自动安装模型或启动推理服务。RoboMeter 的 PyTorch 环境和现有 JAX 环境可以分开；模型常驻，episode 之间只调用推理。

1. 在安装了 RoboMeter 的独立环境中运行适配器。`robometer_backend.py` 来自已核查的 SEAL-0 native backend（参考提交 `8c5985ed02a5e73eebe43cdf19465ccb25049f42`），沿用其 eval/inference_mode、dtype 和相同帧数分桶逻辑。

```bash
python scripts/serve_robometer.py \
  --model-path /path/to/Qwen3-Robometer-4B-combined-local \
  --robometer-root /path/to/robometer \
  --device cuda:0 --port 20008
```

服务启动时打印根据权重、配置和 tokenizer 文件内容计算的 `model_id`。只计算身份、不加载模型可加 `--manifest-only`。本适配器提供 `/health` 和 `/score`；旧仿真服务的 pickle RPC 地址不能原样当成这个新端点使用。

2. 复制本 worktree 的 `config/hilserl.json` 为独立配置，保留原有动作、图像和 seed 约束，加入以下字段。`reward_model_id` 填服务实际打印的 64 位十六进制内容 ID，不填模型目录名。

```json
{
  "reward_mode": "robometer-episode",
  "reward_url": "http://127.0.0.1:20008",
  "reward_model_id": "<服务实际打印的64位内容ID>",
  "reward_task": "Insert the plug into the socket.",
  "reward_image_key": "side_policy",
  "reward_scale": 1.0,
  "reward_max_frames": 8,
  "reward_batch_size": 8,
  "reward_timeout_seconds": 120,
  "reward_seed_cache": "artifacts/reward-seed/v1",
  "discount": 0.98,
  "port": 5688,
  "broadcast_port": 5689,
  "console_port": 8766
}
```

以上是要合并的字段，不是完整可直接覆盖的配置文件。worktree 不自动包含 demo/分类器权重；应为这些只读输入配置正确的绝对路径。端口使用独立值，并让新运行输出留在新 worktree。worktree 隔离代码与文件，不等于自动隔离机器人资源；真机测试前需完成现用 Actor 的设备交接。

3. 用现有示教集离线生成奖励缓存，再检查新配置。

```bash
python scripts/prepare_reward_seed.py \
  --config config/robometer.local.json \
  --output artifacts/reward-seed/v1
python -m hilserl --config config/robometer.local.json check
```

缓存生成只调用奖励服务，不连接机器人。输出目录必须全新，避免覆盖旧版本。

4. 设备可用于本分支测试后，使用这份配置打开工作台，再启动训练。

```bash
python -m hilserl --config config/robometer.local.json console --port 8766
```

上述服务/工作台/训练命令为使用说明，本次开发没有执行它们。回到 `reward_mode=sparse` 使用原有链路；不同奖励版本应分别创建新 run，不能给正在运行的实验热切换定义。

## 已知实测边界

- 真正的 RoboMeter 权重、目标 GPU 上的吞吐和与 JAX 同卡驻留显存尚未实测。暂停参数更新不会释放 JAX 参数、优化器和显存池。
- 本功能记录和等待计奖时间；没有新增“30 分钟自动截止”或“最近 10 次成功自动结束”的实验管理规则。
- 延续原有 Actor 内连续录像。Actor 退出之后的跨进程独立录像仍属于另一项功能。
- 超时/Stop 保留 pending；没有人工最终标签的历史轨迹不会自动推测 success。下一次 Actor 可以恢复已标注数据，不能恢复旧动作。
- 最新同步与回归见 [版本整合报告](robometer-runtime-sync.md)；[首次开发报告](robometer-episode-gap-verification.md) 保留当时结论。不能把模拟设备/模型结果当成真实 4B 模型的性能结论。
