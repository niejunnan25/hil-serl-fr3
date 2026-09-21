# RoboMeter：episode 间隔计奖接入设计

状态：2026-09-21，隔离 worktree 已创建，依赖与导入隔离已修复并测试。下文为待接入的运行协议；真实 RoboMeter、Actor/Learner 和机器人尚未按此协议运行。

## 本轮开发基线

- 主目录：`/home/robot/serl_projects/hil-serl-fr3`，`main` 的提交为 `ed67521251c620e0d5617438af04d5d69f9f09f7`。
- 隔离目录：主目录下的 `artifacts/worktrees/robometer-episode-gap`。
- 分支：`codex/robometer-episode-gap`。
- 现用源码快照：`d6fa510246b8960fe952164640657fd79af11f51`。包含当时尚未提交的相机、动作控制、数据导出等源码，不包含数据、录像和权重。后续新改动从此提交比较。
- 快照证据：主目录的 `artifacts/worktree-audit/20260921T092346Z/`，保存原始 diff、未跟踪源码归档以及 405 个源码文件的哈希/权限清单。创建前后主目录源码、HEAD 和 index 校验一致。
- 四个 upstream 仓库使用独立 checkout，按此快照的 lock/patch 恢复；没有把可修改源码链接回主目录，也没有重新安装共享 Conda 环境。

本轮代码变更仅处理开发隔离：`manage_upstream.py` 支持 linked worktree 的 `.git` 文件；`Config.environment()` 优先导入本 worktree 的 Agentlace，避免共享环境中的 editable install 指向主目录。现有训练奖励逻辑保持现状。

## 约定的一个循环

1. 采集第 n 条 episode。原始步骤交给现有记录器及时保存；第 n 条暂存在待计奖区，尚不进入训练 replay。Learner 在有效采集阶段使用已提交的旧 episode。
2. 采集一结束，立即关闭本轮更新许可，生成唯一的 `gap_id`。Learner 完成本次已经开始的完整更新组，等待对应 JAX 运算结束，再发布与该 gap 匹配的暂停回执。
3. CPU 可以提前准备轨迹和图像。奖励 worker 收到暂停回执后运行 RoboMeter；Actor 主线程继续原来的人工标签、准备和复位流程。这两条流程并行。
4. RoboMeter 计算完成后，结合人工最终标签生成每步奖励，保存奖励 sidecar，再提交完整 episode。接收端在两个 replay 都完成后返回提交回执。
5. Actor 等到复位验证通过、标签已确认、奖励 worker 已释放计算权、episode 已确认入库，才开始下一条。Learner 同时还须满足心跳、身份、手动暂停和最低样本量等条件。

```text
示例：两条流程同时开始，复位 5 秒，计奖＋保存＋提交共 7 秒
时间             0             5     7
机器人           [复位与到位检查] [等2秒] → 下一条采集
奖励和入库       [计算奖励、保存、提交、确认] → 完成
Learner          [不开始新的优化器更新     ] → 恢复更新
接收线程         [保持运行，处理请求与入库 ] → 保持运行
录像/墙钟        [连续记录                ] → 连续记录
```

7 秒是用户认可的举例，不是 FR3 实测值。实际额外等待从“复位和操作员都已就绪”到“奖励及入库屏障打开”计算；应包含暂停排空、图像预处理、模型推理、落盘、传输、入库与回执的实际耗时。若这些结束晚于复位，所有新增等待都计入点击开始在线训练后的 30 分钟。环境步数只在有效 `env.step()` 时增长。

## 必须先解决的现有代码约束

| 位置 | 当前行为 | 新模式的修改 |
| --- | --- | --- |
| `hilserl/episodes.py::run_episodes` | 每步经 `emit` 提交，仅保留末步等待标签；标签之后才进入下一次复位 | 增加 episode 暂存及间隔调度钩子，结束采集时立即关闭更新许可；最后一步图像可先计分，终止奖励等人工标签 |
| `_run_actor.py::actor.emit` | `data_store.insert()` 后 `client.update()` | 新模式只放入 episode 暂存；整条完成后走专门提交路径，保留旧模式分支 |
| `_run_actor.py::actor.emit` | `bool(dones and rewards)` 被当作成功 | 改读独立的 `infos.succeed`/人工 verdict，成功与模型奖励数值分离 |
| `hilserl/training_gate.py::TrainingGate.check` | 要求最近 2 秒收到了本次 collection 开始后的在线数据 | 新模式核验已提交 episode 和 gap 状态；保持 Actor 身份/心跳检查，不再用连续流接收频率判断批量数据健康 |
| `_run_actor.py::learner` / `hilserl/learner_control.py` | 在完整 update group 前检查 gate；UI 的 paused 不是对应 gap 的 GPU 完成回执 | 关闭许可后排空当前更新组，在 JAX 同步完成后记录本 gap 暂停确认；接收服务继续运行 |
| `hilserl/seed_dataset.py` | 固定 XYZ 的示教加载为稀疏人工奖励 | 为新 reward contract 生成独立计奖缓存；在线和 demo 采样使用一致的版本 |
| `hilserl/config.py`、工作台和运行快照 | 尚无 episode-gap reward 配置/状态 | 增加显式模式开关、模型/输入/奖励版本、等待原因和回执指标 |

当前 Agentlace 的通用 `batch_insert` 不是跨 online/demo 两个 buffer 的事务：插到一半出错时会冻结错误，但此前已插入部分数据。因此“客户端已排队”或单个 store 的游标不能被当成整条 episode 成功入库。

## 建议的模块与协议

新增 `hilserl/episode_reward.py` 管理一个待处理 episode，`hilserl/reward_provider.py` 包含统一模型接口和 RoboMeter 客户端，`hilserl/episode_commit.py` 管理接收端提交和恢复。

模型接口可返回按 observation 身份索引的进展分数及模型版本。调度层不依赖模型名称，以后可以换其他奖励模型。第一版只允许一个待处理的完整 episode，不在后台累计多条无界任务。

### 1. 暂停握手与线程边界

建议标识：`run_id / actor_attempt_id / learner_attempt_id / episode_id / gap_id`。每次间隔生成新 gap；旧回执不能打开新间隔。

- Learner 在同一把许可锁下检查是否能开始完整更新组并标记 in-flight；gap 请求在这把锁下撤销后续许可。先开始的更新组允许结束，之后不再开始新组。
- 等参数、优化器状态及相关已提交运算完成，再返回 `paused_for_gap` 和最后完成的 update group。不能只靠 Actor 看见工作台“paused”。
- `env.reset()` 和设备控制继续由 Actor 主线程执行；后台 worker 只读冻结的图像/轨迹，调用模型和提交数据。worker 不能向机器人发指令。
- worker 向主线程投递状态事件，由主线程维护 Actor 状态文件，避免两个线程互相覆盖状态。
- 提交 worker 使用自己创建、自己关闭的网络 client/socket，不能跨线程共享当前 Actor 的 ZeroMQ REQ socket。
- 采集、间隔等待及推理较慢时仍需维持真实 Actor 和 Learner 心跳。健康心跳与新数据到达是两个字段。
- 同卡模式暂停优化器后仍有模型参数和显存池驻留。需要另行测量常驻 RoboMeter＋JAX 的显存；也要检查 reset 路径是否仍调用其他 GPU 分类器。不能由“Learner 停更新”推断整张 GPU 完全空闲。

### 2. 计奖必须保持逐步因果

每个查询只使用本 episode 的起始帧与该时刻之前的历史帧，最多 8 帧，多个这样的前缀组成 mini-batch。不要把成功后的图像放进早期步骤的输入。每个 episode 使用独立服务 session；重试、重新分批时不能继承别的 episode 历史。

建议第一版使用现有前视图的 `side_policy` RGB 裁剪图，保持与 policy observation 的对应，经 RoboMeter 自己的 processor 处理；它与 policy 的 160×160 输入是两份独立的输入契约。服务预处理、输入视角和任务描述需记录为版本。若实测证明 160×160 丢失必要细节，再通过原始帧索引换取高分辨率输入。不要照搬 LIBERO 的 180° 翻转，不在本次改变相机位置或现有裁剪。

对 N 条真实 transition，连续时通常需 N+1 个 observation 分数。只在图像/帧身份确认相同的情况下复用相邻步骤的分数；否则分别对真实 `obs` 与 `next_obs` 计分。不能把重置后的 observation 当作最后一步的 `next_obs`。

建议奖励：

```text
r_train[t] = r_task[t] + lambda * (gamma * mask[t] * Phi(next_obs[t]) - Phi(obs[t]))
```

`r_task` 来自现有人工标签，`gamma` 与实际 Learner 配置一致。普通步、真实终止、时间上限分别沿用清楚的 bootstrap 语义，不由模型自行决定 done/success。第一版可维持当前固定 XYZ 的人工结束即 terminal 语义，并在 reward contract 写明。

奖励 sidecar 至少保存：原始 transition ID、输入帧/图像身份、任务文本/输入版本、模型身份、原始任务奖励、两个 potential、mask、塑形分量、最终奖励和实际计时。原始 NPZ/视频保持可重新计奖，不把模型输出覆盖成人工标签。

### 3. 提交回执与崩溃恢复

优先使用 Agentlace 现有 custom request 扩展点：在两端 `TrainerConfig.request_types` 中注册 episode-gap 请求，由应用层接收，不必先重写传输库。

推荐请求：`gap-status`、`commit-episode`、`get-episode-commit`。具体名字可以调整，但协议至少包含：

状态查询必须立即返回当前状态，不能占住接收回调等待 Learner 暂停或模型推理；提交端未获暂停许可就返回 not-ready。模型推理运行在独立 worker，避免控制/接收线程互相等待。

```text
episode_key = run_id + source_actor_attempt_id + episode_id
commit_key  = episode_key + raw_digest + reward_contract_digest
receipt    = commit_key + learner_attempt_id + online_count + demo_count
             + commit_sequence + committed_at + payload_digest
```

- 提交前校验完整 episode 的数目、顺序、shape/dtype、finite 值、末步标签、图像身份和奖励版本。奖励未算完不发占位零奖励。
- 持久保存完整提交材料，暂停期间依次写 online 和应进入 demo 的条目；只有两个 store 都成功且提交日志完成，才开放训练可见性并发送回执。demo 子集为人工接管/现有成功样本规则，不能把失败的非零奖励误送进去。
- 同一 commit_key 同一内容的重复提交只返回原回执；相同 episode 换了奖励版本不能偷偷追加进同一个 run。
- 第一版不要求环形 buffer 原地回滚：若任一 store 部分写入失败，整个 Learner 的此轮 replay 标记不可训练，保持暂停；通过已持久确认的 episode 日志重建两个 buffer 后才能恢复。必须证明失败期间预取/采样线程不会把部分数据交给更新。
- `committed` 表示当前 Learner generation 中完整入库。Learner 重启后，旧回执不能证明新 RAM 里有数据；应从提交日志重建或重新提交，并返回新 generation 的回执。数据恢复与 Actor 身份核验分开，不能靠伪造旧 Actor 心跳通过 gate。
- 同一重试始终恢复数据，不重新发机器人动作。没有完整 transition 的故障轨迹单独标记，第一版不纳入训练。

### 4. 恢复条件和首次启动

```text
next_episode_allowed = reset_verified AND human_ready AND reward_finished
                       AND reward_compute_released AND replay_commit_ack_valid
                       AND NOT stop_or_fault

learner_allowed = actor_collecting_and_healthy AND gap_released
                  AND committed_replay_healthy AND enough_online_data
                  AND demo_reward_contract_matches AND NOT manual_pause_or_stop
```

首次采集是例外：没有“上一条 episode”，Actor 在模型服务就绪、初始策略已拿到和首次复位通过后可以采集；Learner 仍等待正式 online 数据达到 `training_starts`（当前 100）。否则要求第一条之前已有入库回执，会形成启动死锁。

新模式在一次 episode 内没有新的入库记录，因此不能再用 2 秒数据过期规则；Actor 心跳失效仍立即阻止新更新。第一版保持现有采集期间的更新调度，并记录每条新增样本对应的更新组数；若之后要限制每条样本更新预算，应增加独立明确配置，不顺带更改算法比例。

预取 batch 可以来自已提交的旧数据。入库完成并不保证恢复后的第一个 minibatch 必然抽到最新 episode；SAC replay 随机采样允许这种情况。测试应验证新 episode 已具备被采样资格。

### 5. 结束、异常、数据与计时

- 当前 `max_total_steps`、`max_episodes` 或操作员结束可能使主循环在最后一条后退出。需要专门的 final drain：最后一条也完成计奖和提交，不能只在“下一次 reset”钩子里处理奖励。
- Stop/E-stop 优先终止继续采集，不等待无期限的模型 RPC。停止时将 pending 状态持久保存；下一次只恢复数据工作。普通 30 分钟截止之后的收尾可继续保存，但须单列预算外 drain 时间，不能声称仍在 30 分钟内训练。
- 模型超时、无效分数或版本不匹配：保留原始数据并暂停，不自动退回零奖励继续训练。
- 原始录像依旧覆盖 Actor 运行中的复位和等待；本方案不宣称补上 Actor 退出后的独立连续录像机制。
- 固定 XYZ 训练保持 XYZ actor、当前图像输入、人工接管和 reset 行为。RoboMeter 输出不改变“最近 10 次成功”的人类任务标签或无人接管评估定义。
- 墙钟起点按最新约定为工作台启动在线训练；日志分别记录 collection、label wait、reset、pause drain、reward preprocessing/inference、commit、额外等待和实际 update 数。并行区间保留时间戳，不能把重叠的两段耗时相加当总耗时。

## 实施顺序与验收

1. **状态机和回执**：先接 fake provider/fake robot/内存 replay。验收复位 5 秒、计奖及提交 7 秒时正确等待；反过来模型先结束必须等复位；身份错配、回执丢失、部分入库、stop 和首次/末次 episode 都覆盖。
2. **现有循环接入**：修改 `episodes.py`、Actor emit 和 Learner gate。验收旧模式回归、gap 中接收线程可用、暂停确认之后无新更新、下次采集恢复；原始记录不丢，复位不产生训练样本。
3. **真实模型适配**：在允许的独立测试资源上，用一条已录的成功和失败 episode 运行 RoboMeter。验证批量/逐条结果对应、前缀不含未来帧、帧顺序/视角、模型身份、真实延迟/显存和奖励分布。
4. **demo 和 UI**：生成独立 demo reward 缓存，补充工作台状态与墙钟指标。混用错误奖励版本应启动失败。
5. **短程真机验收**：在独立 run/端口、当前作业可让出设备后进行，确认录像、reset、入库与暂停/恢复全链路，再启用正式 30 分钟任务。

后续所有修改继续提交在本分支。开发测试使用 CPU/fake 设备；真实依赖导入必须解析到本 worktree。真实数据和 checkpoint 不随 Git worktree 自动复制；需要使用明确的只读输入路径，新运行输出与控制目录使用本 worktree 自己的路径。未经设备交接，不从这个目录启动连接真机器人的 Actor。

## 本轮已完成的检查

- 187 项测试通过，范围为依赖管理、worktree 导入、进程配置、现有 TrainingGate、复位边界、Learner 控制/停止以及 Agentlace 实际本地套接字重传测试。测试禁用了 GPU；设备使用 fake，本地套接字使用临时端口。
- 四个依赖的 pin/patch 检查通过；独立 Python 子进程中 `hilserl`、`agentlace`、`serl_launcher`、`franka_env`、`experiments` 的解析路径均在新 worktree 内。
- 输出和控制路径属于新 worktree；主目录源码、HEAD、index 的前后校验一致。
- 这些检查验证了隔离开发准备和原有链路回归，尚未验证上文拟议的奖励状态机、真实模型性能或真机行为。
