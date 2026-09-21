# HIL-SERL 插插头工作台

桌面图标打开本机 Web 控制台。训练、独立评估、Xbox 示教、回放、导出和状态查询共用同一入口与配置。

`bin/hil-serl console --background --open` 只打开工作台。选择模式并点击“开始任务”完成初始化；点击“复位并开始”后执行既有 hold-grip reset。真实运行前按 [Live Robot Gate](../runbooks/03_live_robot_gate.md) 完成现场准备。界面的“停止 Actor”停止后续任务动作，不能替代硬件 E-stop。

## 日常使用

1. 准备 FR3、两台 ZED 和 Xbox，在“底层控制服务”中检查服务状态。服务需要恢复时，停止 Actor、确认工作区安全且 E-stop 可随时操作，再点击“恢复底层服务”。Learner 可保持运行。
2. 选择“在线训练”“独立评估”或“人工示教”。评估填写 checkpoint 路径与条数，固定加载该 checkpoint，不连接 Learner；示教通过 Xbox 采集，独立保存。
3. 确认插头、线缆和工作空间后，点击“复位并开始”。Xbox 首次使用按 RB 解锁；训练中按住 RB 接管。评估不接入 Xbox 动作。
4. 采集中点击或按键 `0` 表示失败、`1` 表示成功。分类器成功或步数上限只结束采集，最终结果仍等待人工 0/1。
5. 下一条需再次确认复位。结束时先停止 Actor，再点击“停止 Learner 并保存”。只有 Learner 实际进入保存函数才显示保存，只有保存提交成功才记录最终路径。下次用“续训 Learner”恢复本轮最新已保存的 checkpoint，再启动 Actor；“启动 Learner”仍会新建训练目录。
6. 在“回放与数据”查看两路同步视频、跳到指定 episode，按结果、动作来源及选中的 episode 导出。未勾选 episode 时导出所有满足筛选条件的有效条目。

### 暂停、503 与续训

Learner 只在同轮 Actor 正在采集、进程和心跳有效、并且最近 2 秒收到该 Actor 的在线数据时更新参数。等待复位、人工裁决、反馈中断或 Actor 退出时自动暂停，保留模型、optimizer、replay 和策略服务。Actor 恢复有效采集后从原 step 继续；已有示教或旧 replay 不单独放行训练。

“手动暂停 Learner”仅暂停参数更新，可以在 Actor 运行时操作。“继续 Learner”解除手动暂停，仍需满足上述采集与数据条件。自动暂停会显示原因和 step。普通 episode 等待不增加 checkpoint；故障、Actor 退出或手动暂停时保存当前完整更新一次，同 step 的已提交 checkpoint 复用，避免反复保存。

运行中的 Actor 遇到 `/getstate` 503 或状态通信中断，会将当前 episode 标为 incomplete，保留实际完成的步骤和 partial command，进入“反馈中断，等待确认复位”。Actor、相机和 Learner 的进程保留。状态恢复并检查现场后，重新点击“复位并开始”；只有复位到位验证通过才采集。旧动作不会重发，旧的复位/标注确认不会沿用。硬件安全超限、记录写入失败仍按 fault 停止；底层进程需要重启时，先停止 Actor 再点击“恢复底层服务”，Learner 保持暂停。

显式续训复用原 run 与完整配置，恢复模型、target parameters、optimizer、RNG 和训练步数，并从原始示教及已落盘的 `buffer` / `demo_buffer` 重建 replay。不会自动恢复任意历史目录，也不覆盖比所选 checkpoint 更新的训练。每次进程启动有独立 attempt；Actor 的数据编号按 attempt 隔离，重启后从 0 开始的新数据仍可接收。进程崩溃前尚未落盘的数据和 replay 的随机采样顺序不保证逐位恢复。

图像 replay 仅在完整 `raw_transition_id` 证明同一次采集的相邻步骤时复用前一帧。跨复位、跨 Actor、稀疏人工介入、重复 credit 或缺少身份的历史示教，都保留样本自身的 observation / next observation，不伪造 terminal reward。新的 buffer 文件编号始终大于已有编号；兼容历史 Actor 重启产生的低编号文件时，按保存时间重载，避免把较旧经验当作最新经验。

### 复位与夹爪调试

“底层控制服务”提供“检查服务状态”和“恢复底层服务”。检查只读取健康状态；恢复按钮明确启动底层控制器并验证状态更新。点击恢复按钮即提交本次恢复操作，界面显示检查、恢复、就绪或具体失败原因；服务就绪表示控制器状态时效已确认，不表示机器人已经复位到任务起点。只有 Actor 已退出且没有正在进行的启动、停止或夹爪操作时才能恢复。恢复期间禁止启动任务、切换 Learner/Actor 或调整夹爪；完成后仍需按原有流程确认复位。

工作台的“夹爪调试”提供“读取开口”“张开夹爪”“闭合夹爪”。Actor 停止时由工作台取得设备锁；Actor 处于等待复位或暂停时，命令交给 Actor 线程执行。复位和采集中禁用手动夹爪控制，过期请求或已经变化的阶段不会执行。每个请求有独立回执；HTTP 应答表示命令被接受，开口变化也不等于插头已抓稳。没有传感器新鲜度证据时只允许读取，禁止发送开合命令。

Franka go home 或重新上机后，请先用“恢复底层服务”恢复控制器，再读取开口、调整插头夹持，最后确认复位。只有显式点击恢复按钮才启动底层服务；恢复不会自动解除 User Stop、执行 home、复位、采集或开合夹爪。底层 freshness 补丁以 ROS 回调接收时刻和实际控制子进程状态识别过期缓存；老服务若不提供这些字段，需要恢复服务以重新加载补丁。

复位必须完成垂直抬升和起始位姿两段到位验证，任何超时、停滞或错误都阻止后续动作与采集，`reset_strict=false` 也不能跳过关键验证。约 8 秒没有可测的进展会终止该段，界面显示阶段、耗时和位置/姿态误差。只有 `reset_info.reset.success=true` 才创建 episode；`succeed` 是任务奖励字段，不代表复位成功。

旧进程仍运行时，界面显示旧入口状态；继续在其原终端操作。软件安装不会重启现有 Actor/Learner，也不会为旧运行补录视频。新功能从下一次统一入口启动开始生效。

## 命令行

以下命令在项目根目录运行。CLI 与 Web 共用 Config、Manager、OperatorControl 和归档实现；设备进程共享独占锁。

```bash
bin/hil-serl check
bin/hil-serl status
bin/hil-serl train
bin/hil-serl train --learner-only
bin/hil-serl train --learner-only --resume-checkpoint /path/to/run/checkpoints/checkpoint_25974
bin/hil-serl learner-activity pause
bin/hil-serl learner-activity resume
bin/hil-serl eval --checkpoint /path/to/checkpoint_2000 --episodes 10
bin/hil-serl collect
bin/hil-serl command label 1
bin/hil-serl command continue
bin/hil-serl command pause
bin/hil-serl command stop
bin/hil-serl stop-learner
bin/hil-serl gripper status
bin/hil-serl gripper open
bin/hil-serl gripper close
bin/hil-serl replay
bin/hil-serl export RUN/CAPTURE --outcome success --source all --format npz
```

`check` 只检查文件、存储余量和 ffmpeg。可加 `--robot-tcp` 检查控制服务器端口，不发送控制命令。CLI 命令只提交阶段操作，终端退出不代表设备任务已完成；用 `status` 或工作台核对。

配置入口为 [config/hilserl.json](../config/hilserl.json)。显式路径用 `bin/hil-serl --config /path/to/profile.json ...`；存储位置用 `--data-dir /path/to/recordings` 或配置中的 `data_dir`。使用前准备好有写权限的目录，不自动挂载、扩分区或删除历史数据。

当前上限为 **190 步**：当前 20 条成功示教中位数为 94.5 步，按约两倍向上取整。配对原始记录为 Xbox、10 Hz，采样间隔约 0.100 秒。每步还有策略推理、HTTP 与图像处理耗时，因此 190 步不是严格的 19 秒墙钟截止。

原来的 VICE、Phase C、formal learner/actor、Xbox 插入示教和部署脚本保留为薄兼容入口；旧环境变量不再构成另一套配置。高级硬件诊断、GELLO 工具、分类器训练和控制服务器脚本保留各自用途。`scripts/run_actor.sh` 继续拒绝旧危险入口并指向新命令。

## 记录边界与故障

Learner 停止使用绑定 PID、进程起始时间和 attempt 的持久化请求，重复点击不重复发信号。Learner 在完成当前 update 的安全边界主动停止，并先保存、后清理通信资源；SIGINT/SIGTERM 只设置停止标志，避免 `KeyboardInterrupt` 落入 JAX GC callback 后被忽略。若未收到确认、保存失败或退出尚未完成，控制台分别显示对应状态，不把请求送达或 PID 消失当作保存成功。

- 两路 1280×720、H.264 MP4 持续保存，从相机成功打开到 Actor 关闭，包括等待、整理、复位、策略初始化和人工裁决时间。相机各有一个采集线程；侧面策略与分类器使用同一份采集帧。
- 有效轨迹从 reset 通过明确到位验证后的首个观测开始，到最后一个实际执行步结束，包含终止步。等待、整理、reset 不进入 episode step 文件。有效任务中的零动作仍保留。
- 最后一步先写入归档，再等待标签。人工裁决写入 episode 元数据；原始模型奖励和结束建议保留。运行中收到 0/1 后在下一个决策边界结束，不为结束标签额外执行一次动作。
- 所有已完成的成功、失败 episode 都保留。中途停止、暂停、相机故障或写入失败的条目标为 incomplete，默认不导出。动作已发出但外层观测/分类器失败时保存 partial attempt 与已发送命令，不伪造 next observation。
- RAM 只作为短队列缓冲，持续压缩落盘；默认每 60 秒分段。全场缓存到 RAM 不能减少最终磁盘占用，并会增加崩溃丢失范围。
- 默认保留 8 GiB 空间。相机停更、编码队列积压、轨迹写入失败或触及容量下限时，Actor 停止新任务动作、保留已有内容并报告 fault；Learner 暂停参数更新。解决后重新启动 Actor 会创建新的 recording attempt。
- 不自动清理旧 checkpoint、视频、轨迹或导出。异常断电时保留已完成的文件；最新编码片段和待写队列不保证完整。未正常关闭的记录保持 recording/incomplete，不作为完整采集验收。

## 数据目录与契约

```text
artifacts/runs/<run>/
  config.json                 # 本轮配置
  run.json                    # 模式、checkpoint、源码哈希
  attempts/                   # 每次 role 启动的不可变有效配置
  logs/
  checkpoints/                # 新训练的权重与训练 buffer
  recordings/<capture>/
    manifest.json
    events.jsonl
    video/{wrist_1,side_policy}/
      000000.mp4
      frames.jsonl
      encoder.log
    preview/                  # 最近一帧的轻量预览
    episodes/000001/
      episode.json
      000000.npz
artifacts/runs/exports/
```

实际 attempt 配置文件的绝对路径在 recording manifest 的 `metadata.config_file` 中。复用现有 Learner 时继承其完整有效配置；每次 Actor 启动独立快照，避免覆盖历史。

原始 step NPZ 只包含数值数组和 JSON 树，使用 `allow_pickle=False` 读取。字段保留：

| 字段 | 含义 |
|---|---|
| observations / next_observations | 原样保存策略输入输出观测；RGB uint8 图像、实际 19D 状态、obs horizon 1 |
| proposed_action | 策略候选的 7D 动作；示教模式为零候选 |
| policy_action | 应用策略旋转锁后的候选动作 |
| actions / source_action | 交给下层包装链的有效 policy/人工动作；不是伺服器实际关节轨迹 |
| controller_input_action | RelativeFrame / HoldGripper 等处理后的环境输入 |
| controller_commands | 发送到服务器的位姿目标、发送/返回时间和返回状态；不等于机械臂已到达目标 |
| raw_state / raw_next_state | 原始机器人状态及状态请求/响应时间：TCP、q/dq、力/力矩、Jacobian、夹爪 |
| frame_references | 相机 capture_id 与采集返回时间，关联视频索引 |
| sample_started / step_started / step_ended | 单调时钟纳秒和 Unix 纳秒；跨机同步精度不作保证 |
| observed_reward / terminated / truncated | 该次环境返回的原始奖励和终止建议 |
| complete_transition | 完整 transition 或动作已发出但缺少后续观测的 partial attempt |

机器人 TCP 平移为米，四元数为环境原有 xyzw 顺序；关节角/速度为 rad、rad/s，力/力矩为 N、N·m。夹爪沿用服务器归一化标度。7D 策略动作保留项目原有归一化顺序 `[dx,dy,dz,rx,ry,rz,gripper]`；具体坐标变换和缩放以同次源码与有效配置为准。图像时间戳是采集返回时刻，不宣称硬件曝光同步。

NPZ 导出 ZIP 包含筛选后的原始 step、episode 元数据与 manifest，不重复打包大视频；视频留在原始归档，通过 recording_id / capture_id 关联。SERL 格式导出每条 episode 的 transitions.pkl，终止奖励使用人工标签。按来源筛选保留原始 next_observations、step 索引和选择是否连续的标记，不把间隔拼接。训练内部 success credit 产生的重复或改写样本只留在训练 buffer，不混进原始 episode 或默认导出。

## 模块职责

- `hilserl/config.py`：配置、受控运行环境、有效参数。
- `hilserl/processes.py`：进程身份、启动/复用、取消、设备互斥。
- `hilserl/control.py`：Actor 拥有状态；命令绑定 attempt、episode 与本次复位确认。
- `hilserl/learner_control.py`：Learner 的持久化停止请求、主循环确认和实际 checkpoint 保存回执。
- `hilserl/training_gate.py`：同轮 Actor 身份、采集阶段、心跳与实际在线入库的训练门。
- `hilserl/gripper.py`：显式开合命令、状态时效与开口回读；不构造环境、不执行 home。
- `hilserl/episodes.py`：统一采集生命周期与人工最终裁决。
- `hilserl/storage.py`、`scripts/video_capture.py`：持续采集、短队列、压缩分段与逐步持久化。
- `hilserl/archive.py`：回放索引与数据筛选导出。
- `hilserl/webserver.py`、`hilserl/web/`：本机界面；只监听 127.0.0.1。
- `_run_actor.py`：训练栈适配器与 Learner；设备 reset/pose 仍由既有任务环境负责。

离线测试覆盖状态、录制、导出、兼容入口和启动匹配。合成画面与 fake environment 不能替代两路 ZED、Xbox、HTTP 控制器和真实机械臂的现场验收。


## 回放缓冲的无损压缩

新的 `checkpoints/buffer` 和 `checkpoints/demo_buffer` 使用带内容校验的 LZ4 level 0，文件名为 `transitions_<编号>.pkl.lz4`。这只改变磁盘编码，不改变数组、奖励、动作、图像分辨率或回放统计。全程视频、原始 NPZ、原始示教目录与导出格式保持原有契约。

统一入口同时读取旧 `.pkl` 与新 `.pkl.lz4`，按原保存时间和编号重建 replay。压缩文件截断、校验失败、未知格式或同编号存在两种副本时明确拒绝继续；不能把两份都入库。读取压缩 chunk 使用 `hilserl.replay_io.load_replay_chunk`，不要直接对压缩字节调用 `pickle.load`。

历史 chunk 转换仅在其写入者停止并取得维护锁后进行。`compress_existing_chunk(path, remove_source=True)` 逐块编码，再完整解压比较 SHA-256 和字节数；目标文件持久化后才释放原未压缩副本，保留原 mtime 和权限。失败保留原文件；若中断后两份都在，可重复调用完成核验和转换。模型 checkpoint 不在该转换范围。

Actor 每次成功复位后发布独立的采集窗口起点；本窗口首批在线数据到来前仅等待，不再把上一条数据的年龄误判为当前断流并另存 checkpoint。本窗口真正开始接收数据后，原有断流暂停和故障保存仍生效。

周期 checkpoint 保存会预留未压缩数组大小及64 MiB元数据余量；压缩缓冲逐块检查磁盘保留空间。故障保存和退出时的最终保存/缓冲收尾可以使用这部分应急余量，但仍检查当前写入是否能容纳，避免有空间却拒绝保存最后进度。该保护与已有录制/导出低水位检查配合，空间不足会明确报错并保留已提交数据。它不是文件系统配额，也不自动清理历史；长期录制仍需归档已结束的轮次。
