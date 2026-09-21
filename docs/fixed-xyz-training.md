# XYZ 训练与旧模型

当前新建训练在 XYZ 契约之上采用 `insert-roi160-v1` 图像版本，seed 为 `demos/fixed_xyz_roi160_v1_20260914`；见 [相机裁切说明](camera-input.md)。下面保留的 `fixed_xyz_v1_20260914` seed 路径与 SHA 描述最初的全图 128 基线，继续供该旧版本的显式续训使用。

项目配置 `config/hilserl.json` 的 `action_contract=fixed-xyz-v1` 只影响新建训练、独立采集。控制台“续训 Learner”读取所选旧 run 的完整配置；评估也读取所选模型的配置，不能套用今天的新默认值。历史配置缺少 action_contract 时明确按 `legacy-hybrid-7d-v1` 解释。没有运行配置的外部 checkpoint 不能猜测动作维度后执行。

新版本使用三维 standard SAC，Actor、critic 输入、目标动作、熵和温度优化均为 XYZ。目标熵为 -1.5。设备接口仍是七维：在相对坐标系变换前后保持旋转和夹爪为零。夹爪调试按钮继续由独立的人工控制入口处理；训练 policy 不能打开夹爪。原有 8 mm 位移上限、控制频率、动作缩放、力、速度、关节约束不变。

三维模型不能直接加载七维 Hybrid 的动作头、critic 和优化器状态。旧示范、旧 buffer 保留，模型 checkpoint 按 [当前保留规则](checkpoint-retention.md) 留下每轮五个；续训旧模型仍使用旧算法约定。新训练从新目录开始，不能声称继承了旧 checkpoint 的全部训练进度。

## 示范与在线回放

`demos/fixed_xyz_v1_20260914` 包含 13 条全程人工接管且人工确认成功的完整轨迹，共 1,004 条 transition。来源是 2026-09-13 三轮原始记录，不使用旧的绝对坐标 PKL 转换集，也不依据文件名生成成功标签。每条原始记录、episode 元数据、派生 NPZ 的 hash 和排除原因均写入 manifest。

Manifest SHA-256 固定在配置和每次运行快照中：

`8b53d30229bb8b142ef4f0d07f4390fcb48566848d3e7ce5726126b19e7a9017`

启动与加载核验全部内容。19D observation 及相机图像原样保留；位姿相对于真实 reset frame，动作是当前末端 body frame 的归一化 XYZ。构建工具核对原始机器人状态及实际控制输入，拒绝不一致数据。重新构建必须选择新目录，再核验和固定新的 manifest。

新版的非终末奖励为 0，人工确认的末步奖励为 1 或 0，末步 `done=True, mask=0`。夹爪惩罚不进入新模型，也不复制成功尾部并改写 reward/done/mask。故障和中断记录仍保留，但不能编造成功或完整 transition。

Seed 仅进入 demo buffer。新 online buffer 从空开始；必须收到至少 100 条真实在线 transition，同时通过 Actor 身份、正在采集、心跳、当前采集窗口和在线数据 freshness 检查，才允许更新。图像 support slots 不计为在线数据。同一 raw transition id 重复入库会被拒绝；传输层丢 ACK 的已提交重试仍幂等。

## 更新与诊断

新版启用 `selective_optimizer_updates`。配置 `cta_ratio=2` 对应一次 critic-only 更新和一次 critic/actor/temperature 更新；critic-only 期间未选 Adam 的参数贡献、动量和计数冻结。所选 critic 对共享视觉编码器的合法梯度仍保留。旧算法缺省关闭此选项，避免改变旧模型续训约定。

每组更新提交、发布和保存前检查完整参数、目标网络、优化器状态及损失的有限性。NaN/Inf 使该组被拒绝，并保留上一完整健康状态。此检查只验证数值有效性，不证明策略任务表现。

新 Learner 在 `run/metrics/learner-<attempt_id>.jsonl` 保存实际 loss、Q/target Q、entropy、temperature、计时和更新计数。每 50 组或 10 秒写入，结束时补齐；数值检查每组执行，不依赖日志采样。存储失败明确报错。

`critic_utd = 本次 Learner attempt 已完成 critic 更新数 / 本次 attempt 新入库且唯一的 online transition 数`。恢复的历史 replay 不进入该分母；N=0 显示 null。它不同于 critic/actor 更新比例，也不因 critic ensemble 有两个网络就额外乘二。首组编译耗时不能当作稳定运行速度。

## 使用与效果验收

控制台刷新后，新建训练使用新版；“续训 Learner”保留所选旧版本。新 Learner 等待在线数据是正常状态，Actor 仍需操作者确认复位、按需接管并给出 0/1 结局。不要为跳过等待而放松 freshness 或安全条件。

局部测试、原生模型小批量更新及保存恢复，只验软件和数据链路。学习速度与稳定性仍需新 run 的真机记录验证。比较应固定 checkpoint 做纯策略评估，分别报告无接管成功率、接管比例和失败类型，不能用人工救回后的成功率代替策略独立成功率。

下一轮分析优先看实际 critic UTD、Q/target Q、entropy/temperature、同版本的接管趋势和动作限幅频率，再决定采样比例、更新速率或动作尺度的单项调整。当前保留原尺度，避免在修复动作和奖励契约时同时改变执行动力学。
