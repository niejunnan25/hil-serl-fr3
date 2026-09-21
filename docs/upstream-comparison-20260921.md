# HIL-SERL 官方仓库与 FR3 插插头工作目录对照

Git 整理说明：本文的 Git 状态统计记录整理前快照；随后已开展版本管理整理，当前操作方式见 [Git 管理说明](git-management.md)。

核对日期：2026-09-21。检查范围为远端 pku5080 的 /home/robot/serl_projects/hil-serl-fr3。官方基线通过 git ls-remote 核对，再从本地上游仓库的 Git 提交对象读取；没有用已修改的工作文件充当原版。本文是源码和配置对照，没有运行训练、测试或硬件操作。除本文外未修改项目文件。

**结论**

当前系统保留 HIL-SERL 的 SAC/RLPD、视觉编码器、Actor/Learner 分工和人工干预学习框架，在其上完成了 FR3、ZED、Xbox 和插插头任务适配，并增加了工作台、逐条人工裁决、运行归档、暂停恢复与数据完整性检查。部分修改直接进入了上游训练库和 Agentlace。因此当前版本属于带本地补丁的应用工程；不能只把它理解成官方示例换了相机序列号，也不能把继承的 SAC 或固定夹爪支持归为本地原创算法。

**1. 对照基线与改动规模**

| 项目 | 核对结果 |
|---|---|
| 官方仓库 | https://github.com/rail-berkeley/hil-serl |
| GitHub 当前 main/HEAD | c32939bccb65f3b8c43a9f9add3d322d4ab0264a |
| 本地 upstream/hil-serl HEAD | 与上面完全一致；提交日期为 2025-10-28 |
| 主工作目录 Git HEAD | 7989af84878059ae2cb2426cfda5abb40aba6119，2026-06-18 |
| 官方库内已修改的受跟踪 Python 文件 | 14 个，git diff 统计 +185 / -32 行，不含 pyc 和安装产物 |
| Agentlace | cf2c337c5e3694cdbfc14831b239bd657bc4894d；3 个 Python 文件和 README 有本地修改 |
| Franka 控制器仓库 | 1f140ef0d8e3fc443569c193d3ede1856e50d521；控制器 YAML 改为 FR3 标识 |
| 应用主训练程序 | 官方 train_rlpd.py 为 525 行；当前 _run_actor.py 为 1256 行 |
| 新工作台包 | hilserl 包含 27 个 Python 文件，约 6001 行，不含 Web 前端 |
| tests 目录 | 75 个 Python 文件，含辅助文件；数量不是本次测试通过的证据 |

行数是本次快照的近似规模指标。主目录含大量未跟踪文件和历史删除项，不能把 git status 的总数当成独立功能数。主程序比较是源码相似性对照，不声称拥有完整的文件重命名历史。

**2. 原版实际提供什么**

官方是用于真机操作的研究代码和示例集合，主要包括：

- serl_launcher：SAC、Hybrid SAC、BC、视觉编码器、图像增强、replay buffer 与训练工具。
- serl_robot_infra：Franka 的 Gym 环境、RealSense 相机、SpaceMouse 接管，以及经 Flask/ROS 使用阻抗控制器的接口。
- examples：示教采集、成功/失败图像采集、奖励分类器训练、RLPD、BC、HG-DAgger，以及 RAM 插入、USB 拾取插入、双臂交接、翻蛋等示例。

典型过程是：先采成功/失败图片训练分类器，再采少量成功示教，启动 Actor 与 Learner，操作者在需要时接管，最后固定 checkpoint 评估。

Actor 执行策略并通过 Agentlace 发送经验；Learner 用 demo/干预池和 online 池各一半的批次训练，周期发送新策略。官方已经支持固定夹爪和可学习夹爪、单臂和双臂。固定夹爪使用普通 SAC；可学习夹爪使用带夹爪分支的 Hybrid SAC。官方并非只有一种动作维度。

官方来源：
- [官方 README](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/README.md)
- [官方操作流程](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/docs/franka_walkthrough.md)
- [官方 RLPD 主程序](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/examples/train_rlpd.py)
- [官方 USB 任务配置](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/examples/experiments/usb_pickup_insertion/config.py)

官方论文或示例给出的成功率、收敛时间属于其场景，不是本项目的实测结果。

**3. 当前真正的启动链**

1. /home/robot/serl_projects/hil-serl-fr3/bin/hil-serl
2. /home/robot/serl_projects/hil-serl-fr3/hilserl/__main__.py：解析 train、eval、collect、console 等命令。
3. /home/robot/serl_projects/hil-serl-fr3/hilserl/processes.py：准备有效配置、运行目录、设备互斥与进程身份，启动角色进程。
4. /home/robot/serl_projects/hil-serl-fr3/_run_actor.py：同一程序根据 --actor / --learner 执行相应角色。
5. /home/robot/serl_projects/hil-serl-fr3/experiments/mappings.py：当前只注册 plug_insertion。
6. /home/robot/serl_projects/hil-serl-fr3/experiments/plug_insertion/config.py：组合任务环境、Xbox、RelativeFrame、观测包装、奖励判定、XYZ 动作包装。
7. 模型和部分环境实现来自带本地补丁的 upstream/hil-serl；通信补丁位于 upstream/agentlace。

当前主线实际使用根目录 experiments/plug_insertion。部分旧 README 仍称这一目录只是历史参考、运行时应使用 fr3_experiments；该说法与当前启动链不一致。fr3_experiments、fr3_hil_bridge、tools 中的早期桥接和无运动验证工具仍然存在，但不应被当成统一工作台的主启动链。

**4. 硬件与任务适配**

| 方面 | 官方框架/USB 示例 | 当前本地实现及影响 |
|---|---|---|
| 机器人 | Franka 通用环境；控制器默认 Panda 标识 | YAML 改为 fr3_joint1…7、arm_id=fr3；使用本地校准的目标、起点、工作空间和关节约束 |
| 相机 | USB 示例为两路腕部加一路外部 RealSense | 一路腕部 ZED-M、一路外部 ZED 2i；通过适配器保留原相机读取接口，复用物理帧 |
| 接管 | SpaceMouse | Xbox，按 RB 接管；含首次解锁、摇杆死区和动作限幅；GELLO 工具保留，但当前主线是 Xbox |
| 任务 | USB 示例包含拾取与插入，允许学习夹爪 | 当前默认预先夹住插头，模型只学习 XYZ 插入；旋转与夹爪由包装层锁定 |
| 复位 | 示例各自定义复位轨迹 | 持续夹持、先垂直抬升再到采样起点，检查实际到位；超时/停滞阻止进入下一条 |
| 控制 | 归一化相对动作、阻抗控制、安全框 | 增加实际位移范数上限、旋转锁、力/关节速度检查；每个 pose 命令默认不再自动 clearerr |
| 状态中断 | 原始环境请求为主 | 识别 HTTP 503/失效状态，当前 episode 标 incomplete，等待重新确认复位；不重发旧动作 |
| 底层管理 | 单独启动机器人服务 | 工作台增加健康字段核验和显式恢复入口，服务实现仍依赖目录外的控制主机 |

当前统一配置默认单步平移范数上限 8 mm、力阈值 45 N、最大关节速度阈值 0.35 rad/s。它们是软件限制，不能仅凭源码断言真机行为已验收。

ZED 曝光参数也作了适配：使用 SDK 的 -1 或 0–100 标度，不把 RealSense 微秒值直接传入。相机 view 名称 side_policy 被保留，但当前前视 profile 对应的外部相机已经移到前方，名称不等于物理机位。

证据：
- /home/robot/serl_projects/hil-serl-fr3/scripts/zed_capture.py
- /home/robot/serl_projects/hil-serl-fr3/scripts/xbox_intervention.py
- /home/robot/serl_projects/hil-serl-fr3/experiments/plug_insertion/env.py
- /home/robot/serl_projects/hil-serl-fr3/experiments/plug_insertion/wrapper.py
- /home/robot/serl_projects/hil-serl-fr3/upstream/hil-serl/serl_robot_infra/franka_env/envs/franka_env.py
- /home/robot/serl_projects/hil-serl-fr3/hilserl/controller.py

**5. 会影响学习行为的修改**

A. 动作空间收缩，并使用版本区分旧模型。

当前 fixed-xyz-v1 的 actor 和 critic 真正使用 3D XYZ 动作；发给设备时扩成 7D，后三个旋转量和夹爪量为零。它不是让 7D 模型继续学习然后仅在执行时屏蔽动作。目标熵为 -1.5，普通 SAC 路径不使用 grasp critic 和夹爪惩罚。旧 legacy-hybrid-7d-v1 保留原 Hybrid 结构，不能直接把旧动作头、critic、优化器状态当成新模型恢复。

这是对官方已有 SAC 路径的任务定制，而非新提出的 SAC 算法。当前仍继承相对坐标转换、19D 本体状态和预训练 ResNet-10；19D 本身不是本地新算法。

B. 奖励和人工介入数据规则改变。

官方示例主要以分类器输出作为奖励和结束条件。当前分类器概率、状态深度门槛和连续命中条件提供结束建议，最终每条由人标 0/1。新 XYZ 训练非终末奖励为 0，人工成功末步为 1，人工失败末步为 0，末步 mask=0。原始模型奖励另存，不覆盖原始记录。

官方在线经验全入 online 池、人工介入步额外入 demo 池；本地当前还把人工确认成功的终末步放进 demo 池，即使该步由策略执行。旧本地版本存在成功尾段复制/credit 机制；新 XYZ 禁止该机制，并将其配置为关闭。

C. 示教来源与启动条件。

新 XYZ 示教只进入 demo 池，online 池从空开始；这与官方新训练的两池初始化原则一致。本地旧 Hybrid 分支曾把示教也填入 online 池，代码为兼容旧训练保留了这一行为。不能把这个旧本地做法说成原版行为。

新版本核验示教 manifest SHA-256、动作版本、图像版本、原始 step 身份。当前默认是 20 条完整人工成功轨迹、2543 步。至少 100 条真实且唯一的 online transition 才能启动更新；不把图像堆叠支撑槽位误算为实际交互条数。

D. Learner 的更新时机改变。

官方在 replay 达到门槛后持续训练，没有当前这套逐次更新前的 Actor 状态门。本地要求同轮 Actor 身份、进程、采集阶段、2 秒内心跳及在线数据接收同时有效。等待复位、等人工裁决、数据断流或 Actor 退出会暂停更新。CTA=2 仍表示每组一次 critic-only，再一次 critic+actor+temperature；它不等于每采一条数据固定更新两次。

E. 修正 critic-only 阶段的优化器行为。

官方把未选择网络的 loss 设为零，但仍调用其 Adam；零梯度仍可能推进动量、计数并产生参数更新。本地增加 optimizer_names，只调用本阶段选定的优化器，保留共享编码器的合法梯度。当前 XYZ 显式启用 selective_optimizer_updates；旧 Hybrid 保留旧约定。

F. 修正图像 replay 的连续性假设。

原版节省内存的图像 replay 依赖顺序帧复用。稀疏接管数据、不同 episode、重启和历史 credit 不一定相邻。本地使用完整 raw_transition_id 验证同一采集流的相邻 step；不能证明连续时保存样本自身 observation/next_observation，并处理环形缓冲边界。此补丁会影响训练实际看到的图像，属于数据正确性修改。

证据：
- /home/robot/serl_projects/hil-serl-fr3/hilserl/action_contract.py
- /home/robot/serl_projects/hil-serl-fr3/hilserl/episodes.py
- /home/robot/serl_projects/hil-serl-fr3/hilserl/learning_replay.py
- /home/robot/serl_projects/hil-serl-fr3/hilserl/training_gate.py
- /home/robot/serl_projects/hil-serl-fr3/_run_actor.py
- /home/robot/serl_projects/hil-serl-fr3/upstream/hil-serl/serl_launcher/serl_launcher/common/common.py
- /home/robot/serl_projects/hil-serl-fr3/upstream/hil-serl/serl_launcher/serl_launcher/data/data_store.py

**6. 视觉、分类器和资源适配**

官方已有 ResNet 和任务裁切。当前新增了可核验的图像 profile：全图 128、侧视 ROI 和前视 ROI，以及多种输入尺寸。当前默认 insert-front-roi160-v1：

- 腕部原帧 1280×720，裁切 [360,0,720,720]，缩放到 160×160。
- 前视外部原帧 1280×720，裁切 [460,280,440,440]，缩放到 160×160。
- 当前成功分类器使用 wrist_1，对应腕部 160 输入；旧侧视/前视分类器路径仍保留。
- 修改 SAC factory，显式传入 encoder_image_size，确保视觉网络不会把 160 输入隐式缩回 128。
- 实时输入和示教重建共享同一裁切定义；更换机位/ROI 需要不同版本，旧 checkpoint 续训和评估保留原配置。
- 当前 batch=128，demo/online 各64；每池容量20000，官方通用默认分别为 batch=256、每池200000。
- 保存连续原分辨率 H.264 视频，训练 buffer 采用带校验的 LZ4 无损压缩；视频有损编码和训练数组无损压缩是不同处理。

分类器支持本地 PyTorch 历史产物和 JAX 路径，并增加输入契约校验、按运行数据构造训练集和验证指标。存在格式转换脚本不等于每个转换路径已验证可用；本文未运行这些工具。

环境锁文件记录 JAX/JAXLIB 0.6.2、NumPy 2.2.6，以及 CUDA 12.9 工具包组件；官方 README GPU 安装示例固定 JAX 0.4.35。本地多处把 jax.tree_map/tree_leaves 改为 jax.tree_util 对应接口。这里说明锁文件和源码适配，不代表本次重新核验了目录外 Conda 安装。

**7. 通信、运行管理和数据工程**

Agentlace 本地补丁包括：
- 最新策略快照请求 get-network，加 revision，处理 Actor 晚订阅或重连错过广播。
- Actor 启动等待 Learner 参数；Learner 启动阶段重试广播。
- 按 Actor attempt/client ID 管理数据游标，避免重启后低编号被旧游标跳过。
- 已提交数据的 ACK 丢失重试幂等；插入状态不明时停止该数据流，避免重复入库。
- 报告实际数据到达时间，供训练状态门使用。
- 调整 ZeroMQ socket 的线程使用和停止收尾。
- 协议版本本地改为 0.0.3；两端应使用一致依赖，不能随意混用未打补丁的客户端。

工作台新增 CLI/Web 共用配置、独占设备锁、进程身份、训练/评估/示教模式、人工标注、夹爪调试、暂停续训、控制器健康读取、回放和导出。原有 Phase C/VICE/formal 脚本多已成为兼容入口。

每轮保存配置、源码 hash、角色启动快照、日志、模型、经验块、连续双路视频和有效 episode。原始记录包含候选动作、实际选择的动作、控制命令、机器人状态、视频帧引用、时间戳和人工结果；等待/整理/复位与有效采集区分。中断保留 partial/incomplete 数据，默认不当完整轨迹导出。

官方已经有 checkpoint 和 buffer 保存恢复。本地增加明确续训选择、完整状态校验、实际保存回执、同 step 复用和异常收尾；当前默认每轮最多留5个模型，保留最新续训点并参考有效独立评估分数。未落盘经验和崩溃瞬间的视频不保证恢复。

当前独立评估固定 checkpoint、禁用 Xbox 动作、使用确定性策略动作和可复现的 reset seed，单独记录纯策略表现；训练中的人工接管成功不能直接等同独立策略成功率。

**8. 当前默认与历史训练不是同一个版本**

截至本次检查，/home/robot/serl_projects/hil-serl-fr3/config/hilserl.json 指向：
- fixed-xyz-v1
- insert-front-roi160-v1
- demos/fixed_xyz_front_roi160_v1_20260920
- classifier_ckpt_wrist_scene_20260920，classifier_image_key=wrist_1

但已有9月20日18:09、18:24、19:12几轮 train 的顶层配置仍指向9月17日前视示教与 classifier_ckpt_front；19:40 collect 也保存了该旧组合。本次没有逐个读取所有 attempt 快照，因此此处仅报告顶层 run 配置，不能用今天的默认推断每个历史角色实际使用的配置。

版本演进可概括为：早期桥接和无运动验证；GELLO/Xbox 数据工具与7D Hybrid 插入；统一工作台；XYZ动作与受核验示教；侧视ROI；前视机位；腕部成功分类器。文件日期和配置支持这个演进顺序，不足以恢复每一项修改的作者、原因或完整提交历史。

**9. 复现和维护时必须保留的上下文**

- 当前代码明显晚于主仓库最后提交。_run_actor.py、hilserl 包、config/hilserl.json、docs 和多份测试处于未跟踪状态；仅 clone 主仓库 HEAD 无法恢复正在使用的系统。
- 主仓库忽略 /upstream/。上游 HIL-SERL、Agentlace 和控制器的补丁需要单独保存；仅提交根目录修改不够。
- README 和旧配置注释存在过时内容；判断运行路径应以统一启动入口、有效配置及 run/attempt 快照为准。
- 原版已有 fixed-gripper、SAC、视觉裁切、相对坐标、示教采样和保存恢复；本地贡献应准确描述为任务收缩、硬件适配、运行流程和数据/优化器修正。
- 当前泛用 FrankaEnv 的动作缩放已按本地7轴数组修改，不能假定官方所有保留示例无需调整即可直接运行。
- 本目录引用外部控制主机及已安装的控制服务。根据限定范围，本次没有进入该外部目录或验证它的部署内容；本目录本身不是完整硬件部署包。
- 软件改动的存在不证明比原版训练更快或成功率更高；需要固定任务条件、动作/图像版本、接管统计和独立评估做比较。

**建议的后续顺序**

先把当前可运行版本连同嵌套依赖补丁整理成可复现快照，再修正文档与入口说明，随后按算法/数据版本整理每轮训练和独立评估。比较训练质量时，优先区分旧7D Hybrid、新XYZ、侧视/前视、不同示教与分类器，不将不同版本的成功率直接合并。
