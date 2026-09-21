# HIL-SERL → Oopsie 离线转换验证

核查日期：2026-09-21。仅处理已有归档，没有启动机器人、训练或上传。

## 结果

- 30 条完成转换：28 条人工标记成功、2 条人工标记失败，共 3,700 步。
- 30/30 通过 Oopsie 官方工具 1.1.0 的严格校验；逐项原始数据复核也全部通过。
- 两路共 60 个视频、7,400 帧，均为 1280×720、10 fps、H.264 CRF19。
- 原始人机来源：1,900 步人工接管、1,800 步策略动作。成功标签不能解释为无人接管成功。
- episode 内容合计 107,871,243 字节，约 108 MB；附带审核报告后的整个包略大。

这是**本地格式与数据对应关系验证包，尚不是已获准的正式提交包**。真实 lab_id、操作员和原标注员身份、夹爪型号等资料尚未齐备；占位内容在每条 HDF5 和说明中都明确标记。未将身份不明的旧人工标签冒称为新的实名标注。

## 两个先行样例

| 样例 | 步数 | 人工 / 策略步 | 人工结果 | 原始终止原因 | 实测采集区间 | 导出视频播放时长 |
|---|---:|---:|---|---|---:|---:|
| 20260917_220321… / 20260917_220349… / 000001 | 95 | 61 / 34 | 成功 | classifier | 16.792666 秒 | 9.5 秒 |
| 20260920_182417… / 20260920_183716… / 000001 | 190 | 62 / 128 | 失败 | time_limit | 30.520455 秒 | 19.0 秒 |

两条样例先通过官方验证和数值、帧索引检查，再批量转换。首、中、末步骤的大图与原始模型裁剪已生成预览并检查。失败样例的相机视野内目标位置与成功样例不同，末段仍未完成插入；保留源人工失败标签，未据此补造失败类别或人类描述。

## 每条轨迹保存什么

每条轨迹包含一个 HDF5、两个 MP4、原 episode/config 快照、verification.json，以及首/中/末的全图 JPG 与原始模型 RGB PNG 预览。

| HDF5 路径 | 形状 | 含义 |
|---|---|---|
| observations/robot_states/cartesian_position | N×7 | 动作前测得的绝对 TCP 位姿 |
| observations/robot_states/joint_position | N×7 | 测得的 7 个关节角，rad |
| observations/robot_states/gripper_position | N×1 | 原桥接器反馈标量，保留原值，未假设为米 |
| actions/cartesian_position | N×7 | controller_commands 中实际发送且返回的绝对末端目标 |
| actions/gripper_binary | N×1 | 恒为 1；本转换约定 1=关闭、0=打开 |
| observations/video_paths/front_cam | 一个相对路径 | side_policy 原始全图，逐步抽帧 |
| observations/video_paths/wrist_cam | 一个相对路径 | wrist_1 原始全图，逐步抽帧 |
| additional_data/tcp_wrench | N×6 | 原始末端力与力矩；N、Nm，保持桥接器坐标系 |
| additional_data/joint_velocity | N×7 | 关节速度，rad/s |
| hilserl/original_state、next_original_state | 各 N×19 | 原模型当前/下一状态 |
| hilserl/next_cartesian_position、next_joint_position | 各 N×7 | 动作后的测量状态，独立于动作标签 |
| hilserl/is_human、policy_revision | 各 N | 人工接管标记与策略版本 |
| hilserl/normalized_selected_action | N×3 | 原始归一化学习动作，供追溯 |
| hilserl/normalized_policy_proposal、normalized_controller_input | 各 N×7 | 原始策略提议与控制器输入 |
| hilserl/time/* | 各 N，int64 | 相机、状态请求/响应、采样、控制步骤和命令的真实纳秒时间戳 |
| hilserl/terminated、truncated、observed_reward | 各 N | 原始逐步终止/截断标记和观测奖励 |
| hilserl/episode_end | N | 完成归档的最后一步；独立于环境 terminated/truncated |

`hilserl` 还保留原始 transition ID、Jacobian、控制命令 JSON、完整 episode/config 元数据和原始文件校验和。官方未使用的动作类型按其规范写为空数据集，不伪造关节目标动作。

绝对末端位姿顺序为 `[x,y,z,qx,qy,qz,qw]`，平移单位米，机器人基座坐标系，四元数无量纲。源代码 `upstream/hil-serl/serl_robot_infra/robot_servers/franka_server.py` 的 `move()` 明确按此顺序向 ROS 的 Point/Quaternion 赋值；状态由 `O_T_EE` 和 SciPy `as_quat()` 产生。转换没有归一化、差分、交换四元数元素或用下一状态替代动作。

夹爪动作来自用户确认的“整个当前任务保持关闭”约定，是恒定期望状态；不声称每一步真的发送了一条关闭命令。真实夹爪反馈独立保留，夹持物体时它不必是零开度。仅将这项约定应用于显式 fixed-xyz-v1 的训练/评估轨迹，未套用到来源不明的旧数据。

原始每步 160×160 模型图像仍在源 NPZ 内。标准输出使用同一 capture_id 对应的 1280×720 全图视频；包中仅另附三组原始模型图像预览，未重复打包全部 NPZ。

## 时间和图像验证

- 每步通过 capture_id 与采集时间戳同时查视频索引，定位具体 segment 和 frame；输入和 next_observation 的引用都要求能匹配。
- 对每一个输出步骤，解码源视频、按保存的图像 profile 重新裁剪，并与 NPZ 中的模型 RGB 对比。3,700 步、两路图像的最大逐图平均像素误差为 3.071 / 255。
- 输出视频逐帧解码检查，帧数必须与步骤数完全一致；输出与源帧的采样像素最大逐图平均误差为 2.482 / 255。
- MP4 是有损压缩，以上检查不代表像素逐字节一致。源视频原为 CRF23，转换成 CRF19 不能恢复原先丢失的像素信息。
- 两路视频一行数据对应一帧；保存各自原始时间戳，不宣称两台相机硬件同步或图像与状态完全同时采样。
- 10 fps 是归档配置的名义控制频率，用于步骤视频轴；没有插入中间动作或对状态插值。真实时间戳未重写。
- 全部 episode 的采集区间合计 535.844641 秒，而步骤视频合计 370 秒；二者不是同一个时间口径，前者也不包含 episode 外的复位和等待。
- 38 个步骤的至少一路输入图像距离动作开始超过 200 ms，均留有诊断标记，未因延迟删除或改写标签。

## 终止、失败与人工接管

成功/失败取自 episode.json 的原人工最终结果；分类器终止原因和奖励仍独立保留。失败不等于文件损坏。

特别核查了一条 72 步失败记录：最后一步的 `terminated=False`、`truncated=False`、`termination_reason=None`，但 episode 汇总原因为 `manual`。源采集代码在两步之间接受人工结束，再完成 episode，因此这是正常差异。转换保留原字段，同时用 `episode_end` 标明归档边界，人工失败标签保持 0。没有把最后一步改造成环境自然终止。

人工/策略来源和策略版本均逐步保存，未把混合轨迹改称纯策略轨迹；也未把失败动作改称专家监督动作。后续若用于 SFT/DAgger，还需按训练目标选择接管片段或设计损失；本任务完成的是 Oopsie 数据组织与验证。

## 未转换清单

盘点 315 条 episode；303 条已完成，12 条不完整。完整记录中转换 30 条，其余保留原档，并逐条列在 `exclusions.json`：

| 原因 | 数量 | 处理 |
|---|---:|---|
| 旧记录缺少明确动作约定 | 233 | 单独列出，需补审历史动作/夹爪映射；不能据此断言数据损坏 |
| 人工示教 collect 模式 | 40 | 本批仅处理在线 train/eval，示教原档另存 |
| 不完整 episode | 12 | 不混入本批输出 |

233 条旧记录包含 197 条成功和 36 条失败，均未删除。正式批量转换的 30 条中没有剩余转换错误。清点初期有一条录制中的记录，最终清点时已成为不完整记录；转换未操作该采集进程。

## 校验器和反例测试

官方工具仓库：https://github.com/oopsie-data/oopsie-data-tools

固定版本：`cca4e6b23f97d2732bd28eaa472d509828e17e4e`（1.1.0）。使用官方 CLI 的严格标注检查，完整 JSON 报告见 `official-validation.json`；两条先行样例报告见 `pilot-official-validation.json`。

另用 `verify_oopsie.py` 对所有 HDF5 与源 NPZ 逐项比对，并用临时副本执行五项反例：

| 故意引入的错误 | 检查结果 |
|---|---|
| 四元数变为全零 | 官方校验拒绝 |
| 关节状态少一行 | 官方校验拒绝 |
| 视频文件路径不存在 | 官方校验拒绝 |
| 用下一帧测量位姿替代实际动作 | 格式本身可通过，源动作一致性检查拒绝 |
| 把 XYZW 四元数改成 WXYZ | 格式本身可通过，源动作一致性检查拒绝 |

反例只修改临时 HDF5 副本，完成后清理；原始数据与已转换数据未改写。结果见 `semantic-audit.json`。

## 重跑命令

在服务器项目目录 `/home/robot/serl_projects/hil-serl-fr3` 执行。工具集位于项目 `artifacts/oopsie-tooling/cca4e6b`，无需改变共享 Python 环境。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
/home/robot/miniconda3/envs/hilserl-fr3/bin/python scripts/export_oopsie.py \
  --root /home/robot/serl_projects/hil-serl-fr3 \
  --output /home/robot/serl_projects/hil-serl-fr3/artifacts/oopsie-validation-20260921 \
  --toolkit /home/robot/serl_projects/hil-serl-fr3/artifacts/oopsie-tooling/cca4e6b \
  --resume

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=artifacts/oopsie-tooling/cca4e6b \
/home/robot/miniconda3/envs/hilserl-fr3/bin/python -m oopsie_data_tools.cli validate \
  --path artifacts/oopsie-validation-20260921/episodes --json

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
/home/robot/miniconda3/envs/hilserl-fr3/bin/python scripts/verify_oopsie.py \
  --root /home/robot/serl_projects/hil-serl-fr3 \
  --output /home/robot/serl_projects/hil-serl-fr3/artifacts/oopsie-validation-20260921 \
  --toolkit /home/robot/serl_projects/hil-serl-fr3/artifacts/oopsie-tooling/cca4e6b \
  --negative-controls
```

转换器可重复添加 `--episode RUN/CAPTURE/EPISODE` 仅处理指定样例；不传则扫描归档。源文件保持只读，输出必须位于项目 artifacts 内且不能位于 artifacts/runs。每条先写 staging，经检查后移入 episodes；失败隔离并记录原因。`--resume` 仅复用相同源摘要、任务文本和兼容映射版本，并复核视频哈希与官方格式；最终 `verify_oopsie.py` 进一步做逐项源数据检查。

## 正式提交前的剩余事项

本批已证明：显式支持的这 30 条历史轨迹具备转换所需的图像、状态、实际末端目标与来源信息，无需为这些数据重新采集。该结论不自动覆盖缺少明确动作约定的旧归档。

仍需填写注册的 lab_id、真实操作员及原标注员资料（无法追溯的身份不能编造），核实夹爪型号，确认任务文字。向接收方如实说明这是在线学习中的混合人工/策略轨迹，并披露非匀速原始时序和名义频率步骤视频。若接收方要求更细的人类失败描述，再由真人复核补充；当前保留原人工二元结果，未生成冒充人工的标注。

官方校验通过只证明其程序检查通过，不等于主办方已经审核接收。所有条目均标为 `submission_ready=False`，没有执行上传。
