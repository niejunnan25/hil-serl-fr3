# CHECKLIST — C/D 训练前 action/obs/reward 约定对齐

milestone: v2.2.1-hybrid-teleop
probed: 2026-06-16 ~14:00 CST（只读、零运动）
live config: `experiments/plug_insertion/config.py`（已于 2026-06-16 标定）

> 数据来源:本表的 live config 数值（ACTION_SCALE / TARGET_POSE / RESET_POSE / reward gate /
> classifier_keys / proprio_keys）来自 2026-06-16 只读核验 `config.py`，可信。FK 偏差、gripper
> 符号约定、旋转表征（XYZ-intrinsic / xyz-extrinsic / rotvec）、"8D joint state 中间格式" 等
> 细节出自 v2.2.1 既有 recon + `STATE.md`（见第 72 行 `CorrectFK = pinocchio(panda_link8) ∘
> T_offset[z≈0.1034m, Rz -45°]`）+ `inv-gello-polymetis/` 调查，**非**本次 readiness 探测所测;
> 精确数值以 `scripts/hybrid_teleop.py`（CorrectFK）与 demo 导出脚本为准——标 🟠 项即"待对 live 复核"。

## 目的

在线 RB-intervention 的动作空间、demo 数据、policy 三者，必须共享同一套
action / obs / reward 约定，才能进入 Phase C/D 训练。本表逐项核对：哪一项已
经一致、哪一项待核、哪一项不一致；每项给出 live 真值、判定、验证手段、责任人。

判定图例：🟢 一致 / 🟠 待核 / 🔴 不一致。
责任人：我 = claude（可在 Mac/desktop 只读核验）；你 = 操作者（real-machine /
Desk UI）；远端写入 = 需在 zktitan 上对 demo pkl / learner 状态做核验或重导。

## 核对表

| # | 关注点 | live 真值（from FACTS） | 现状判定 | 怎么验（命令 / 文件:行） | 谁清 |
|---|--------|------------------------|---------|------------------------|------|
| 1 | ACTION_SCALE（动作尺度） | live env `ACTION_SCALE = [0.015,0.015,0.015, 0.1,0.1,0.1, 1.0]`（SERL 官方 7D）。在线 RB 路径 `XboxIntervention -> env.step` 走的是同一 live env，尺度一致。OFFLINE gello 转换器 default `action_scale 0.1 / 0.2` 与 live `0.015 / 0.1` 不符。 | 在线 RB 路径 🟢；offline 转换 🔴；已入 demo buffer 的 23 个 `*_success.pkl` 🟠（需以 live 尺度复核后才可信任） | 在线：`experiments/plug_insertion/config.py` 搜 `ACTION_SCALE`，确认 `[0.015,0.015,0.015,0.1,0.1,0.1,1.0]`。offline：定位 gello 转换器 default `0.1/0.2` 并比对。demo：在 zktitan `/nvme/fzt/hilserl-deploy/serl19_insert/*_success.pkl` 读 actions，统计逐步位移幅值是否落在 `0.015 m / 0.1 rad` 量级（而非 `0.1/0.2`）。 | 在线=我（只读核 config）；offline 转换器=我；23 demos 复核=远端写入 |
| 2 | Gripper 语义（夹爪符号约定） | GELLO panda `1=closed`，SERL `-1=closed`（符号相反）。live `setup_mode="single-arm-learned-gripper"`；`XboxIntervention` 的 gripper 取自 RT/LT；wrapper 栈含 `HoldGripperWrapper` + `GripperPenaltyWrapper(penalty=-0.02)`。 | 🟠 待核 | 三者必须同号：RB-intervention 输出 gripper 符号 == demo pkl 内 gripper 符号 == env 约定。验：读 `XboxIntervention` RT/LT -> gripper 映射；读 demo pkl `actions[...,6]` 取值分布（确认 closed 是 +1 还是 -1）；比对 `FrankaEnv` / `GripperPenaltyWrapper` 对 gripper 维的约定。若 demo 来自 GELLO（panda `1=closed`）而 env 用 SERL（`-1=closed`），需翻号。 | 我（核 env/wrapper 与 XboxIntervention 符号）+ 远端写入（核 demo pkl gripper 符号） |
| 3 | FK（正运动学 → tcp 位姿） | offline repo `fk_converter` 约偏 ~50cm（错）。live 用正确 pinocchio：`panda_link8 ∘ T_offset`，z≈0.1034m，`-45deg`。在线 RB 路径不经过 offline FK（走 live env） = 一致。 | 在线 RB 路径 🟢；demo 转换 🟠（必须确认 demo 转换用的是正确 FK，而非 offline `fk_converter`） | 在线：确认 RB 路径不引用 offline `fk_converter`（wrapper 栈直接用 live env tcp pose）。demo：核 23 个 `*_success.pkl` 的 tcp_pose 是否在合理工作空间内（与 `TARGET_POSE=[0.6743,-0.0074,0.1355,...]` / `RESET_POSE=[0.6259,-0.0161,0.1925,...]` 同量级），排除 ~50cm 偏差；确认导出脚本调用的是 pinocchio 正确 FK（z≈0.1034m / `-45deg`）而非 `fk_converter`。 | 在线=我；demo FK 复核=远端写入 |
| 4 | Rotation 表示（旋转表征） | offline = delta **XYZ-intrinsic euler**；SERL obs = **xyz-extrinsic euler**；live `env.step` 动作 = **rotvec（axis-angle）**。三套表征不同。 | 🟠 待核（针对 demo 转换一致性） | live：确认 `env.step` 旋转动作按 rotvec 解释；obs 经 `Quat2EulerWrapper` 出 xyz-extrinsic euler。demo：核导出脚本旋转分量的表征——若 demo 用 offline delta XYZ-intrinsic euler 而 env 期望 rotvec，则 action 旋转维不可直接喂入 buffer，需转换。注意 `euler_2_quat` 非自逆（TARGET/RESET euler 为 seated quat 的数值反解）。 | 我（核 live 表征链）+ 远端写入（核 demo 旋转表征） |
| 5 | Obs schema（观测结构） | live `proprio_keys=[tcp_pose, tcp_vel, tcp_force, tcp_torque, gripper_pose]`（笛卡尔 tcp 空间）；`image_keys=[side_policy, wrist_1]`。中间 offline 格式曾是 **8D joint state**，必须转换到 tcp 空间。 | 🟠 待核（demo 必须是 tcp 空间，非 8D joint state） | live：`config.py` 搜 `proprio_keys` / `image_keys`，确认 `SERLObsWrapper(proprio_keys=...)`。demo：读 `*_success.pkl` 的 observations，确认 proprio 是 `tcp_pose/vel/force/torque/gripper_pose` 笛卡尔布局，且含 `side_policy`+`wrist_1` 两路图像；若仍是 8D joint state，则 demo 与 policy obs 不对齐，必须重导。 | 我（核 live schema）+ 远端写入（核 demo obs 布局） |
| 6 | classifier_keys（reward 分类器相机） | live `classifier_keys=[wrist_1]`；v2.2.1 DECISION 文档写 `[side_classifier]`。 | 🟠 doc/impl 分歧 | 比对 `config.py` 的 `classifier_keys=[wrist_1]` 与 `DECISIONS.md`（本目录）的 `[side_classifier]`。关键：确认已训练的 `classifier_ckpt/reward_classifier.pt`（44MB, 6/15 21:16）到底是在哪路相机上训的——读 `classifier_ckpt/training_history.json` 的相机/输入键，对齐 `classifier_keys`。若 ckpt 训于 side 而 live 喂 wrist_1（或反之），reward 分类器在线推理会错。 | 我（核 config / DECISIONS / training_history.json） |
| 7 | Reward gate（奖励门控） | live（2026-06-16 标定）：`relz = obs["state"][2]`（RelativeFrame z vs `RESET_POSE`）；`cls = sigmoid(classifier_fn(obs))[0]`；`reward = int((cls>0.7) and (relz < -0.05))`。旧 placeholder `<0.22` 在 signed rel-z 上每步触发（no-op bug），现已修复。 | 🟢 已标定（2026-06-16） | `config.py` 搜 reward / `reward_func`，确认门控为 `(cls>0.7) and (relz < -0.05)`，且 `relz` 取 `obs["state"][2]`（RelativeFrame vs `RESET_POSE`）。确认旧 `<0.22` 已不存在（grep 排除）。 | 我（只读核 config） |
| 8 | Demo provenance（demo 来源与一致性） | 23 个 demo `/nvme/fzt/hilserl-deploy/serl19_insert/gello_demo_20260615_*_success.pkl` 正在喂运行中 learner（pid 3417682，RLPD demo buffer，非 BC 预训练）。 | 🟠 待核（覆盖 #1/#2/#3/#4/#5 全部约定） | 在 zktitan 对 23 个 `*_success.pkl` 一次性核验：(a) actions 尺度匹配 live `ACTION_SCALE`（#1）；(b) gripper 符号匹配 env 约定（#2）；(c) tcp_pose 用正确 FK、无 ~50cm 偏差（#3）；(d) 旋转表征与 env rotvec 一致（#4）；(e) obs 是 tcp 笛卡尔 schema、含 `side_policy`+`wrist_1`（#5）。任一不符则当前 demo buffer 已被污染，需重导后 `--resume_from` 重启 learner。 | 远端写入（zktitan 上读 pkl 复核 / 必要时重导） |

## 2026-06-16 实测更新（只读核验 + 一处修复，覆盖上表判定）

- **#1/#2/#5 demo 侧 → 🟢 已核**：核 desktop 源副本 `demos/hybrid/*success.pkl`(30 条)——SERL transition 格式，
  obs = `state(25, tcp 空间)` + `side_policy/wrist_1/side_classifier`(3,128,128)，action ∈ [-1,1]，
  gripper ∈ {-1,+1}(SERL 约定)。**无** 8D-joint / 夹爪反号 / ~50cm-FK 污染。demo buffer 可信。
- **#3/#4 在线 RB 路径 → 🟢**：走 live env ACTION_SCALE + 正确 FK，本就一致(不经离线转换器)。
- **#6 classifier 相机 → 🟢 已修(关键)**：实测分类器 success/failure 判别——`side_classifier sep +0.810`、
  `side_policy +0.810`、**`wrist_1 +0.007`(失明)**。证明 ckpt 训练于侧相机、wrist_1 完全不可分。
  已把 `experiments/plug_insertion/config.py:266` `classifier_keys` 由 `[wrist_1]` 改为 `[side_classifier]`
  (备份 `.bak_classifierkeys_20260616`，py_compile OK)。**未修则 Phase C reward = 噪声、训练不收敛。**
- **#7 reward gate → 🟢**（2026-06-16 已标定，未动）。
- classifier 加载：actor env `hilserl-fr3` 有 jax 0.6.2、无 torch → 走 JAX/Orbax `checkpoint_100` 路径
  （`.pt` 仅回退、本次离线核验用）；首次 actor 起动时确认 classifier 正常加载。

## 结论（原始；🟠 项多数已被上节实测覆盖）

阻塞可信在线训练（must-fix before trusting Phase C/D）：
- #8 demo provenance —— 23 个 `*_success.pkl` 已进运行中 learner 的 demo buffer，但其 action 尺度 / gripper 符号 / FK / 旋转表征 / obs schema 均未对 live 约定复核；这是 #1–#5 在数据侧的总收口，未核前 buffer 可信度未知。
- #1（offline/demo 侧）、#2、#3（demo 侧）、#4、#5 —— 均通过 #8 落到同一批 demo 上；其中任一不符即污染 buffer。
- #6 classifier_keys 分歧 —— `reward_classifier.pt` 若训练相机与 live `classifier_keys=[wrist_1]` 不一致，在线 reward 推理失真，直接影响 RLPD 学习信号，阻塞。

非阻塞（在线 RB 路径本身已对齐 / non-blocking-for-online）：
- #1 在线 RB 路径（`XboxIntervention -> env.step` 走 live `ACTION_SCALE`）🟢。
- #3 在线 RB 路径不经过 offline `fk_converter`🟢。
- #7 reward gate 已于 2026-06-16 标定、旧 `<0.22` no-op bug 已修复 🟢。

简言之：在线 RB 采集这条路径的 action/FK/reward 约定已自洽（#1/#3/#7 在线侧绿）；
真正的风险全在已加载的 23 个 demo 与 classifier ckpt 相机这两处——必须先核
#8（含 #1/#2/#3/#4/#5 数据侧）与 #6，再信任 Phase C/D 的训练结果。
