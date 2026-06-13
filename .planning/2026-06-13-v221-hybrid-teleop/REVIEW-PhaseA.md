# v2.2.1 Phase A — Code Review (2026-06-13)

Reviewer: Opus 4.8 (Fable 5 已停用，按 model-stage-policy 由 Opus 承担 review)。
方法: 5 维度对抗式 review workflow (wf_46bd312a-4ae, 16 agents, ~1.1M tok) + 主会话独立通读
A1/A2 核心文件 + 实测复现。每条 critical/important 经独立对抗校验 (confirmed/refuted)。

## 结论

Phase A 在 STATE.md 已被标 ✅ CLOSED，但本 review 判定 **不应按 CLOSED 信任进入 Phase B**：
- Phase A 自身的"零真机 motion"边界**成立**（没有任何真机被驱动，approval gating 守住了）。
- 但存在 **2 个 critical + 7 个 important** 已确认问题，全部是"mock 测试看不见、真机 Phase B 一接线就触发"的潜伏缺陷。
- 217 passed 的绿测试**掩盖**了这些问题（mock backend 固定快照 + 测试用合法姿态绕开了 CLI 路径）。

建议把 Phase A 状态从 CLOSED 下调为 **code-complete (known issues)**，新增一个 gap-closure 修复
pass（A6）作为 Phase B 的前置闸；critical 两条必须在任何 Phase B motion 前修掉。

## must_have 记分卡

| 维度 | met | not_met | uncertain |
|------|-----|---------|-----------|
| A2 介入/仲裁 | 4/6 | **2** (max_step 经 Xbox 路径可绕过; 设备异常崩循环) | 0 |
| A4 motion/approval | 5/5 | 0 | 0 (但有 1 critical+1 important finding) |
| A1 hub/xbox 契约 | 5/5 | 0 | 0 |
| A3 录制器 | 4/5 | 0 | 1 (pkl metadata key 分叉) |
| 质量/卫生 | — | 1 (死代码/重复常量/未用 import) | 1 (evidence .log 命名) |

## CRITICAL（必须在 Phase B motion 前修）

### C1 — 设备读异常崩溃训练循环  [A2-F1, confirmed]
`scripts/gello_intervention.py:235,241`：`action()` 里 `_ensure_gello()` 和
`self._gello.get_joints()` 无保护，唯一的 `try/except` 只接 `RuntimeError`（:252）。Dynamixel
串口超时 / USB 拔出抛 `OSError/IOError`，直接穿出 `action()` 和 `arbiter.step()`。实测注入
IOError → `arbiter.step RAISED: OSError - USB unplugged mid-training`。真机上这会在机械臂运动中途
halt RL 循环。违反 PLAN-A2 must_have「设备异常永不中断训练循环」。
**修**：device-read + agent.step 用宽 `except Exception` 包住，log 一次后返回 `(action, False)`，
比照 budget-overrun 降级；加一条注入 OSError 的回归测试。

### C2 — motion driver 把归一化 delta 当绝对位姿 POST  [A4-002, confirmed]
`scripts/p2_t3_e2e_motion_driver.py:186-194`：`_send_full` 把 agent 的归一化 7D 动作
`[dx,dy,dz,droll,dpitch,dyaw,gripper] ∈[-1,1]` 直接 `_post_pose` 到 `/pose`，但 franka_server 的
`/pose` 期望**绝对位姿** `[x,y,z,qx,qy,qz,qw]`（见 record_gello_demos_serl.py:170、
verify_franka_server.py:62）。把归一化 delta 当绝对位姿发 → EE 朝原点冲、四元数非单位 → 大幅失控运动。
当前仅因 full 模式经 shell 不可达（见 I6/A4-001）而潜伏，Phase B 一接 `--mode full` 即触发。
**修**：用 `prev_pose + raw cartesian_delta → [x,y,z,quat]` 重建绝对位姿再 POST（或改打 server 真正
当作相对动作解释的 endpoint）；在转换匹配 server 契约前禁止 `--mode full`；加 payload 单位四元数断言测试。

## IMPORTANT（Phase B / 数据采集前修）

### I1 — Xbox 路径无 3mm 限幅  [A2-F2, confirmed]
`scripts/xbox_intervention.py:133-150` 只 `np.clip(action,-1,1)`，不过 agent、无 max_step。仲裁器在 RB
按住时全程路由到 Xbox（teleop_arbiter.py:97-99）。coarse 档 + 满杆 + env pos_scale=0.1 → 单步 ~100mm
（≈33× 的 3mm 上限）。是否安全完全取决于下游 env 是否限幅。违反 must_have「max_step 任何路径不可绕过」。
**修**：(a) 显式文档化 + 集成测试断言下游 env 对 Xbox 通道强制 3mm/步限幅；或 (b) 把 Xbox 动作过一遍
共享的 per-step clamp，使 3mm 不变量在 teleop 工具链内部成立。

### I2 — 双重 env.reset()  [主会话独立发现，workflow 漏掉]
`teleop_arbiter.py:113-117` 的 `reset()` 先调 `self.gello.reset()`（其内部 :353 已调
`self.env.reset()`），再调一次 `self.env.reset()`。三者是同一 base env 的兄弟 wrapper（仲裁器自己
`self.env.step()`，只用 wrapper 的 `.action()`）。实测：一次 `arbiter.reset()` → `env.resets==2`。
真机每个 episode 归位两次（双重 homing 运动 + 2× reset 耗时，第二次可能干扰第一次）。
**修**：`arbiter.reset()` 只调 `self.gello.reset()`（它已负责 env.reset），删掉第二次 `self.env.reset()`；
或重构为链式 wrapper。加 CountEnv 断言 reset 恰好一次。

### I3 — 介入误标污染 HIL-SERL 数据  [XCQ-04, confirmed；与主会话独立发现一致]
`teleop_arbiter.py:100-110` 丢弃 wrapper 的 `replaced` 布尔（`expert, _ = self.gello.action(...)`），
只凭按键贴 `intervene_device`。当 LB 按住但 GELLO 降级返回 `(action,False)`（budget 超限/未动）时，
仲裁器仍标 `DEVICE_GELLO` 且 `info["intervene_action"]=policy_action` → 把策略自身动作当人工介入样本
写进 intervention buffer，污染 HIL-SERL。test_teleop_arbiter.py:196-213 反而把这个错误行为断言成预期。
**修**：传播 `replaced`——为 False 时返回 `DEVICE_NONE`（或 `gello_downgraded`），不写 intervene_action；
改对应测试。

### I4 — 混合录制器 Xbox 段不记录 Xbox 夹爪  [XCQ-02, confirmed]
`record_hybrid_demos.py:392`：`gripper_states.append(float(raw_all[-1]))` 在循环公共尾部，无论哪段都从
GELLO 第 8 通道取夹爪——Xbox 段操作员的 RT/LT 被丢弃。插入精细段的夹爪信号是错的/陈旧的。
**修**：MODE_XBOX 下从 Xbox state 取夹爪（RT=+1/LT=-1，比照 xbox_intervention.py:142-148）；加测试；修文档。

### I5 — 混合录制器 Xbox 段用关节空间映射，与 XboxIntervention 的笛卡尔空间不一致  [XCQ-01, confirmed→verifier 改判 minor]
`record_hybrid_demos.py:226-237` 的 `_xbox_action_to_joint_delta` 把摇杆直接映射到**关节**增量；而
`XboxIntervention._state_to_action` 把同样的轴映射到**笛卡尔**通道。docstring(:22) 声称"用
XboxIntervention._state_to_action 映射"实则没有（那几个 import 全是未用 import）。录制的 demo 与策略期
Xbox 介入处于不同动作空间，违背"single source of truth"。
**修**：让录制器调 XboxIntervention._state_to_action 再经 FK/Jacobian 转关节（与 GELLO 段一致）；或显式
标注为 B4 占位 + 加等价性测试；至少删掉误导 docstring 与未用 import。

### I6 — 16_*.sh motion 模式静默跑成 dry-run  [A4-001, confirmed]
`scripts/setup/16_gello_e2e_motion_test.sh:310-316` 调 driver 时**不传 `--mode`**，driver 默认 dry-run
（:203）。实测 `bash 16... motion` → rc=0、log 显示 `mode=dry-run`、mock server 收到 0 次 /pose。shell 的
rc 分发还处理 dry-run 永不会产生的 rc=8/9。宣称的"FULL E2E motion stream"从未真正跑过。对 Phase A 是
安全正向（强化 must_have 1），但是功能缺口。
**修**：motion() 显式传 `--mode full`（Phase B 解锁时才真流），或明确传 `--mode dry-run` 并改 banner/rc 处理；
加测试断言 Phase A 下 `16 motion` 产生 0 次 /pose。

### I7 — CLI dry-run 自检静默产出空  [A3-F1, confirmed]
`record_hybrid_demos.py:472` `q0=np.zeros(7)`，首 tick 因关节限位（joint4/6 限位不含 0）被 clip 出
0.19rad 跳变（63× max_step）→ gello_step 在 step 0 返回 ok=False → 循环 break、0 样本、main 静默无输出。
宣称的"5 秒合成 episode 端到端自检"啥也没做且无诊断。单测因用合法姿态 q0 而绕过。
**修**：main() 用 `FR3_DEFAULT_JOINTS`（:84 已有）替代 zeros；run_dry 返回 None 时打印 `[ABORT] reason=...`。

### I8 — Phase A gate evidence 被 git 忽略、未跟踪  [XCQ-05, confirmed]
`.gitignore:40 = *.log`，而 evidence 目录本身命名 `phase-a-gate-{local,desktop}.log/`（目录名以 .log
结尾）→ 整棵 evidence 树被忽略；`git ls-files .../evidence/` 为空。HEAD commit d42cd8f 与 STATE.md:72
都把它们当"已保存证据"引用，但 repo 里其实没有。
**修**：evidence 目录重命名为不以 .log 结尾（如 `evidence/phase-a-gate-local/`），内部 per-suite *.log 仍被忽略
但目录结构可跟踪；或加 `!.planning/**/evidence/**` 反忽略 + force-add 精选摘要；并改 commit/STATE 措辞。
（本 REVIEW 文件已入库，部分修复此条。）

## MINOR / 卫生

- `record_hybrid_demos.py:82-84` 重复 `FR3_LOWER/UPPER_LIMITS/DEFAULT_JOINTS`（4+ 文件各自重定义）→ 抽公共模块。
- 死代码 `record_hybrid_demos.py:177-179 _clip_to_limits` 定义未调用。
- 未用 import：`DX_IDX..GRIPPER_IDX, SCALE_FINE, XboxIntervention`（这正是 I5 的根因——声称用却没用）。
- pkl/npz：两者都用 `np.savez`（非 pkl）；8 个核心轨迹数组 key+shape 等价，但 metadata key 分叉
  （gello 用 `meta_*` 前缀，hybrid 用裸名 + 新增 active_device/switch_step/... 且丢了 meta_duration/
  meta_joint_signs/meta_server_url）。是否要紧取决于下游 SERL loader 读哪些 key——需确认。

## REFUTED（对抗校验后排除，不计入）

- XCQ-03「仲裁器一步 poll 2-3 次 hub、'单一快照'是假的」→ 对抗校验改判 minor：mock 下确定性、GELLO 路径快照
  使用一致；其实质风险已被 I3（confirmed）覆盖。仅作 minor 鲁棒性备注。

## 建议修复顺序（gap-closure A6）

1. **C1 + C2**（critical，阻断 Phase B）
2. **I2 双重reset + I3 误标 + I1 Xbox限幅 + I4 夹爪 + I5 动作空间**（important，污染数据/真机行为）
3. **I6 + I7**（功能缺口，影响联调与自检可信度）
4. **I8 evidence 命名 + minor 卫生**

C1/C2 修完前，Phase B 的任何 `--mode full` / 真机 motion 均不应启动。
