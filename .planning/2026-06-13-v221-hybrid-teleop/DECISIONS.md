# v2.2.1 立项决策记录（grill-me 九分支，2026-06-13）

会话：本地 Claude Code session（前序调查 sess 28c86798-0a4e-457b-b47c-a7134b970b2d）。
方法：/grill-me 逐分支询问 + 代码核查 + 文献核查；每项含定案与依据。

## 1. 范围
**定案**：工具链 + Stage-1（Xbox 纯插入）训成 + 完整过程（端到端）训成；v2.2.1 接管 v2.2
未启动的 Phase 3–5；Phase 6 (sim) 留 v2.2。
**依据**：用户选"完整过程一起训"；旧 Phase 3–5 为 GELLO-only 设计，与混合方案冲突，避免双线重复劳动。

## 2. Policy 结构
**定案**：纯端到端单 SAC policy；完整过程 demo 全新采集，Stage-1 数据不复用；两段拼接仅作退路。
**依据**：用户在三选项讲解后选"纯端到端"。上游 HIL-SERL usb_pickup_insertion 同构任务已验证；
拼接方案交接缝是手工的，与"学习完整过程"目标不符。

## 3. π0.5 路线
**定案**：v2.2.1 纯 SAC；v2.3 候选 = RLDG（SAC specialist 生成数据 → zktitan OpenPI LoRA 微调 pi05）；
ConRFT/π_RL 式直接 VLA 在线 RL 为 v2.3 备选研究线。
**依据**：RLDG (arXiv:2412.09858, Berkeley RAIL) 在 connector insertion 上 RL 数据微调 generalist
比人类示教高至多 40%；π_RL (arXiv:2510.25889) 依赖 320 并行仿真环境，单真机不可行；
ConRFT (RSS 2025) 真机可行但需换整个训练栈，周级工程。zktitan 已有 OpenPI workspace
(/nvme/njn/workspace/openpi/, serve_policy.py:8010, pi05 base checkpoint)，RLDG 路线零浪费，
直接服务 FR3-8010 目标。

## 4. Xbox 硬件
**定案**：已有手柄，USB 有线接 fr3-desktop-ts。
**核查**：当前 desktop 无 /dev/input/js*、无 Xbox USB 设备；hilserl-fr3 env 已有 pygame 2.6.1。

## 5. 示教切换机制
**定案**：手动按键（RB）触发、单向 GELLO→Xbox、同一 episode 连续录制；device/switch_step
写入 metadata，不进 observation。
**依据**：确定性零误触发；操作员自主决定"临近插座"时机，无需阈值标定。

## 6. 训练期介入设备
**定案**：分段双设备——前半程（抓取/搬运）GELLO，后半程（精插）Xbox。用户基于任务两段式
结构的明确偏好，覆盖单设备推荐。
**配套工程**（必做）：
- GelloIntervention max_total_delta=30mm 累积预算改造 + env.reset 时 agent.reset（现仅首连 reset）
- 防误触：GELLO 介入需按住 Xbox LB 使能（1mm 移动阈值裸用必误触发）
- 双设备互斥仲裁：同 tick 双输入时 Xbox 优先
**代码核查**：GelloIntervention 为 delta-jog 模式（pos_scale=0.1, FK 增量），mid-episode 接管
无需 leader-follower 对齐——早先"必须对齐"的论断对该实现不成立，已更正。
**文献**：HIL-SERL 原生用 SpaceMouse 瞬时介入；leader-arm 作 RL 介入设备无成熟实践。

## 7. Reward 设计
**定案**：单终点 binary classifier（side_classifier 相机 + state 条件）；抓取不单独奖励，
仅作 metrics；保留 gripper penalty wrapper。升级预案单向门：加大介入 → 补 demo →
两阶段 reward（需另标定抓取判据）。
**依据**：上游 usb_pickup_insertion 实测同构（reward = sigmoid(classifier)>0.7 且 state 条件）；
价值回传 + 50/50 demo 采样 + 介入补课足以学会抓取段；中间奖励有 hacking 风险。

## 8. 数据量
**定案**：Stage-1 Xbox demo 20 条；完整混合 demo ≥25 条；classifier ≥200 正 / ≥600 负
（两阶段共用，插入成功定义相同）。沿 v2.2 roadmap 标准。

## 9. Motion 批准
**定案**：一次性联合验收 session：GELLO e2e（P2-T3）→ Xbox e2e → 切换 e2e；
E-stop 就位、用户在场；沿用 approval 环境变量模式。

## 附带事项
- 2026-06-12 事故：v2.2 planning 文件（未 commit）随 working-tree reset 丢失，已于
  2026-06-13 从 transcript 重建；此后 .planning 变更随做随 commit。
- 执行方式（2026-06-13 更正，覆盖 2026-06-12 禁用令；2026-06-15 对齐）：fan-out subagents 与
  ultracode workflows 已解禁、按需调用（审慎提醒：2026-06-12 >40GB 事故，先清残留 worktree）；
  Fable 5 权限切断，grill/discuss/plan/review 改 Opus Max/ultracode，execute 仍 Opus 4.8 xhigh；
  Superpowers 规范 + GSD 官方流程仍为骨架；真机运动执行仍顺序 + human-in-loop 握手。
  见 model-stage-policy / prefer-workflows / v2.1.1 DECISIONS.md。
