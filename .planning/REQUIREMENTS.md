# REQUIREMENTS — hilserl-fr3 (v2.2 + v2.2.1)

> 重建说明：原 v2.2 REQUIREMENTS.md 在 2026-06-12 working-tree reset 中丢失，
> 2026-06-13 重建。INFRA/GELLO 状态来自 session 28c86798 的 Phase 1/2 关闭记录；
> DATA/TRAIN/EVAL 按 v2.2.1 决策重划（原 GELLO-only 表述废弃）。

## INFRA（v2.2 Phase 1）✅ 全部关闭 2026-06-12

- [x] INFRA-01 libfranka 0.15.3-1 read-only gate
- [x] INFRA-02 ROS Noetic + franka_ros 0.10.1
- [x] INFRA-03 serl_franka_controllers FR3 适配
- [x] INFRA-04 franka_server guarded HTTP contract
- [x] INFRA-05 ZED 2i external (36276705) frame gate
- [x] INFRA-06 ZED-M wrist (13132609) frame gate
- [x] INFRA-07 ZEDCapture adapter

## GELLO（v2.2 Phase 2）5/6

- [x] GELLO-01 GelloIntervention wrapper（delta-jog，34 tests）
- [x] GELLO-02 GELLO→FK→Cartesian delta 流程（19 tests）
- [ ] GELLO-03 端到端 GELLO→FrankaEnv→FR3（→ v2.2.1 B1，motion 批准后）
- [x] GELLO-04 夹爪映射（第 8 轴）
- [x] GELLO-05 安全机制移植标定（max_step 2.98mm / max_total 29.93mm @10Hz，B4 重标）
- [x] GELLO-06 record_gello_demos_serl HTTP API 迁移（33 tests）

## XBOX（v2.2.1 Phase A/B）

- [x] XBOX-01 XboxIntervention wrapper（pygame，按住 RB 介入，deadzone，7D delta） — A1; 17 tests green
- [ ] XBOX-02 手柄 USB 接入 fr3-desktop-ts 并被识别（human，物理）
- [ ] XBOX-03 Xbox e2e 真机验证（B2）
- [ ] XBOX-04 fine-scale 档位 + deadzone 实标（B4）

## HYBRID（v2.2.1 Phase A/B）

- [x] HYBRID-01 GelloIntervention 介入改造：预算 + per-episode reset + LB 使能 + 互斥仲裁 — A2; 18 tests green
- [x] HYBRID-02 record_hybrid_demos：单 episode 内 GELLO→RB→Xbox 连续录制，metadata 含 device/switch_step — A3; 11 tests green
- [ ] HYBRID-03 切换 e2e 真机验证：无跳变、仲裁正确（B3）

## DATA（v2.2.1 Phase C/D）

- [ ] DATA-01 IMAGE_CROP 标定（真实 ZED 视野）
- [ ] DATA-02 classifier 数据 ≥200 正 / ≥600 负（Xbox 采集，两阶段共用）
- [ ] DATA-03 Stage-1 demo 20 条（Xbox）
- [ ] DATA-04 完整过程混合 demo ≥25 条（GELLO→Xbox）
- [ ] DATA-05 全部转 SERL pkl（128x128 图像 + 完整 state 字段）格式验证通过

## TRAIN（v2.2.1 Phase C/D）

- [ ] TRAIN-01 reward classifier P/R ≥ 0.90（单终点 + state 条件）
- [ ] TRAIN-02 Stage-1 SAC 训练（actor=fr3-desktop-ts, learner=zktitan）
- [ ] TRAIN-03 actor-learner 网络通信验证（经 desktop 跳板，端口/RPC/checkpoint 同步）
- [ ] TRAIN-04 完整过程端到端 SAC 训练；介入：前半程 GELLO / 后半程 Xbox

## EVAL（v2.2.1 Phase C/D）

- [ ] EVAL-01 Stage-1 ≥3 次自主插入成功
- [ ] EVAL-02 完整过程 ≥3 次自主 抓取→插入 成功（v2.2.1 完成判据）
- [ ] EVAL-03 失败 case 记录（观测+动作+原因）

## SIM（v2.2 Phase 6，与 C/D 可并行）

- [ ] SIM-01~07 沿原定义：IsaacLab 失败场景负样本 ≥500，混合训练 P/R ≥ 0.85
