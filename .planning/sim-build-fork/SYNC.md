# sim-build-fork → mainline 同步协议 (SYNC.md)

> **milestone**: sim-build-fork v2.1 finalization (2026-06-11)
> **base branch (v2.0)**: `074dd68`
> **fork branch (v2.1)**: `ai/20260228-optimize-cge`
> **目的**: 把 sim fork 的代码/脚本/docs 同步回主线 (主分支), 触发 phase6-ready gate 评估

---

## 1. `sim-code-ready` 状态

**`v2.1 sim-code-ready: PASS`** (2026-06-11)

**触发条件 (per spec section 1.9 + D5b + D11)**:
- ✅ A1-A10 全部 done (8 个 sim-side 改造 + domain randomization + failure_scenario + verify schema)
- ✅ A11/A12 schema smoke pass (mock-only, NOT readiness)
- ✅ 3 follow-up issues from codex `REVIEW-final.md` 全部 resolved (#1 ABI pin, #6/#7 标签语义)
- ✅ L1 isolation gate: clean (无 sim_remote 硬编码 / panda_joint / /home/robot 残留)
- ✅ VERIFY.md 出现 `v2.1 sim-code-ready` 显式 label
- ✅ SYNC.md (本文件) 落地

**宣告 subagent**: sim-build-fork finalize agent (2026-06-11)
**证据位置**:
- `.planning/sim-build-fork/VERIFY.md` — top-of-file status line + "Final Review" section
- `.planning/sim-build-fork/PROGRESS.md` — "A9-A12 + 3 fixes 完成 ✅" section
- `.planning/sim-build-fork/SYNC.md` — 本文件

---

## 2. `phase6-ready` 要求 (NOT in this fork)

**`phase6-ready: DEFERRED`** — 不在 sim fork 范围, 需后续在主线 (real-machine) 满足以下 **全部 4 项**:

| # | 要求 | 判据 | 当前位置 |
|---|------|------|----------|
| 1 | Real pkl | 50%+ positive ratio, 真实图像 (来自 real FR3 8010 录制) | 暂无; 需真机部署后回采 |
| 2 | Balanced fixture | 50/50 pos/neg (real 阳性 + sim 阴性) | A11 mock 80% pos_ratio 不算 |
| 3 | Confusion matrix | ROADMAP Phase 6 precision ≥ 0.85, recall ≥ 0.85 | A12 mock 跑通不算 |
| 4 | User approval gate | per spec D11 (人工签字) | 待用户合并 + 跑真实 pkl + 验收 |

**任一缺失 → `phase6-ready` 不构成**。
**A11/A12 mock smoke ≠ readiness**。

---

## 3. Files to merge back to mainline

**所有 sim-side work 已在 `ai/20260228-optimize-cge` 分支上, 包含 43 commits 自 `074dd68` (v2.0 base)。**

### 3.1 代码 (sim/)
```
sim/data/contract.py                       [A1+A7] 单一来源 (state 25D, image 3-key, action_scale, thresholds)
sim/data/verify_sim_data.py               [A10]   真实 pkl schema 校验
sim/data/failure_scenario_generator.py    [A9]    4 classes (mis_alignment/angle_offset/insufficient_force/drop)
sim/data/plug_reward_labeler.py           [A7]    8mm/2mm/5° thresholds
sim/data/gello_replay.py                  [A2+A4+A5] state 25D, action_scale from contract, fr3_joint
sim/scenes/plug_scene.py                  [A3+A8] PlugSceneCfg + 3 cameras + DR
sim/safety/feasibility_checker.py         [A1]    隔离 gate
sim/safety/runtime_check.py               [A1]    隔离 gate
sim/obs/camera_wrapper.py                 [A1]    ported, drop droid.sim legacy obs_dict
sim/transforms/delta_actions.py           [A1]    ported (8D, rewritten in A2)
sim/kinematics/fr3_fk.py                  [A1]    ported
sim/scripts/gen_mock_real_pkl.py          [A11]   mock fixture
sim/scripts/test_domain_alignment.py      [A11]   smoke test
sim/scripts/test_mixed_training.py        [A12]   smoke test
sim/scripts/requirements.txt              [#1 fix] numpy<2.0 + scikit-learn==1.3.0 ABI pin
```

### 3.2 计划/Review 文档 (.planning/sim-build-fork/)
```
DESIGN.md                                 spec section 1
PLAN.md                                   12 plans A1-A12
PROGRESS.md                               推进日志 + Final Review
VERIFY.md                                 sim-code-ready 证据 + Final Review
REVIEW-{design,impl-batch1,impl-batch2,reception,final}.md
plans/PLAN-{A1..A12}.md                   12 plan files
SYNC.md                                   本文件
```

### 3.3 Merge 建议

**直接 fast-forward / 3-way merge `ai/20260228-optimize-cge` → mainline**:
- L1 isolation gate clean, 不污染 real 侧
- 43 commits 全部为 sim-side work + 文档, 无 real 侧 import
- 合并后主线上重跑 A11 + A12 with real pkl, 走 phase6-ready gate 评估

**Or cherry-pick 关键 commits**:
- `e35c323` ABI pin (env critical)
- `4cf0e1a` A10 verify_sim_data (entry point for real pkl validation)
- `b9a28c3` A9 failure_scenario_generator (training data augmentation)

---

## 4. What's NOT in this fork (out-of-scope)

| 范围 | 原因 | 接手位置 |
|------|------|----------|
| **Background domain randomization (DR) runtime tuning** | A8 DR 范围 = light/camera/plug xy jitter; 实际 IsaacLab scene 实例化 + 物理仿真调优需 desktop env (fr3-desktop-ts) | mainline agent + IsaacLab GUI |
| **Real pkl validation** | 本地无 real FR3 8010 数据, A11 mock 80% pos_ratio 仅作 schema smoke | mainline + 真机录制 |
| **Env-specific runtime tests** | numpy 2.x + sklearn 1.5.x ABI 冲突 (#1); 需 mainline conda env 验证 A12 真实跑通 | mainline conda env |
| **ROADMAP Phase 6 precision/recall 评估** | 需 real pkl + balanced fixture, 不在 sim fork 范围 | mainline + phase6-ready gate |
| **User approval gate** | 人工签字 per spec D11 | 用户合并时 |
| **IsaacLab scene instantiation tests** | 本地无 isaaclab 安装; cfg 字段正确性 = 单元测试覆盖; scene 实例化 = desktop env 验证 | fr3-desktop-ts + IsaacLab |

---

## 5. Open issues for successor agent (接手清单)

### 5.1 优先 (blocking phase6-ready)

1. **A12 ABI runtime verify**
   - 现状: `sim/scripts/requirements.txt` pin `numpy<2.0` (1.26.4) + `scikit-learn==1.3.0`
   - 需做: 在 mainline conda env 安装 pin 版本, `python -m pytest sim/scripts/tests/test_mixed_training.py -v` 跑通
   - 阻塞: 不阻塞合并, 但阻塞 phase6-ready 评估 (混淆矩阵需要 sklearn)

2. **Real pkl A11/A12 re-run**
   - 现状: A11 mock pkl pos_ratio=0.8, A12 mock 跑通 ≠ readiness
   - 需做: 替换 mock pkl 为 real FR3 8010 录制 pkl (50%+ positive, 真实图像)
   - 阻塞: 不阻塞合并, 但阻塞 phase6-ready

3. **ROADMAP Phase 6 precision/recall ≥ 0.85 验证**
   - 现状: 数值未测 (无 real pkl)
   - 需做: real pkl + balanced fixture → confusion matrix → precision/recall
   - 阻塞: phase6-ready 必经项

### 5.2 非优先 (post-merge cleanup)

4. **sim_remote/ 目录完全清理**
   - 现状: A1 删了 `sim_remote/sim/` 重复目录 + `sim_remote/obs/obs_dict.py` + `sim_remote/data/demo_relative_replay.py` + `sim_remote/assets/configuration/`
   - 残留: `sim_remote/` 根目录 + `.pyc` 缓存 + 可能的小文件
   - 需做: 合并后二次扫描 + 清理

5. **IsaacLab scene runtime test**
   - 现状: cfg 字段正确性 = 单元测试覆盖 (PASS)
   - 需做: fr3-desktop-ts 上 `isaaclab` 实际实例化 PlugScene, 跑一段 episode
   - 阻塞: 不阻塞合并, 阻塞 sim 真接入训练流水线

6. **A8 DR 物理仿真调优**
   - 现状: A8 DR 范围已写入 cfg (light 800-1200, camera ±5°, plug ±1cm xy)
   - 需做: 实际 IsaacLab 跑 sim-only training, 看 DR 是否影响 policy 收敛
   - 阻塞: 不阻塞合并, 阻塞 production-ready

7. **Codex #2 / #3 / #4 / #5 follow-up**
   - 现状: REVIEW-final.md 列了 7 项, #1/#6/#7 已关闭; #2/#3/#4/#5 已在 PROGRESS.md/VERIFY.md 标注
   - 需做: 接手 agent 评估是否需要进一步 follow-up

---

## 6. 退出条件检查 (checklist)

- [x] `sim-code-ready: PASS` 显式标注 (VERIFY.md top + Final Review)
- [x] `phase6-ready: DEFERRED` 显式标注 (VERIFY.md + PROGRESS.md + SYNC.md)
- [x] 12 plans A1-A12 全部 done (PROGRESS.md + VERIFY.md)
- [x] 3 follow-up issues resolved (codex REVIEW-final #1/#6/#7)
- [x] L1 isolation gate clean (无 sim_remote / panda_joint / /home/robot 残留)
- [x] SYNC.md 落地 (本文件)
- [x] 43 commits 自 v2.0 base (074dd68..HEAD)
- [x] 文档完整 (DESIGN/PLAN/PROGRESS/VERIFY/REVIEW-* 全部就位)

**→ sim fork v2.1 finalize 完成, ready for mainline merge。**
