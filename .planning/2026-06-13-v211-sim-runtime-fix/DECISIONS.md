# v2.1.1 立项决策记录

> 日期：2026-06-13
> 立项依据：见同目录 `AUDIT.md`（v2.1 sim-build-fork 审计 + P1-P7）

## 用户决策（2026-06-13，本会话三问）

1. **定位**：v2.1.1 **设为 current milestone，暂停 v2.2.1**（v2.2.1 Phase A 的 plan/commit 保留，暂停推进）。
2. **范围**：**全量 P1-P7**（见 AUDIT.md）。
3. **执行/模型路径反转**：
   - **Fable 5 使用权限已被切断**；原指派给 Fable 5 的 plan/discuss/review 改由 **Opus Max / ultracode** 承担。
   - **解除**"禁用 fan-out subagents / workflows"的 2026-06-12 限制，**按需调用**。
   - 已同步更新全局记忆 `prefer-workflows` / `model-stage-policy`。

## 工程决策

- **GSD 落地方式**：本项目 `.planning` 是手维护的 GSD-flavored 布局（date-slug 里程碑目录 + 顶层 PROJECT/ROADMAP/STATE），
  无 `config.json`/`phases/`/`gsd-tools` 初始化。因此 v2.1.1 **沿用本项目既有约定手写艺术品**，
  不跑 `gsd-tools.cjs` / `gsd-roadmapper`（会强加标准 `phases/NN-*` 布局并冲突）。
  `gsd-new-milestone` / `gsd-audit-milestone` 标记为 **skill invoked（作目标指引）+ inline 适配执行**。
- **里程碑目录**：`.planning/2026-06-13-v211-sim-runtime-fix/`。
- **phase6-ready 仍 DEFERRED**，不在本里程碑范围。

## 开放问题的解决（2026-06-13 用户拍板，Phase 2 渲染落地后）

- **P3 canonical 源 → desktop viewer 为准**。以本里程碑修好的 desktop `plug_scene_viewer.py`
  为 canonical 可渲染场景；repo `sim/` 继续作数据/契约层，后续按需把 `sim/scenes/plug_scene.py`
  对齐到 desktop viewer（不反向回写）。
- **P4 schema → producers 产 3 键**。理由：用户要求 sim 资产尽量贴近真实场景，保证 sim-to-real
  policy 能把 sim 训的拿到 real 用。因此 gello_replay / failure_scenario_generator 改产
  3 键（side_policy/wrist_1/side_classifier）匹配真 SERL pkl，verify_sim_data 通过真实生产链产物 + 补 TDD。
- **渲染完善度 → 补完整场景**。FR3+桌+插头+插座完整场景；**复用之前 FR3-蔬菜抓取 sim 项目（droid）
  的 FR3 与桌资产**（`droid.sim.assets.official_fr3_loader.build_fr3_cfg` + `primitive_scene` 的桌/标定布景），
  插头/插座叠加到桌面。
- **P6 fork 合并债 → 暂缓**。当前有并发 ultracode 会话在同一 working tree 改 v2.2.1 Phase A，
  删 worktree/分支有风险；P6 留到并发会话结束后再做。

## 任务语义澄清（2026-06-13，重要 — 影响真机任务设计）

- **插座硬件 = 公牛 GN-109K**（六口、机身 204×92×29mm、白色、6×**五孔**通用孔、总控红色开关、自带尾线三脚插头）。
  实物已用 ZED 2i 实拍 grounding。
- **插入任务对象 = 插排自己的尾线三脚插头**，目标 = 插排自己的某个五孔孔位（**自插回环**）。
  **理由（用户，安全设计）**：用插排自己的尾线插头插回自身，则回路**不带电**，真机 RL 训练更安全。
  → sim 场景**不再放独立两脚插头**；操作物就是排插尾线末端的三脚插头（FR3 抓取它再插入排插五孔）。
- 影响：v2.x 插插头任务的真机/ sim reward 与抓取目标均以"尾线三脚插头 → 自身五孔"为准。
- 资产来源（开放问题，asset-hunt workflow 调研中）：是否直接采用在线/开源排插 3D 资产（USD/GLB），
  还是沿用 viewer 内 primitive 程序化模型；待调研结果定。
