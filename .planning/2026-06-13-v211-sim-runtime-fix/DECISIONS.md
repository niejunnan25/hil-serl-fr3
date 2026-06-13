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

## 待执行阶段确认的开放问题（plan-phase 时定）

- P3 canonical 源：以 repo `sim/scenes/plug_scene.py` 为准回写 desktop，还是以 desktop `plug_insertion_scene.py` 为准回迁 repo？
- P4 schema 方向：sim-only 阶段把 verify/contract 放宽到单 `pixels`，还是让 producers 产 3 键以匹配真实 SERL pkl？
- P6 fork 合并债：v2.1 sim 与 v2.2.1 Phase A 是否拆成两条分支，还是接受同分支并在合并时分别处理？
