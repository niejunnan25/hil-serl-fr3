# v2.1 (sim-build-fork) 审计 — v2.1.1 立项依据

> 审计日期：2026-06-13
> 审计人：Claude (Opus 4.8 [1m])，inline 顺序审计（gsd-audit-milestone skill invoked 作目标指引；未跑会 spawn 子代理的官方编排）
> 来源：session `9e40c72c-5013-4f50-b196-e1133cbdf0cb`（2026-06-11 起，1635 行 transcript）+ `.planning/sim-build-fork/` + 本地 repo 实测 + fr3-desktop-ts SSH 实查

---

## 1. v2.1 是什么

ROADMAP 定义：**v2.1 = sim 主线 fork（sim-code-ready PASS）**，证据目录 `.planning/sim-build-fork/`。
意图：把 IsaacLab 插头插入 sim 侧的代码/脚本/契约做就绪（A1-A12），单测通过、L1 隔离干净，
待合并回主线后再做 phase6-ready 实测（real pkl + precision/recall ≥ 0.85）。
该 fork 在 session 9e40c72c 中用 fan-out subagents + workflows 执行（即 2026-06-12 内存事故的来源会话之一）。

## 2. 一句话结论

v2.1 的 `sim-code-ready: PASS` 标签**仅对"repo 单测/契约就绪"成立**；对 sim milestone 的
**真实意图（IsaacLab 里能渲染并跑插头场景）并未达成**。存在一个文档承认却未修的真实 schema bug，
外加证据丢失、分支混杂、desktop orphan 进程等工程债。

## 3. 声明 vs 实际（实测对照）

| v2.1 声明 | 实测结果 | 判定 |
|---|---|---|
| A1–A12 代码就绪 | `sim/` 全模块在位，commit 于 `ai/20260228-optimize-cge` | 成立 |
| `111 passed, 2 skipped` | 当前分支 **93 collected / 91 passed / 2 skipped** | 文档漂移（套件仍绿，计数不符；疑为 workflow worktree 聚合计数） |
| `test_contract.py` 3 个 threshold 失败待修 | `contract.py` 已含 `INSERTION_DEPTH_THRESHOLD=0.008`/`XY_TOLERANCE=0.002`/`ANGLE_TOLERANCE_DEG=5.0`，0 失败 | 已修，文档过时 |
| `contract.py` = 图像 schema 单一来源（3 键） | 生产链实写单 `pixels` 键，验证器要 3 键 | 内部 bug，未修 |
| L1 隔离对 5 个目标文件干净 | 成立（另 4 个 out-of-scope 文件残留，已记录） | 成立 |
| IsaacLab runtime 验证 = DEFERRED | desktop 跑的是另一个脚本，从未与 repo 对齐 | 目的未达成 |

## 4. 当前问题清单（P1-P7，含证据 + 修复方向）

### P1 [阻塞·desktop] IsaacLab 插头场景黑屏 / 卡启动
- session 尾部用户："截图里面是纯黑的呀"；此前会话却声称"看到 FR3+桌+插头+插座" → verify 造假风险。
- session 死于 context 超限（`API Error: 400 ... context window exceeds limit`），调试中途未定论。
- desktop 脚本 `/home/robot/plug_insertion_sim/sim-scene/plug_insertion_scene.py`（23KB，2026-06-12）三个黑屏候选根因：
  1. 坐标系标 **"Y 轴朝上"**，IsaacLab/USD 默认 Z-up → 相机取景大概率指向空处；
  2. 腕相机 prim 路径 `{ENV_REGEX_NS}/Robot/panda_hand/WristCamera`（`panda_hand`），与 A5 的 `panda_*→fr3_*` 矛盾，若机器人 USD 是 fr3 则相机挂到不存在 prim；
  3. `DistantLightCfg` 在 IsaacLab 5.1.0 无 `direction` 字段（脚本注释已承认）→ key/fill 光未定向、欠曝。
- 修复方向：逐项排查 + 干净重启采集首帧 + 产出用户可确认的非黑屏截图。

### P2 [活跃·desktop] 上次死会话的 orphan 进程仍在跑
- 实查（2026-06-13）：PID 2876313 跑 `plug_insertion_scene.py`，存活 ~73349s（~20h），占 **4637 MiB GPU**（nohup+disown 孤儿）。
- 进程树：2876297(bash -c …nohup…) → 2876300 → 2876302(isaaclab.sh) → 2876313(kit)。
- 修复方向：清掉孤儿 + 给 GUI launcher 加进程组/PID 文件管理，避免再孤儿。

### P3 [架构] repo `sim/` ↔ desktop 场景脚本分叉
- v2.1 "sim-code-ready" 建立在 repo `sim/scenes/plug_scene.py`（3 相机 + DR + fr3_joint）上。
- desktop 实际渲染的是独立手写 `plug_insertion_scene.py`（panda_hand / Y-up / 无方向光），从未与 repo 对齐。
- 这是 P1 的根：repo 的 sim-code-ready 修复（fr3_joint、DR、3 相机）从未传播到实际运行的场景。
- 修复方向：定 canonical 源（repo 还是 desktop 脚本），单向收敛。

### P4 [真 bug·repo] 图像 schema 不一致（`pixels` vs 3 键）
- 实测：`sim/data/gello_replay.py:315` `replay_pure_fk` 返回 `{"state":…, "pixels":…}`（单键）；transitions 存 `"pixels"`（行 469/473/626/630）。
- `failure_scenario_generator` 复用 replay_pure_fk + `_PIXELS_PLACEHOLDER`（行 38）→ 产物单 `pixels`。
- 但 `sim/scripts/verify_sim_data.py` + `contract.VALID_PKL_IMAGE_KEYS = (side_policy, wrist_1, side_classifier)` 要 3 键 → **验证器拒绝自家生产链产物**。
- A11/A12 smoke 能过仅因 mock 生成器单独造 3 键数据，绕过真实链路。PROGRESS.md 已标"需 A9.1 修"，未修。
- 修复方向：统一 schema（sim-only 阶段接受单 pixels，或 producers 产 3 键），让 verify 通过真实产物 + 补 TDD。

### P5 [证据] 规划文档丢失 + 现实漂移
- `sim-build-fork/` 磁盘仅剩 `PROGRESS.md / SYNC.md / VERIFY.md` + `plans/PLAN-A12.md`。
- SYNC.md 引用的 `DESIGN.md / PLAN.md / REVIEW-{design,impl-batch1,impl-batch2,reception,final}.md / plans/PLAN-A1..A11.md` **全部不存在**（2026-06-12 working-tree reset 丢失）。
- "111 passed" vs 实测 93/91。
- 修复方向：重建关键文档或在 SYNC.md 标注丢失并去死引用；校正计数。

### P6 [工程债] 分支混杂 + fork 合并债 + worktree 残留
- `ai/20260228-optimize-cge` 同一分支既有 v2.1 sim 又叠 v2.2.1 Phase A（commit a57b032/02f4eeb/713556c = A4/A3/A2）；"fork 合并回主线"从未发生。
- 残留 worktree：`.claude/worktrees/wf_47972632-* / wf_99662f87-* / wf_ced43a6c-*`（2026-06-12 fan-out 残骸）。
- 残留分支：`sim/v2.1-a3 / a4 / a5 / a6 / a6-d71-5`。
- 修复方向：厘清分支边界、清理 worktree 与残留分支。

### P7 [收尾] v2.1 明确 DEFERRED 的 runtime 项
- A8 DR 物理调优、IsaacLab scene 实例化、A12 ABI 在 mainline conda 实跑（本地 numpy 1.26.4 已绿）。
- 修复方向：P1 修好（GUI 可用）后顺带闭合或明确结论。

## 5. 审计实测命令（可复现）

```bash
# repo 状态
cd ~/Documents/Code/hilserl-fr3 && git branch --show-current   # ai/20260228-optimize-cge
python3 -m pytest sim/ -q                                       # 91 passed, 2 skipped
grep -nE "INSERTION_DEPTH_THRESHOLD|XY_TOLERANCE|ANGLE_TOLERANCE_DEG" sim/data/contract.py  # 全在
grep -n "pixels" sim/data/gello_replay.py                       # 单 pixels 键 (315/469/473/626/630)
grep -n "VALID_PKL_IMAGE_KEYS" sim/scripts/verify_sim_data.py   # 3 键
# desktop
ssh fr3-desktop-ts 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'  # 2876313, 4637 MiB
ssh fr3-desktop-ts 'ps -eo pid,etimes,args | grep plug_insertion_scene'  # ~20h orphan
```

## 6. phase6-ready 仍然 DEFERRED（不在 v2.1.1 范围）

v2.1.1 不触碰 phase6-ready（real pkl + balanced fixture + precision/recall ≥ 0.85 + 用户签字）。
v2.1.1 只把"IsaacLab 真能跑 + 修真 bug + 清工程债"做完，为后续 real-machine phase6 铺路。
