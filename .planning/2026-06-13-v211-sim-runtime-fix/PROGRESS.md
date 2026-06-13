# v2.1.1 Sim Runtime Fix — 推进日志

> 执行：Claude (Opus 4.8 [1m], ultracode)。fan-out workflow + 顺序 GPU 操作混合。
> 证据目录：`evidence/desktop-snapshot-20260613/`

## 2026-06-13 Phase 1 — Desktop 止血 ✅

- **IH-01 orphan 清理 ✅**：杀掉 PID 2876313 进程树（`plug_insertion_scene.py`，etime ~21.5h，
  nohup+disown 孤儿）。GPU 4700 MiB → 18 MiB，`nvidia-smi --query-compute-apps` 为空。
- **anti-orphan 启动方式 ✅**：放弃 nohup+disown（孤儿来源），改为前台 `timeout 340 isaaclab.sh -p`
  + 脚本内 `os._exit(0)`（headless 下 `simulation_app.close()` 会挂起，实测首跑 RC=124；
  v2 起改为存盘后直接 `os._exit(0)`，RC=0，无孤儿）。
- **SR-01/03 诊断 ✅**：拿到启动日志 + 首帧截图，确认"黑屏"真相（见 Phase 2）。

## 2026-06-13 Phase 2 — 黑屏根因修复（核心已修，场景内容待完善）

### 根因（比 AUDIT 推测更根本）
desktop 实际运行的 `plug_insertion_scene.py` 的 `__main__` **从未实例化场景**：只
`SimulationContext()+reset()+step(render=True)` 跑空 stage，没有任何 prim / 灯光 / 相机取景。
11 个 `@configclass`（FR3/plug/socket/table/camera/light）只是 spec 定义，从未 spawn。
→ 渲染纯黑。AUDIT 列的 Y-up/panda_hand/DistantLight 是 config 内潜在 bug，但都**不相关**
（config 从没被实例化）。叠加 `assets/fr3/fr3.usd`、`assets/table/table.usd` 缺失，
`panda_instanceable.usd` 0 字节。

### 修复
新写自洽 viewer `plug_scene_viewer.py`（direct-spawn 路径，参考同机已验证可渲染的
`droid/scripts/sim/phase3_scene_preview.py` 与 `phase3_simrgb_diagnostic.py`）：
ground + DomeLight + DistantLight + plug USD + socket USD + Pinhole Camera，
headless RGB → PNG（绕开上次"纯黑截图"的 X11 窗口抓取陷阱）。

| 迭代 | 关键改动 | RGB_MEAN | 结果 |
|------|----------|----------|------|
| (旧脚本) | 空 stage | ~0 | 纯黑（用户实测） |
| v1 | 实例化 + dome1000/sun1400 | 225.22 | 非黑但过曝、无地面、单白块 |
| v2 | Y-up→Z-up 旋转 + 本地 cuboid 地面 + 灯光降到 300/600 + 近相机 | 167.18 | 有地面/打光的白块 |
| v3 | 每物体定向（plug 针朝上 −90°X / socket 槽朝上 +90°X）+ 3/4 抬高相机 | 167.82 | 仍是单个无细节白块 |

证据：`evidence/desktop-snapshot-20260613/capture_0{1,2,3}.png`。

### 结论
- **黑屏 bug 已修**：IsaacLab 能渲染插头场景（mean 0 → 167）。
- **遗留（场景内容质量，非黑屏 bug）**：plug/socket 的 USD 是 STL→USD 粗转，渲染为无细节白块
  （插脚/插孔不可见）；只清晰框到一个部件；场景缺 FR3 + 桌（其 USD 缺失/为空）。
  → 这正是 AUDIT **P3（repo `sim/scenes/plug_scene.py` ↔ desktop `plug_insertion_scene.py` 分叉）**
  的实证：所谓 "sim-code-ready" 从未对应一个能跑、可检视的真实场景。
- **待用户拍板（DECISIONS.md 开放问题）**：P3 canonical 源方向；是否要把 viewer 扩成完整
  FR3+桌+插头+插座场景（需补/换 FR3 与桌 USD、修 STL→USD 细节）。

## 待办（本里程碑剩余）
- Phase 2 收尾：定 P3 canonical 源 + 决定渲染完善程度（完整场景 vs 仅证明可渲染）。
- Phase 3 (SD-01)：`pixels` ↔ 3 键 schema 统一（实测：gello_replay/failure_generator 产单 `pixels`，
  `verify_sim_data` 要 3 键，验证器拒绝自家产物）。
- Phase 4 (DOC-01)：sim-build-fork 文档校正（"111 passed" → 实测 91；SYNC.md 死引用）。
- Phase 5 (IH-02/03, SR-04)：分支拆分 + worktree/分支清理 + deferred runtime 收尾。
  **注意：当前有并发 ultracode 会话在改 v2.2.1 Phase A（scripts/tests），P6 删 worktree/分支暂缓，避免破坏并发会话。**

## 环境/并发说明
- 本次 commit 仅用 `.planning/2026-06-13-v211-sim-runtime-fix/` 窄 pathspec；
  working tree 中 `hilserl-fr3/scripts/*` `tests/*` 的改动属另一并发会话的 v2.2.1 Phase A C1/C2 review 修复，本里程碑不触碰。
- 参考脚本（droid，外部项目，未入库）：desktop `/home/robot/droid/scripts/sim/phase3_scene_preview.py`、
  `phase3_simrgb_diagnostic.py`、`droid/sim/assets/{lighting,official_fr3_loader,primitive_scene}.py`。
