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

---

## 2026-06-13（续）— 用户拍板后：全场景 + P3/P4/P5

用户决策：渲染补成完整场景（复用 droid FR3+桌资产）；P3 desktop viewer 为 canonical；P4 producers 产 3 键（sim-to-real）。

### Phase 2 全场景 ✅（SR-01 核心达成）
- 新写 `plug_fullscene_viewer.py`（workflow mine→author→review；复用 droid `build_fr3_cfg`
  → FR3 USD `fr3_gripper_collision.usd` 17MB 实在；`build_table_cfg` 程序化黑桌 top z=0；
  `make_scene_cfg` InteractiveScene 模式 + dome/sun light），叠加插头(0.45,0.12)+插座(0.45,-0.12) 于桌面，
  FR3 home pose pin 住，headless 抓帧。
- 渲染：**FR3 + 桌 + 插头 + 插座一图齐全，有阴影、曝光正常（RGB_MEAN=130）**，RC=0 无孤儿。
  证据 `evidence/desktop-snapshot-20260613/capture_fullscene_01.png`。
- 遗留（资产质量）：插头/插座仍是粗 STL→USD 白块（无插脚/插孔细节）。后续若要逼真需重做 USD 或补细节
  （`sim/scenes/plug_scene.py` 已有 `_build_wrist_1_cam_cfg`，可接腕相机出第二真实视角）。
- **P3 落地**：desktop `plug_scene_viewer.py` / `plug_fullscene_viewer.py` 为 canonical 渲染场景。

### Phase 3 schema 3 键 ✅ (SD-01)
- `gello_replay.py` 加 `_build_image_dict()`，4 处发射点单 `pixels` → 3 键
  (side_policy 渲染图 / side_classifier = side_policy 别名 copy / wrist_1 占位)；`validate_output()` 改 3 键校验。
- `failure_scenario_generator.py` 复用 replay_pure_fk 自动继承 3 键 + `_write_pkl` 内 `_ensure_3key_images` 兜底。
- 新增 `sim/data/tests/test_image_schema_3key.py`（8 tests，驱动真实 producers 过 `verify_sim_data` image_keys_complete）。
- 全套回归：**99 passed, 2 skipped**（原 91 + 8 新；零 regression）。contract.py/verify_sim_data.py 无需改（本就 3 键）。

### Phase 4 文档校正 ✅ (DOC-01)
- sim-build-fork/{SYNC,PROGRESS,VERIFY}.md：所有丢失文档（DESIGN/PLAN/REVIEW-*/PLAN-A1..A11）标注
  `[LOST 2026-06-12 reset]`；"111 passed" → 实测 "91 passed, 2 skipped"（注：111 疑为 fan-out 跨 worktree 聚合）。
- SYNC.md 加 DOC-01 校正说明。

### Phase 5 暂缓（不变）
P6 worktree/分支清理仍因并发 ultracode 会话暂缓。SR-04 deferred runtime 收尾留待。

---

## 2026-06-13（续2）— 插座按真实硬件打磨（ZED 实拍 + 公牛 GN-109K）

用户指明实物为公牛 **GN-109K / GN-B51D** 六口排插, 要求按 ZED 实拍图优化细节 (sim-to-real)。

- **ZED 实拍 grounding**: live 抓帧 ZED 2i external (serial 36276705, 1080p, 只读无 motion)
  → `evidence/.../zed_external_01.png`(+`_strip_crop.png`)。实物 = 白色、圆角、紧凑排插, 侧面出线。
- **GN-109K 规格** (web 查证, 苏宁): 6 位、机身 **204×92×29mm**、白色、总控开关、新国标。
  92mm 宽 ⇒ **2 排 × 3 孔** 布局 (非最初的 270×50 单排, 已纠正)。
- **重建**: viewer 内 `spawn_six_outlet_strip` 改 GN-109K 尺寸 + 2排3列 + 红色总控 + 侧出线 + 三脚墙插。
  渲染 `evidence/.../capture_gn109k_01.png` 与实拍形态/比例吻合。
- **待用户确认**: GN-109K 目录标 "6位五孔"(universal 五孔), 而用户描述为 "3 双口 + 3 三口"。
  当前按用户描述建 (上排三口/下排双口); 若实物为五孔可一键切换。孔位细节在当前机位偏小,
  可出 close-up 复核。
- 工具: `generate_socket_strip_usd.py`(独立 USD 生成器, 备用; 因 pxr 仅 app 内可用改为 viewer 内 primitive 直生),
  `_zed_capture.py`(ZED 抓帧)。

---

## 2026-06-13（续3）— 做出真正的国标五孔 USD 资产

用户决策：**找/做一个真正的国标五孔 USD**。FIND 结论：开放平台无合规 CN 五孔 资产（全是欧/英标），
Chinese 站(爱给网)许可不清 → **MAKE**。

- **新资产（已入库）**：
  - `evidence/.../cn_gn109k_strip.usd`（9KB）— 公牛 GN-109K 机身 204×92×29mm、**6 个真凹陷五孔**
    （白色边框网格凸起 + 灰色凹底 = 真实凹槽，非贴图）、每孔 GB 五孔（三极品字 + 两极）、红色总控、侧出线；
    UsdPreviewSurface 材质 + kinematic + box collision。
  - `evidence/.../cn_gn109k_plug.usd`（2.5KB）— 排插自带尾线**三脚插头**（品字三脚 + 机身），dynamic rigid + convexHull collision（可抓取操作物）。
  - 生成器 `generate_gn109k_usd.py`（headless Kit→pxr 直接 author，含真凹槽 frame-grid 法）。
- **关键修复**：USD `defaultPrim` 必须是顶层 prim（最初建在 `/World/GN109K` 下导致 defaultPrim 无效、
  reference 拉不进几何 → 渲染空白）。改到顶层 `/GN109K`、材质移入 root 子树后正常。
- viewer 改为 spawn 这两个 USD（替代 viewer 内 primitive）。渲染 `capture_usdasset_03.png`：
  六个凹陷五孔孔位 + 红开关 + 三脚插头 + FR3 + 桌，清晰可辨（RGB mean 92.5）。
- 五孔单孔细节在远景偏小，可出 close-up 复核（待定）。

---

## 2026-06-13（续4）— 更逼真 + 接入真训练场景

用户：(1)「欧/英标小改成国标」可行但不划算（插孔必须重做=工作量相当，且下载有 token/外网摩擦）→ 选自建升级；
(2) 要更逼真 + **接入真正的 sim 训练场景**。

- **Stage 1 更逼真**：`generate_gn109k_usd.py` 材质改光泽白塑料 / 暗光泽插孔 / 金属插脚；
  三极 L/N 改**八字**斜插（`add_box rot_deg`）；修 z 叠放 bug（灰底盖住插孔→抬高 slot）。
  close-up 渲染 `capture_gn109k_closeup2.png`：每孔 GB 五孔（三极品字八字 + 两极）清晰可见。
- **Stage 2 接入真训练场景**：USD 入库 `sim/assets/cn_gn109k_{strip,plug}.usd`；
  `sim/scenes/plug_scene.py`（canonical InteractiveScene）的 `_build_socket_cfg`/`_build_plug_cfg`
  从占位 CylinderCfg → **UsdFileCfg 引用 GN-109K strip / 三脚 plug**（repo 相对路径，L1 干净）。
  `pytest sim/` 仍 99 passed/2 skipped；L1 gate 干净（无 /home/robot）。
- 待办：desktop 实跑 `plug_scene.py` 全场景验证（依赖 repo `assets/fr3.usd` 是否在 desktop 就位；
  否则用已验证的 `plug_fullscene_viewer.py`=droid FR3 + GN-109K USD 作 desktop 工作场景）。
