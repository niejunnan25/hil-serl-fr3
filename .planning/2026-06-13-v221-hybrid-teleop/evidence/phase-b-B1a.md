# Phase B — planning + B1a software evidence (2026-06-13)

## 规划

- Phase B 契约 grounding：fan-out 三源读真实代码，产出 `B-RESEARCH.md`（权威 franka_server 契约）。
- 4 份 PLAN：`plans/PLAN-B1..B4.md`。B1/B2 拆 software(可先做) / live(真机) 两段；B3/B4 全真机。

## B-RESEARCH 关键结论（解决 C2）

- franka_server **唯一运动命令 = 绝对 `POST /pose {"arr":[x,y,z,qx,qy,qz,qw]}`**（米 + scalar-last 单位四元数，base frame "0"）；**无关节命令端点**。
- 状态读 `POST /getstate` → `{q[7], pose[7], ...}`。
- 已验证 HTTP 路径 = `record_gello_demos_serl.py`：关节目标 `q0+raw_delta*signs*scale` → `forward_kinematics` → POST `/pose`。
- C2 原 bug = 把 `GelloCartesianDeltaAgent` 的归一化 [-1,1] delta 当绝对位姿 POST。

## B1a 实现（软件，零真机）

- 新 `scripts/gello_pose_follow.py`：纯 numpy follow（joint_target + GelloPoseFollower，abort-on-unsafe，FK→绝对位姿），faithful to record_gello_demos_serl。关节弧度门 max_step=0.003/max_total=0.03。
- 重写 driver `_send_full`（解除 A6 fail-closed）：`/getstate` 读 q0+currpos → **FK-bias 门**(‖FK(q0)-currpos‖>5mm → rc 11 拒绝，零 POST) → 每 tick `/clearerr` + `/pose {"arr"}`。approval 门保留(rc5)。
- 16_*.sh motion() 传 `--mode full` + rc 11→exit 33；banner/header 更新（不再 fail-closed）。
- 单位 hygiene：follower 用自身弧度默认，**不**接 driver 的米制 MAX_STEP（对抗校验 foot-gun 修复）。

## 验证

```
tests/test_gello_pose_follow.py        6 passed   (纯 follow 数学)
tests/test_p2t3_motion_driver.py      14 passed   (含 mock HTTP server: payload 'arr'+单位四元数+首命令无跳变+bias门 rc11 零POST+approval rc5)
full suite                           232 passed   (A6 后 225 → +7)
```

- **对抗式复核 (Opus, 独立 agent)**：8 条安全不变量全部 CONFIRMED-SAFE，**APPROVE**。3 条非阻断质量缺陷：(1) MAX_STEP/MAX_TOTAL_DELTA 注释单位误导→已修(单位分离)；(2) full 模式 agent 参数+seed 读为死代码(无害,留);(3) /clearerr 失败误报为 /pose reject(cosmetic,留)。
- verify 子代理在 ECONNRESET 后由 Opus 主会话兜底（按 model-stage-policy）。

## 边界 / 留真机 (B1b)

- B1a 全程零真机：full 模式仍受 `FR3_GELLO_E2E_APPROVAL` 门，首次 live 必须 dry-run→no-op(命令当前位姿)→micro→full ramp + 用户现场 + E-stop。
- 真机现场核对清单（B-RESEARCH live-verify + PLAN-B1b）：/healthz 可能 404(用 /getstate 探活)、payload key 'arr'、FK 偏置(可能触发 rc11→需标定 FK/flange)、四元数约定、是否需先 /startimp、fork config 是否覆盖 ACTION_SCALE/safety box/gripper。
