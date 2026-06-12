# Phase A Research — 代码现状核查（2026-06-13, Fable 5 plan 会话）

来源：SSH 实读 fr3-desktop-ts:/home/robot/hilserl-fr3 + 上游 hil-serl 仓库 + 文献核查。
结论已驱动 DECISIONS.md 的 #6/#7；此处仅存执行相关事实。

## 代码位置与存活状态

| 资产 | 位置 | 状态 |
|---|---|---|
| GELLO 控制栈 (P2-T1~T6) | fr3-desktop-ts:/home/robot/hilserl-fr3/{scripts,tests} | ✅ 唯一存活副本，从未进 git |
| 本地 repo | ~/Documents/Code/hilserl-fr3 | scripts/tests 中 GELLO 文件已随 2026-06-12 reset 丢失 |
| 上游 hil-serl | fr3-desktop-ts:/home/robot/serl_projects/hil-serl-fr3/upstream/hil-serl | 参考实现 |
| pygame | hilserl-fr3 conda env | 2.6.1 已装，A1 无新依赖 |
| Xbox 手柄 | 未接入（无 /dev/input/js*） | 用户有手柄，USB 待插（Phase B 前提，不阻塞 A） |

## 关键实现事实

1. **GelloIntervention（scripts/gello_intervention.py, 227 行）**
   - delta-jog 模式：`GelloCartesianDeltaAgent.step(joints)` → 归一化 7D action；
     pos_scale=0.1, rpy_scale=0.2, max_step=0.003m, max_total_delta=0.03m
   - 介入检测：leader 平移 >1mm（`_MOVE_THRESHOLD=0.001`）+ 0.5s hold 窗（仿 Spacemouse）
   - `_agent.reset()` 仅在 `_ensure_gello()` 首连时调用 → **累积预算跨 episode 不重置**（A2 要改）
   - 接口：`action(action) -> (action, replaced)`；`info["intervene_action"]`
2. **record_gello_demos_serl.py**：joint 空间相对跟随
   `target = q0 + (raw - raw_gello0) * joint_signs * leader_scale`，leader_scale 默认 0.50；
   安全检查 check_max_step / check_max_total_delta（joint 空间）；A3 复用此机制做 GELLO 段
3. **上游 SpacemouseIntervention 语义**：任意输入非零→接管；无显式使能键 → A1 改为
   RB deadman（显式使能）是对上游语义的收紧，接口不变
4. **上游 usb_pickup_insertion reward**：`int(sigmoid(classifier(obs))>0.7 and obs["state"][0,0]>0.4)`，
   单终点 + state guard；v2.2.1 Phase C/D 照此结构
5. **P2-T3 scaffold**：scripts/setup/16_gello_e2e_motion_test.sh 四模式、approval-gated；
   缺 motion driver（A4 补 `p2_t3_e2e_motion_driver.py`）
6. **测试基线**：desktop gate 113 passed（4 套件）；上次会话本地 245 passed（含 sim 等）

## 对 PLAN 的直接约束

- A0 必须先行：desktop→local rsync 回迁 + git 入库，否则本地无法开发/测试
- 全部新代码 TDD（superpowers 纪律），mock 设备后端，零真机 motion
- A2/A4 为安全关键 → 执行后需 Fable 5 review gate（模型分工决议）
- 双设备共享一个 pygame 实例 → 需要 TeleopDeviceHub 单点持有 joystick 状态
