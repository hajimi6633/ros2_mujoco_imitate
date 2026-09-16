"""诊断：整栈抓枪段 eih 画面存帧，定位枪尾码板 0 检出原因。

构造 vision 栈 → 分段 spin → 抓取 /image_eih 关键帧（bus latest）。
"""
import os
import platform

# 先定 GL 后端（无显示 → EGL），再 import 栈
if (platform.system() == "Linux" and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")):
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import rcs.launch as launch
from rclike import GoalState

ex, h = launch.build_charging_stack("models/scene_table.xml",
                                    vision=True, render=True)
action = h["task"].action
h["task"].send_goal({"task": "diag"})
for _ in range(2):                       # spin() 结束会 shutdown 全部 worker，
    ex.spin_once()                       # 循环判定必须用 spin_once（不触发 shutdown）

import cv2
os.makedirs("diag_frames", exist_ok=True)
CHECKPOINTS = {30: "1a0_预接近", 80: "1a_到位", 130: "1c_接近枪尾",
               180: "1e_抓取", 230: "2a_移动", 320: "3_插枪"}
tick = 2
last_phase = ""
while tick < 400:
    ex.spin_once()
    tick += 1
    st = action.handle.state if action.handle else None
    if st in (GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED):
        break
    # 按检查点存帧（eih/e2h 各一张）
    for cp, label in CHECKPOINTS.items():
        if tick == cp:
            for topic, cam in (("/image_eih", "eih"), ("/image_e2h", "e2h")):
                s = ex and h and None or None
            # bus latest 取帧
            for topic, cam in (("/image_eih", "eih"), ("/image_e2h", "e2h")):
                s = h["pose"].latest(topic)
                if s is not None:
                    img = cv2.cvtColor(s[0], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"diag_frames/t{tick}_{cam}_{label}.png", img)
                    print(f"t={tick} {cam} {label}: 已存帧 "
                          f"(stamp={s[1]:.1f})")
    # 阶段日志
    ph = h["task"].phases[h["task"]._pi] if h["task"].phases else None
    if ph and ph.name != last_phase:
        last_phase = ph.name
        print(f"t={tick} 进入阶段 {ph.name}")

# 保存视觉话题的检出情况
for topic, cam in (("/ee_pose_vision", "e2h"), ("/target_fine", "eih")):
    s = h["pose"].latest(topic)
    print(f"{cam} 最新视觉输出: {'有' if s else '无'}")
print("帧已存 diag_frames/")
