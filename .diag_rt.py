"""诊断 v2：realtime 运行消除渲染滞后，在抓枪关键检查点采帧检测。

检查点：1a0 完成（EE 距枪尾 0.15m 预接近）/ 1a 完成（EE 贴枪尾）/
1e 完成（闭合手指抓枪）——覆盖"码板可见性"的三种几何。
每检查点停 0.6s 让渲染追上（10fps ≥ 6 帧），存帧 + ArUco 检测。
"""
import os
import platform

if (platform.system() == "Linux" and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")):
    os.environ["MUJOCO_GL"] = "egl"

import time
import cv2
import numpy as np
import rcs.launch as launch
from rclike import GoalState

ex, h = launch.build_charging_stack("models/scene_table.xml",
                                    vision=True, render=True)
task = h["task"]
action = task.action
task.send_goal({"task": "diag"})

det = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

# 检查点：进入该阶段前采帧（=上一阶段刚完成、下一步执行前的姿态）
PAUSE_BEFORE = {"1a": "1a0完成_预接近", "1b": "1a完成_贴枪尾",
                "1e": "1d完成_闭爪前", "2a": "1e完成_抓枪后"}
CAPTURED = set()
os.makedirs("diag_frames2", exist_ok=True)

tick = 0
t_real = time.monotonic()
next_t = t_real + 0.05                     # realtime 节拍
while tick < 3000:
    ex.spin_once()                        # 不走 spin() 的 shutdown
    tick += 1
    st = action.handle.state if action.handle else None
    if st in (GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED):
        break
    # 当前阶段名
    ph = task.phases[task._pi] if task.phases else None
    if ph and ph.name in PAUSE_BEFORE and ph.name not in CAPTURED:
        CAPTURED.add(ph.name)
        time.sleep(0.6)                   # 等渲染 worker 追上（最新帧）
        for topic, cam in (("/image_eih", "eih"), ("/image_e2h", "e2h")):
            s = h["pose"].latest(topic)
            if s is None:
                print(f"[{PAUSE_BEFORE[ph.name]}] {cam}: 无帧")
                continue
            img = cv2.cvtColor(s[0], cv2.COLOR_RGB2BGR)
            fn = f"diag_frames2/{ph.name}_{cam}.png"
            cv2.imwrite(fn, img)
            corners, ids, _ = det.detectMarkers(img)
            if ids is None:
                red = cv2.inRange(img, (0, 0, 150), (120, 120, 255))
                print(f"[{PAUSE_BEFORE[ph.name]}] {cam}: stamp={s[1]:.1f} "
                      f"无码板 红枪像素={int(red.sum()/255)}")
            else:
                for c, i in zip(corners, ids.ravel()):
                    e = np.mean([np.linalg.norm(c[0][j] - c[0][(j+1) % 4])
                                 for j in range(4)])
                    print(f"[{PAUSE_BEFORE[ph.name]}] {cam}: stamp={s[1]:.1f} "
                          f"检出 id={i} 边长={e:.0f}px")
    if ph and ph.name == "2a":
        break                              # 2a 起移动段不再需要
    # realtime 节拍（1x）
    delay = next_t - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    next_t += 0.05
    if next_t < time.monotonic() - 0.05:
        next_t = time.monotonic() + 0.05
