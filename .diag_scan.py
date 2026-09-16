"""沿 1a0 轨迹扫描码板可见性：每 10 步输出码板在 eih 相机系的几何。

入画判定：|u-320|<320 且 |v-240|<240 且深度>0；
可解码判定：码板 -z 面法向·视线 > cos(60°)=0.5。
"""
import os
import platform

if (platform.system() == "Linux" and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")):
    os.environ["MUJOCO_GL"] = "egl"

import time
import numpy as np
import mujoco
import rcs.launch as launch
from rclike import GoalState

ex, h = launch.build_charging_stack("models/scene_table.xml",
                                    vision=True, render=True)
task, sim = h["task"], h["sim"]
action = task.action
task.send_goal({"task": "diag"})

m, d = sim.model, sim.data
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_eih")
bmk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gun_marker")
f = (480 / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)

print("1a0 轨迹扫描（每10步）:")
tick = 0
while tick < 2000:
    ex.spin_once()
    tick += 1
    st = action.handle.state if action.handle else None
    if st in (GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED):
        break
    ph = task.phases[task._pi] if task.phases else None
    if ph and ph.name == "1a0_pre":
        if ph.step_i % 10 == 0 and ph.step_i > 0:
            p_cam = d.cam_xpos[cid].copy()
            R_cam = d.cam_xmat[cid].reshape(3, 3).copy()
            c = R_cam.T @ (d.xpos[bmk] - p_cam)
            depth = -c[2]
            if depth > 0.02:
                u = f * c[0] / depth + 320
                v = -f * c[1] / depth + 240
                n = -d.xmat[bmk].reshape(3, 3)[:, 2]
                view = (d.xpos[bmk] - p_cam)
                view /= np.linalg.norm(view)
                cosang = float(n @ view)
                inpic = (0 <= u < 640 and 0 <= v < 480)
                print(f"  step={ph.step_i:3d} 码板相机系=({c[0]:+.3f},{c[1]:+.3f},"
                      f"{c[2]:+.3f}) 深={depth:.2f}m "
                      f"像素=({u:.0f},{v:.0f}) 入画={'Y' if inpic else 'N'} "
                      f"正对cos={cosang:+.2f}"
                      f"{' ★可检' if inpic and cosang > 0.5 else ''}")
    if ph and ph.name == "1a":
        break
