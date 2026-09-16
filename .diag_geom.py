"""数值验证：1a 完成（EE 对齐 gun_site_2）时 eih 相机与码板的几何。

计算：相机位姿/视线、码板中心在画面上的像素位置、码板是否入画、
手指（±y 0.02~0.04，z -0.027~-0.107，x±0.02）是否挡视线。
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

tick = 0
while tick < 2000:
    ex.spin_once()
    tick += 1
    st = action.handle.state if action.handle else None
    if st in (GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED):
        break
    ph = task.phases[task._pi] if task.phases else None
    if ph and ph.name == "1b":                # 1a 刚完成
        break

m, d = sim.model, sim.data
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_eih")
bid_mk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gun_marker")
bid_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "charging_gun_1")
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gun_site_2")

p_cam = d.cam_xpos[cid].copy()
R_cam = d.cam_xmat[cid].reshape(3, 3).copy()   # 列 = 相机 x,y,z（世界系）
p_mk = d.xpos[bid_mk].copy()
p_g = d.xpos[bid_g].copy()
p_s2 = d.site_xpos[sid].copy()

# 相机系坐标（MuJoCo 相机：视线沿 -z）
def to_cam(p):
    return R_cam.T @ (p - p_cam)

c_mk = to_cam(p_mk)
c_s2 = to_cam(p_s2)
print(f"1a 完成时（仿真 {d.time:.2f}s）:")
print(f"  eigh 相机 pos={p_cam.round(3)}")
print(f"  码板   相机系={c_mk.round(3)}（z<0 即视线前方）")
print(f"  枪尾site相机系={c_s2.round(3)}")
print(f"  枪 body 世界={p_g.round(3)}")
# 画面像素（u 沿相机 +x，v 沿相机 -y[图像向下]，MuJoCo 相机 y 上）
f = (480 / 2) / np.tan(np.radians(40) / 2)
for name, c in (("码板", c_mk), ("枪尾site", c_s2)):
    u = f * c[0] / (-c[2]) + 320
    v = -f * c[1] / (-c[2]) + 240
    print(f"  {name}: 画面像素 (u={u:.0f}, v={v:.0f}), 深度={-c[2]:.3f}m")
# 手指是否挡码板视线：手指在相机系的位置范围
for fb in ("finger_left", "finger_right"):
    bf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fb)
    cf = to_cam(d.xpos[bf])
    print(f"  {fb} 相机系={cf.round(3)}")
# 视线与码板法向夹角（码板 -z 面朝相机才能解码）
R_mk = d.xmat[bid_mk].reshape(3, 3).copy()
n_mk = -R_mk[:, 2]                            # 码板 -z 面（朝 EE）
view = (p_mk - p_cam) / np.linalg.norm(p_mk - p_cam)
cosang = float(n_mk @ view)
print(f"  码板 -z 面法向·视线 = {cosang:.3f}"
      f"（>0 可见面，1=正对）")
