"""复现 test_eih 几何，保存渲染帧 + 码板相机系几何分析。"""
import numpy as np
import mujoco
import cv2
import os.path

MODEL = "models/scene_table.xml"
m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

from mujoco.rendering.classic import gl_context
gl = gl_context.GLContext(640, 480)
gl.make_current()
ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(m, 10000)
vcam = mujoco.MjvCamera()
vcam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_eih")
vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, vcam,
                       mujoco.mjtCatBit.mjCAT_ALL.value, scene)
rect = mujoco.MjrRect(0, 0, 640, 480)
mujoco.mjr_render(rect, scene, ctx)
img = np.empty((480, 640, 3), np.uint8)
mujoco.mjr_readPixels(img, None, rect, ctx)
img = img[::-1].copy()

cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_eih")
cam_pos = d.cam_xpos[cid].copy()
cam_R = d.cam_xmat[cid].reshape(3, 3).copy()
print(f"cam_eih fovy={m.cam_fovy[cid]}")

# 摆枪（同测试）
ang = np.radians(15)
z = np.array([0, np.sin(ang), -np.cos(ang)])
x = np.array([1, 0, 0])
y = np.cross(z, x)
x = np.cross(y, z)
R_gun = cam_R @ np.column_stack([x, y, z])
quat = np.empty(4)
mujoco.mju_mat2Quat(quat, R_gun.ravel())
gun_pos = cam_pos + cam_R @ np.array([0, 0, -0.7])

qadr = m.jnt_qposadr[m.body_jntadr[
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "charging_gun_1")]]
d.eq_active[mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_socgun_1")] = 0
d.qpos[qadr:qadr+3] = gun_pos
d.qpos[qadr+3:qadr+7] = quat
mujoco.mj_forward(m, d)

# 码板几何
bmk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gun_marker")
c_mk = cam_R.T @ (d.xpos[bmk] - cam_pos)
R_mk = d.xmat[bmk].reshape(3, 3)
n_plus = R_mk[:, 2]                     # 码板 +z 法向（世界）
view = d.xpos[bmk] - cam_pos
view /= np.linalg.norm(view)
f = (480/2)/np.tan(np.radians(m.cam_fovy[cid])/2)
u = f*c_mk[0]/(-c_mk[2]) + 320
v = -f*c_mk[1]/(-c_mk[2]) + 240
print(f"码板相机系={c_mk.round(3)} 深={-c_mk[2]:.3f} 像素=({u:.0f},{v:.0f})")
print(f"+z法向·视线={float(n_plus @ view):+.3f}（>0 可见+正常图案）")

# 渲染 + 检测
mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, vcam,
                       mujoco.mjtCatBit.mjCAT_ALL.value, scene)
mujoco.mjr_render(rect, scene, ctx)
mujoco.mjr_readPixels(img, None, rect, ctx)
img = img[::-1].copy()
cv2.imwrite("test_eih_frame.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
dp = cv2.aruco.DetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
det = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), dp)
corners, ids, rejected = det.detectMarkers(img)
print("ids =", None if ids is None else ids.ravel(),
      "rejected =", len(rejected) if rejected is not None else 0)
if rejected:
    for r in rejected[:3]:
        c = r[0]
        print(f"  rej@({c[:,0].mean():.0f},{c[:,1].mean():.0f}) "
              f"边{np.mean([np.linalg.norm(c[j]-c[(j+1)%4]) for j in range(4)]):.0f}px")
ctx.free(); gl.free()
