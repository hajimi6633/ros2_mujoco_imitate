"""验证 MuJoCo box ±z 面的 ArUco 纹理方向：正对渲染两面对比检测。

相机看 +z 面（板上表面）与 -z 面（板下表面）各渲染一帧，
跑同一检测器——若 -z 面 rejected/+z 面检出，即 -z 面纹理镜像。
"""
import os
import platform

if (platform.system() == "Linux" and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")):
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import os.path
from mujoco.rendering.classic import gl_context
import cv2

# 板在相机前 0.3m；quat 绕 x 转 180° 的板让 -z 面朝相机
XML = """
<mujoco>
  <asset>
    <texture name="aruco1" type="2d" file="ARUCO1"/>
    <material name="m" texture="aruco1"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <!-- FACEUP=1: +z 面朝相机（板绕 x 转 180°，-z 朝上）；
         FACEUP=0: -z 面朝相机（quat 恒等，板默认朝向） -->
    <body name="mk" pos="0 0 -0.3" quat="QUAT">
      <geom name="g" type="box" size="0.05 0.05 0.003"
            material="m" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def render_and_detect(quat, label):
    xml = (XML.replace("QUAT", quat)
              .replace("ARUCO1", os.path.abspath("models/meshes/aruco1.png")))
    with open("/tmp/face_test.xml", "w") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_path("/tmp/face_test.xml")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gl = gl_context.GLContext(640, 480)
    gl.make_current()
    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    scene = mujoco.MjvScene(m, 10000)
    vcam = mujoco.MjvCamera()
    vcam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, vcam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    rect = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjr_render(rect, scene, ctx)
    img = np.empty((480, 640, 3), np.uint8)
    mujoco.mjr_readPixels(img, None, rect, ctx)
    img = img[::-1].copy()
    ctx.free(); gl.free()

    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), dp)
    corners, ids, rejected = det.detectMarkers(img)
    print(f"{label}: ids={None if ids is None else ids.ravel()} "
          f"rejected={len(rejected) if rejected is not None else 0}")


# quat 恒等（1 0 0 0）：板的 -z 面朝相机（MuJoCo 相机视线 -z）
render_and_detect("1 0 0 0", "-z 面朝相机（整栈现状，EE 从枪尾看）")
# 绕 x 转 180°（quat 0 1 0 0）：板翻面，+z 面朝相机
render_and_detect("0 1 0 0", "+z 面朝相机（e2e 测试的等效面）")
# -z 面朝相机 + 绕 y 斜 25°（复现整栈 1a 完成时的 23° 斜视）
qy = np.empty(4)
half = np.radians(25) / 2
qy[:] = [np.cos(half), 0, np.sin(half), 0]
render_and_detect(" ".join(f"{v:.6f}" for v in qy), "-z 面 + 绕y斜25°")
qz = np.empty(4)
half = np.radians(25) / 2
qz[:] = [np.cos(half), np.sin(half), 0, 0]
render_and_detect(" ".join(f"{v:.6f}" for v in qz), "-z 面 + 绕x斜25°")
