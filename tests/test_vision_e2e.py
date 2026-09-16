"""视觉位姿 vs GT 端到端验证（渲染真实帧 → ArUco+PnP+偏差补偿 → 对比 GT）。

验证链路：MuJoCo 离屏渲染（EGL）→ VisionNode._detect（ArUco id 过滤 +
PnP + 坐标转换 + 标定板→目标物偏差补偿）→ 与仿真派生量（xpos/xmat）对比。

两个用例：
  eih（真实场景）：枪斜置 30° 放相机前——输出应为补偿后的枪 body 位姿
    （枪尾码板 x 偏置 -0.13，PnP 码面位姿 ∘ 板→枪偏差）
  e2h（合成近距场景）：真实场景标定板在车插座旁距 e2h 3.71m，0.10m 板
    图案仅 ~9px 不可检测（见 launch 注释）——用独立近距场景验证
    e2h 算法 + 偏差补偿（输出 = target body 世界位姿）

精度阈值（640×480，DICT_4X4_50）：pos < 10mm、rot < 5°。
无 GL 环境自动 skip。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mujoco = pytest.importorskip("mujoco")
from mujoco.rendering.classic import gl_context  # noqa: E402

from rclike import SimClock, Bus  # noqa: E402
from rcs.nodes.vision_node import VisionNode  # noqa: E402

MODEL = "models/scene_table.xml"
# 合成近距场景（e2h 用）：相机俯视，标定板贴目标物旁
SYNTH_XML = """
<mujoco>
  <asset>
    <texture name="aruco0" type="2d" file="ARUCO0"/>
    <material name="aruco0_mat" texture="aruco0"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 1.0" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <body name="target" pos="0.15 0 0" quat="0.9659 0 0.2588 0">
      <geom name="tgt" type="box" size="0.06 0.06 0.06" rgba="0.8 0.2 0.2 1"/>
      <body name="marker" pos="0 0.15 0">
        <geom name="mk" type="box" size="0.05 0.05 0.003"
              material="aruco0_mat" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class Harness:
    """模型 + 离屏渲染（复用 RenderNode 的 MjrContext 管线）。"""

    def __init__(self, xml_path):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.gl = gl_context.GLContext(640, 480)
        self.gl.make_current()
        self.ctx = mujoco.MjrContext(
            self.m, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.m, 10000)
        self.vcam = mujoco.MjvCamera()
        self.rect = mujoco.MjrRect(0, 0, 640, 480)

    def render(self, camera_name):
        self.vcam.fixedcamid = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mujoco.mjv_updateScene(self.m, self.d, mujoco.MjvOption(), None,
                               self.vcam, mujoco.mjtCatBit.mjCAT_ALL.value,
                               self.scene)
        mujoco.mjr_render(self.rect, self.scene, self.ctx)
        img = np.empty((480, 640, 3), np.uint8)
        mujoco.mjr_readPixels(img, None, self.rect, self.ctx)
        return img[::-1].copy()             # GL 原点左下 → 垂直翻转

    def cam(self, name):
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return (self.d.cam_xpos[cid].copy(),
                self.d.cam_xmat[cid].reshape(3, 3).copy())

    def freejoint(self, body_name):
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.m.jnt_qposadr[self.m.body_jntadr[b]]

    def close(self):
        self.ctx.free()
        self.gl.free()


@pytest.fixture(scope="module")
def h_real():
    try:
        return Harness(MODEL)
    except Exception as e:                  # 无 EGL / 无 GL 库
        pytest.skip(f"GL 不可用: {e}")


@pytest.fixture(scope="module")
def h_synth():
    try:
        import os.path
        xml = SYNTH_XML.replace(
            "ARUCO0", os.path.abspath("models/meshes/aruco0.png"))
        with open("/tmp/vision_synth.xml", "w") as f:
            f.write(xml)
        return Harness("/tmp/vision_synth.xml")
    except Exception as e:
        pytest.skip(f"GL 不可用: {e}")


class FakeSim:
    """VisionNode._init_* 只需 model/data 属性。"""

    def __init__(self, m, d):
        self.model, self.data = m, d


def make_vision(h, role, camera, size, mid, mbody, tbody, face):
    return VisionNode(Bus(), SimClock(), "/img", "/out", role,
                      sim=FakeSim(h.m, h.d), camera=camera,
                      marker_size=size, marker_id=mid,
                      marker_body=mbody, target_body=tbody,
                      face_offset=face)


def rot_angle_err(R_det, R_true):
    dR = R_det @ R_true.T
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def test_eih_outputs_gun_pose(h_real):
    """eih：枪尾码板（x 偏置）→ 输出补偿后的枪 body 位姿（相机系）。

    枪斜置 15° 放相机前 ~0.7m（斜视伺服语义；正对时枪头圆盘部分
    遮挡码板，码板 x 偏置后仍露出可解码）。
    """
    cam_pos, cam_R = h_real.cam("cam_eih")
    # 枪姿态：相机系下 +z=(0,-sin30,cos30)——码板 quat 恒等（+z 面朝枪
    # 尾外/EE），枪 +z 偏下朝相机时相机看到码板 +z 面（正常图案；
    # -z 面镜像不可解码，正对渲染实验实测）
    ang = np.radians(30)
    z = np.array([0, -np.sin(ang), np.cos(ang)])
    x = np.array([1, 0, 0])
    y = np.cross(z, x)
    x = np.cross(y, z)
    R_gun = cam_R @ np.column_stack([x, y, z])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_gun.ravel())
    # 枪原点放相机前 0.7m（码板 ~0.34m 前，143px 可检测）
    gun_pos = cam_pos + cam_R @ np.array([0, 0, -0.7])

    qadr = h_real.freejoint("charging_gun_1")
    mujoco.mj_resetData(h_real.m, h_real.d)
    h_real.d.eq_active[mujoco.mj_name2id(
        h_real.m, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_socgun_1")] = 0
    h_real.d.qpos[qadr:qadr+3] = gun_pos
    h_real.d.qpos[qadr+3:qadr+7] = quat
    mujoco.mj_forward(h_real.m, h_real.d)

    n = make_vision(h_real, "eih", "cam_eih", 0.074, 1,
                    "gun_marker", "charging_gun_1", +0.003)
    r = n._detect(h_real.render("cam_eih"))
    assert "target_pose" in r, "枪尾码板未检出（斜视几何或遮挡变化）"
    p, R = r["target_pose"]
    # GT：枪 body 位姿 → cam_eih 相机系
    bid = mujoco.mj_name2id(h_real.m, mujoco.mjtObj.mjOBJ_BODY, "charging_gun_1")
    p_gt = cam_R.T @ (h_real.d.xpos[bid] - cam_pos)
    R_gt = cam_R.T @ h_real.d.xmat[bid].reshape(3, 3)
    assert np.linalg.norm(p - p_gt) < 0.010
    assert rot_angle_err(R, R_gt) < 5.0


def test_e2h_outputs_target_pose(h_synth):
    """e2h：标定板贴目标物旁 → 输出目标物 body 世界位姿（偏差补偿后）。"""
    n = make_vision(h_synth, "e2h", "cam", 0.074, 0,
                    "marker", "target", +0.003)
    r = n._detect(h_synth.render("cam"))
    assert "target_pose" in r, "合成场景码板未检出（渲染/检测回归）"
    p, R = r["target_pose"]
    bid = mujoco.mj_name2id(h_synth.m, mujoco.mjtObj.mjOBJ_BODY, "target")
    assert np.linalg.norm(p - h_synth.d.xpos[bid]) < 0.010
    assert rot_angle_err(R, h_synth.d.xmat[bid].reshape(3, 3)) < 5.0
