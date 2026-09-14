"""视觉位姿 vs GT 端到端验证（渲染真实帧 → ArUco+PnP → 对比 GT）。

验证链路：MuJoCo 离屏渲染（EGL）→ VisionNode._detect（ArUco id 过滤 +
PnP + OpenCV→MuJoCo 坐标转换）→ 与仿真派生量（xpos/xmat）对比。

精度基线（640×480，DICT_4X4_50）：
  e2h 场景码板 id=0（图案 0.444m @1.97m，~104px）：
    pos < 35mm（深度向 PnP 噪声 ~1% 量级）、rot < 5°
  eih 枪尾码板 id=1（图案 0.074m @0.5m 斜置 30°，~97px）：
    pos < 10mm、rot < 5°
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


class Harness:
    """模型 + 离屏渲染 + GT 读取（复用 RenderNode 的 MjrContext 管线）。"""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MODEL)
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

    def body_gt(self, name, face_offset):
        """marker 图案中心 GT = body 位姿 + R·(0,0,face_offset)（码面表面）。"""
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        R = self.d.xmat[b].reshape(3, 3).copy()
        return self.d.xpos[b].copy() + R[:, 2] * face_offset, R

    def freejoint(self, body_name):
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.m.jnt_qposadr[self.m.body_jntadr[b]]

    def close(self):
        self.ctx.free()
        self.gl.free()


@pytest.fixture(scope="module")
def h():
    try:
        return Harness()
    except Exception as e:                  # 无 EGL / 无 GL 库
        pytest.skip(f"GL 不可用: {e}")


def make_vision(h, role, camera, size, mid):
    # sim 只需 model/data（_init_camera 用），FakeSim 免起整个 SimNode
    class FakeSim:
        pass
    fs = FakeSim()
    fs.model, fs.data = h.m, h.d
    return VisionNode(Bus(), SimClock(), "/img", "/out", role,
                      sim=fs, camera=camera, marker_size=size, marker_id=mid)


def rot_angle_err(R_det, R_true):
    dR = R_det @ R_true.T
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def test_e2h_scene_marker_pose(h):
    """e2h：场景码板 id=0 世界系位姿（经固定外参），vs GT 误差。"""
    n = make_vision(h, "e2h", "cam_e2h", 0.444, 0)
    r = n._detect(h.render("cam_e2h"))
    assert "ee_pose" in r
    p, R = r["ee_pose"]
    p_gt, R_gt = h.body_gt("scene_marker", 0.01)   # +z 面表面
    assert np.linalg.norm(p - p_gt) < 0.035
    assert rot_angle_err(R, R_gt) < 5.0


def test_eih_gun_marker_pose(h):
    """eih：枪尾码板 id=1 相机系位姿。

    枪斜置 30° 放相机前 0.5m（EE 沿枪轴正对时枪头圆盘会遮挡码板，
    伺服段语义即斜视角观测），码面（局部 -z 面）朝相机。
    """
    cam_pos, cam_R = h.cam("cam_eih")
    # 枪姿态：相机系下 +z=(0,-sin30,cos30)（码面法向朝相机偏上 30°）
    ang = np.radians(30)
    z = np.array([0, -np.sin(ang), np.cos(ang)])
    x = np.array([1, 0, 0])
    y = np.cross(z, x)
    x = np.cross(y, z)
    R_gun = cam_R @ np.column_stack([x, y, z])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_gun.ravel())
    # 枪体（marker 在枪系 0,0,-0.422）使 marker 中心落相机前 0.5m
    p_marker = cam_pos + cam_R @ np.array([0, 0, -0.5])
    gun_pos = p_marker - R_gun @ np.array([0, 0, -0.422])

    qadr = h.freejoint("charging_gun_1")
    mujoco.mj_resetData(h.m, h.d)
    # 关插座 weld（免枪被拉回），写 freejoint 位姿后重算派生量
    h.d.eq_active[mujoco.mj_name2id(
        h.m, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_socgun_1")] = 0
    h.d.qpos[qadr:qadr+3] = gun_pos
    h.d.qpos[qadr+3:qadr+7] = quat
    mujoco.mj_forward(h.m, h.d)

    n = make_vision(h, "eih", "cam_eih", 0.074, 1)
    r = n._detect(h.render("cam_eih"))
    assert "target_fine" in r
    p, R = r["target_fine"]
    # GT：世界系 → cam_eih 相机系（cam_xmat 为 相机→世界 旋转）
    p_w, R_w = h.body_gt("gun_marker", -0.003)     # -z 面表面
    p_gt = cam_R.T @ (p_w - cam_pos)
    R_gt = cam_R.T @ R_w
    assert np.linalg.norm(p - p_gt) < 0.010
    assert rot_angle_err(R, R_gt) < 5.0
