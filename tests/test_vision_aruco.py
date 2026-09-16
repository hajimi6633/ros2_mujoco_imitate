"""VisionNode ArUco 检测的单元测试（合成 marker 图像，无需场景/GL）。

核心验证：已知 marker 位姿 → 正投影合成图像 → _detect 反解 →
PnP 位姿闭环一致。

精度注记：PnP 用 4 个共面角点解位姿，倾斜分量（绕 x/y 的姿态）对
亚像素角点误差极敏感——5cm marker @0.35m（图上 ~65px）时 5° 级别的
倾斜噪声属固有量级。故旋转断言用"相对旋转角度误差"度量而非逐元素
allclose（逐元素 atol 无法区分物理量级）。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rclike import SimClock, Bus
from rcs.nodes.vision_node import VisionNode

MARKER_SIZE = 0.05          # 5cm 标记
K = (460.0, 460.0, 320.0, 240.0)   # 与 eih fovy 推出值同量级的合成内参


@pytest.fixture()
def node():
    bus, clock = Bus(), SimClock()
    n = VisionNode(bus, clock, "/img", "/out", "eih")
    n._K = K                               # 注入合成内参（免场景依赖）
    return n


def synth_frame(tvec, R_mj):
    """按 MuJoCo 相机系约定（x 右 / y 上 / z 后）合成含 ArUco 标记的图像。

    tvec/R_mj 为 marker→MuJoCo 相机系位姿（= _detect 的输出约定）。
    正立面朝相机时 R_mj=I（marker y 上与相机 y 上同向）。
    """
    fx, fy, cx, cy = K
    s = MARKER_SIZE
    # 与 _detect 相同的官方角点约定（marker 系 y 向上，TL,TR,BR,BL）
    objp = np.array([[-s/2, s/2, 0], [s/2, s/2, 0],
                     [s/2, -s/2, 0], [-s/2, -s/2, 0]], dtype=float)
    pts = []
    for p in objp:
        q = R_mj @ p + np.asarray(tvec, float)
        # MuJoCo 针孔投影：z<0（前方），v 向下 = -y
        pts.append((fx * q[0] / -q[2] + cx, fy * q[1] / q[2] + cy))
    pts = np.array(pts)
    # marker 原图四角 → 目标角点的单应变换（含亚像素，精度优于逐点绘制）
    # 周围 pad 白色安静区（ArUco 检测要求 marker 边框外有白边）
    pad = 30
    marker = np.pad(cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 200),
        pad, constant_values=255)
    src = np.array([[pad, pad], [pad + 200, pad],
                    [pad + 200, pad + 200], [pad, pad + 200]],
                   dtype=np.float32)
    # marker 印刷图行方向 = 检测角点顺序（TL,TR,BR,BL），src 直接对应
    H = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    # 灰底（非纯黑）：纯黑大背景会让自适应阈值失效，真机图像不会全黑
    img = np.full((480, 640), 128, dtype=np.uint8)
    return cv2.warpPerspective(marker, H, (640, 480),
                               dst=img, borderMode=cv2.BORDER_TRANSPARENT,
                               flags=cv2.INTER_LINEAR)


def rot_angle_err(R_det, R_true):
    """相对旋转 dR = R_det·R_trueᵀ 的转角（度）——姿态误差的物理量纲。"""
    dR = R_det @ R_true.T
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def test_detect_recovers_known_pose(node):
    tvec = np.array([0.02, -0.01, -0.35])     # MuJoCo 系：前方 0.35m（z 负）
    frame = synth_frame(tvec, np.eye(3))
    result = node._detect(frame)
    assert "target_pose" in result
    pos, rot = result["target_pose"]
    assert np.linalg.norm(pos - tvec) < 0.008  # 位置闭环 < 8mm（warp 亚像素极限）
    assert rot_angle_err(rot, np.eye(3)) < 6.0  # 姿态 < 6°（PnP 倾斜固有噪声）
    assert "overlay" in result                 # 角点框数据（CamShow 画框用）
    assert np.array(result["overlay"]["corners"]).shape == (1, 4, 2)


def test_detect_rotated_marker(node):
    rvec = np.array([0.1, -0.2, 0.3])          # 任意小扰动（MuJoCo 系 Rodrigues）
    R_true, _ = cv2.Rodrigues(rvec)
    tvec = np.array([-0.03, 0.02, -0.3])
    frame = synth_frame(tvec, R_true)
    result = node._detect(frame)
    pos, rot = result["target_pose"]
    assert np.linalg.norm(pos - tvec) < 0.008
    assert rot_angle_err(rot, R_true) < 6.0


def test_detect_empty_image_returns_empty(node):
    assert node._detect(np.zeros((480, 640, 3), np.uint8)) == {}
