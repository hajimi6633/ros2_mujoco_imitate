"""VisionNode ArUco 检测的单元测试（合成 marker 图像，无需场景/GL）。

核心验证：已知 marker 位姿 → 正投影合成图像 → _detect 反解 →
PnP 位姿闭环一致（位置误差 < 5mm）。
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


def synth_frame(tvec, rvec=None):
    """按已知位姿合成含 ArUco 标记的图像。

    rvec 为 marker 朝向（Rodrigues）。默认正立面对相机（绕 x 转 180°：
    marker 系 y 向上 vs 相机系 y 向下），此时印刷内容正向、可解码——
    rvec=0 会令印刷上下颠倒，旋转 180° 后 bit 图案不再是合法 id。
    """
    if rvec is None:
        rvec = np.array([np.pi, 0.0, 0.0])
    fx, fy, cx, cy = K
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    s = MARKER_SIZE
    # 与 _detect 相同的官方角点约定（marker 系 y 向上，TL,TR,BR,BL）
    objp = np.array([[-s/2, s/2, 0], [s/2, s/2, 0],
                     [s/2, -s/2, 0], [-s/2, -s/2, 0]], dtype=float)
    pts, _ = cv2.projectPoints(objp, rvec, tvec, Km, None)
    pts = pts.reshape(-1, 2)
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
                               flags=cv2.INTER_LINEAR), rvec


def test_detect_recovers_known_pose(node):
    tvec = np.array([0.02, -0.01, 0.35])      # 0.35m：66px，避开小目标检测临界
    frame, rvec = synth_frame(tvec)
    result = node._detect(frame)
    assert "target_fine" in result
    pos, rot = result["target_fine"]
    assert np.linalg.norm(pos - tvec) < 0.005  # 位置闭环 < 5mm
    R_true, _ = cv2.Rodrigues(rvec)
    assert np.allclose(rot, R_true, atol=0.02)  # 旋转闭环


def test_detect_rotated_marker(node):
    rvec = np.array([np.pi + 0.1, -0.2, 0.3])   # 正立基础姿态 + 任意扰动
    tvec = np.array([-0.03, 0.02, 0.3])
    frame, rvec = synth_frame(tvec, rvec)
    result = node._detect(frame)
    pos, rot = result["target_fine"]
    assert np.linalg.norm(pos - tvec) < 0.005
    R_true, _ = cv2.Rodrigues(rvec)
    assert np.allclose(rot, R_true, atol=0.05)  # 旋转闭环（PnP 姿态精度低于位置）


def test_detect_empty_image_returns_empty(node):
    assert node._detect(np.zeros((480, 640, 3), np.uint8)) == {}
