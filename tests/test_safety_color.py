"""SafetyNode 颜色分割检测的单元测试（合成图像，无需 GL/渲染）。

核心验证：世界点 → 正投影像素 → 合成黄色前景 blob → 颜色分割检测 →
反投影距离闭环一致；同时验证"画面其他变化（非目标色）不触发检测"。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mujoco = pytest.importorskip("mujoco")

from rclike import SimClock, Bus
from rcs.nodes.sim_node import SimNode
from rcs.nodes.safety_node import SafetyNode, Zone

# 与场景 intruder 一致的亮黄（HSV ≈ H24, S227, V230）
YELLOW = (0.9 * 255, 0.75 * 255, 0.1 * 255)
RED = (0.9 * 255, 0.2 * 255, 0.2 * 255)      # 枪/插座色（非目标色对照）


@pytest.fixture()
def node():
    clock, bus = SimClock(), Bus()
    sim = SimNode(bus, clock, "models/scene_table.xml")
    n = SafetyNode(bus, clock, sim, "/image_e2h", "cam_e2h")
    # 合成 blob 为贴地细矩形（无 3D 前缘偏移），关掉保守补偿测纯几何闭环
    n.params.override("ground_inset", 0.0)
    return n


def project(node, p_world):
    """世界点 → e2h 像素（用节点自身内外参正向投影，与反投影互逆）。"""
    fx, fy, cx, cy = node._K(480, 640)
    p_cam = node._cam_R.T @ (np.asarray(p_world, float) - node._cam_pos)
    # MuJoCo 相机看向 -z：u 沿 +x，v 沿 -y
    return (fx * p_cam[0] / -p_cam[2] + cx,
            fy * p_cam[1] / p_cam[2] + cy)


def make_frame(node, world_pts, color=YELLOW, bg=(40, 40, 40)):
    """合成一帧：暗底 + 每个世界点处一个竖直前景矩形（底边中心=投影像素）。"""
    img = np.full((480, 640, 3), bg, dtype=np.uint8)
    for p in world_pts:
        u, v = project(node, p)
        u, v = int(round(u)), int(round(v))
        img[max(0, v - 60):v + 1, max(0, u - 15):u + 15] = color
    return img


def detect_distance(node, world_pts, color=YELLOW):
    node._shape = (480, 640)
    intruders = node._detect_intruders(make_frame(node, world_pts, color))
    return node._min_distance(intruders), intruders


def test_far_intruder_normal_zone(node):
    # 测试点随 cam_e2h 拉近 (2.6,-3.4,2.6)→(1.29,-1.96,1.77) 调整：
    # 原 x 向远点 (2.4,-0.5,0) 投影 u>640 出画（无 blob 可检出）；
    # 改 +y 方向远点，距基座 ~2.33m → d ≈ 1.0 NORMAL 区
    d, intr = detect_distance(node, [(0.3, 2.3, 0)])
    assert len(intr) == 1
    assert d > 0.8                                       # NORMAL 区


def test_mid_intruder_slow_zone(node):
    # +y 方向中距点，距基座 ~1.79m → d ≈ 0.5 SLOW 区
    d, intr = detect_distance(node, [(0.3, 1.75, 0)])
    assert len(intr) == 1
    assert 0.4 < d < 0.8                                 # SLOW 区


def test_near_intruder_stop_zone(node):
    # +y 方向近点，距基座 ~1.07m → 已入 workspace（d<0）→ STOP 区
    d, intr = detect_distance(node, [(0.3, 1.0, 0)])
    assert len(intr) == 1
    assert d < 0.4                                       # STOP 区


def test_pixel_roundtrip_consistency(node):
    """像素级闭环：反投影距离与几何真值一致（误差 < 0.1m）。"""
    p = (0.3, 1.75, 0)                                   # 同 mid 点
    d, _ = detect_distance(node, [p])
    truth = np.hypot(p[0] - node._base[0], p[1] - node._base[1]) \
        - node.get_parameter("workspace_radius")
    assert abs(d - truth) < 0.1


def test_non_target_color_not_detected(node):
    """核心语义：非目标色（红色物体、画面变化）不触发检测。"""
    d, intr = detect_distance(node, [(1.5, -0.5, 0)], color=RED)
    assert intr == []                                    # 红色物体（枪同色）不检出
    assert d == float("inf")


def test_scene_without_intruder_stays_normal(node):
    """空场景（无人/无黄色物）零检出。"""
    node._shape = (480, 640)
    intr = node._detect_intruders(np.full((480, 640, 3), 40, np.uint8))
    assert intr == []
