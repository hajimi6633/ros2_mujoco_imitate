"""SafetyNode 背景差分检测的单元测试（合成图像，无需 GL/渲染）。

核心验证：世界点 → 正投影像素 → 合成前景 blob → 差分检测 →
反投影距离闭环一致（像素级闭环，理论上无近似误差）。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mujoco = pytest.importorskip("mujoco")

from rclike import SimClock, Bus
from rcs.nodes.sim_node import SimNode
from rcs.nodes.safety_node import SafetyNode, Zone


@pytest.fixture()
def stack():
    clock, bus = SimClock(), Bus()
    sim = SimNode(bus, clock, "models/scene_table.xml")
    return bus, clock, sim


def make_node(stack):
    bus, clock, sim = stack
    return SafetyNode(bus, clock, sim, "/image_e2h", "cam_e2h")


def project(node, p_world):
    """世界点 → e2h 像素（用节点自身内外参正向投影，与反投影互逆）。"""
    import mujoco
    fx, fy, cx, cy = node._K(480, 640)
    p_cam = node._cam_R.T @ (np.asarray(p_world, float) - node._cam_pos)
    # MuJoCo 相机看向 -z：u 沿 +x，v 沿 -y
    return (fx * p_cam[0] / -p_cam[2] + cx,
            fy * p_cam[1] / p_cam[2] + cy)


def make_frame(node, world_pts):
    """合成一帧：黑底 + 每个世界点处一个前景矩形（底边中心 = 投影像素）。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for p in world_pts:
        u, v = project(node, p)
        u, v = int(round(u)), int(round(v))
        img[max(0, v - 60):v + 1, max(0, u - 15):u + 15] = 255
    return img


def detect_distance(node, world_pts):
    """首帧锁背景 → 第二帧检测 → 返回反投影最小距离。"""
    node._detect_intruders(np.zeros((480, 640, 3), dtype=np.uint8))
    intruders = node._detect_intruders(make_frame(node, world_pts))
    return node._min_distance(intruders), intruders


def test_first_frame_locks_background(stack):
    node = make_node(stack)
    assert node._detect_intruders(np.zeros((480, 640, 3), np.uint8)) == []


def test_far_intruder_normal_zone(stack):
    node = make_node(stack)
    d, intr = detect_distance(node, [(2.4, -0.5, 0)])   # 距基座 2.47m
    assert len(intr) == 1
    assert d > 0.8                                       # NORMAL 区


def test_mid_intruder_slow_zone(stack):
    node = make_node(stack)
    d, intr = detect_distance(node, [(1.85, -0.5, 0)])  # d ≈ 0.55
    assert len(intr) == 1
    assert 0.4 < d < 0.8                                 # SLOW 区


def test_near_intruder_stop_zone(stack):
    node = make_node(stack)
    d, intr = detect_distance(node, [(1.5, -0.5, 0)])   # d ≈ 0.27
    assert len(intr) == 1
    assert d < 0.4                                       # STOP 区


def test_pixel_roundtrip_consistency(stack):
    """像素级闭环：反投影距离与几何真值一致（误差 < 0.1m）。"""
    node = make_node(stack)
    p = (1.85, -0.5, 0)
    d, _ = detect_distance(node, [p])
    truth = np.hypot(p[0] - node._base[0], p[1] - node._base[1]) \
        - node.get_parameter("workspace_radius")
    assert abs(d - truth) < 0.1


def test_workspace_mask_excludes_arm_region(stack):
    """掩膜几何：视线穿过工作区圆柱的像素应被排除（臂不误报）。"""
    node = make_node(stack)
    mask = node._build_mask(480, 640)
    # 基座正上方像素（视线必然穿过圆柱）应为 False=不允许检测
    u, v = project(node, node._base + np.array([0, 0, 1.0]))
    assert not mask[int(v), int(u)]
    # 远处空地像素应为 True=允许检测
    u, v = project(node, (2.4, -0.5, 0))
    assert mask[int(v), int(u)]
