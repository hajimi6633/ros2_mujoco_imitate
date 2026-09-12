"""launch：节点栈组装器（等价 ROS2 launch + 参数注入）。

组装原则：
  - 主循环节点 add 顺序 = 执行顺序 = 数据依赖顺序
    （任务 → 控制闸门 → 物理 → 渲染/显示）
  - SafetyNode 为旁路线程（未来接检测算法）；渲染/显示为
    主线程节点（Windows 线程 GL 限制，见 render_node.py）
显示三件套独立开关：render（渲染数据流）/ viewer（主窗口）/ cams（相机窗口）
"""
from __future__ import annotations

from rclike import SimClock, Bus, Executor

from rcs.nodes.sim_node import SimNode
from rcs.nodes.arm_controller import ArmController
from rcs.nodes.ik_service import IKService
from rcs.nodes.grasp_node import GraspNode
from rcs.nodes.pose_node import PoseNode
from rcs.nodes.render_node import RenderNode
from rcs.nodes.vision_node import VisionNode
from rcs.nodes.safety_node import SafetyNode
from rcs.nodes.viewer_node import ViewerNode
from rcs.nodes.cam_show_node import CamShowNode
from rcs.tasks.charging_action import ChargingAction


def build_charging_stack(scene_xml: str, vision: bool = False,
                         render: bool = True, viewer: bool = False,
                         cam_show: bool = False):
    """组装充电枪任务全栈。返回 (executor, 句柄 dict)。

    render：创建渲染 + 安全节点（GL 不可用时渲染自动降级禁用）
    viewer：主窗口（MuJoCo passive viewer，需桌面环境）
    cam_show：相机画面窗口（OpenCV imshow，需 --render）
    """
    clock, bus = SimClock(), Bus()

    # ---- 主循环：任务 → 控制 → 物理 ----
    sim = SimNode(bus, clock, scene_xml)
    ctrl = ArmController(bus, clock)
    ik = IKService(bus, clock, sim)
    grasp = GraspNode(bus, clock, sim)
    pose = PoseNode(bus, clock, sim)
    task = ChargingAction(bus, clock, sim, ik, grasp, pose)

    # ---- 渲染与显示（主线程；render 开启才创建）----
    renders = []
    safety = None
    if render:
        renders = [RenderNode(bus, clock, sim.model, "cam_e2h", "/image_e2h"),
                  RenderNode(bus, clock, sim.model, "cam_eih", "/image_eih")]
        safety = SafetyNode(bus, clock, "/image_e2h")     # 旁路线程
        if vision:
            renders += [VisionNode(bus, clock, "/image_e2h",
                                   "/ee_pose_vision", "e2h"),
                        VisionNode(bus, clock, "/image_eih",
                                   "/target_fine", "eih")]

    viewer_node = ViewerNode(bus, clock, sim) if viewer else None
    cam_show_node = (CamShowNode(bus, clock,
                                 [("/image_e2h", "cam_e2h"),
                                  ("/image_eih", "cam_eih")])
                     if (cam_show and render) else None)

    ex = Executor(clock, dt=0.05)             # 50Hz 控制拍
    for n in (task, ctrl, ik, grasp, pose, sim,          # 主循环顺序
              *renders, viewer_node, cam_show_node):
        if n is not None:
            ex.add(n)
    if safety is not None:
        ex.add(safety)                         # threaded：只启动 worker

    # viewer/cam_show 供上层做退出判定（未启用时为 None）
    handles = {"task": task, "ctrl": ctrl, "sim": sim, "ik": ik,
               "grasp": grasp, "pose": pose, "executor": ex,
               "viewer": viewer_node, "cam_show": cam_show_node}
    return ex, handles
