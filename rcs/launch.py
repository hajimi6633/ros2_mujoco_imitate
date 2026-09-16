"""launch：节点栈组装器（等价 ROS2 launch + 参数注入）。

组装原则：
  - 主循环节点 add 顺序 = 执行顺序 = 数据依赖顺序
    （任务 → 控制闸门 → 物理 → 显示）
  - 旁路线程节点（threaded）：渲染（RenderNode，GL 绑定）、视觉
    （VisionNode，纯计算）、安全哨兵（SafetyNode，每帧必检）——
    数据经 topic 的 latest 缓存/订阅回调交接，不占主循环
    （历史：渲染曾为主循环节点，离屏渲染同步回读占主循环 90%+，
    已迁移旁路线程，见 render_node.py 线程模型说明）
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
                         cam_show: bool = False, safety: bool = False):
    """组装充电枪任务全栈。返回 (executor, 句柄 dict)。

    render：创建渲染节点（GL 不可用时渲染自动降级禁用）
    viewer：主窗口（MuJoCo passive viewer，需桌面环境）
    cam_show：相机画面窗口（OpenCV imshow，需 --render）
    vision：视觉定位节点（ArUco + PnP，e2h/eih）
    safety：安全哨兵（颜色分割检测黄色标识物，默认关——检测策略
            调优期间暂不启动，run_stack --safety 显式开启）
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
    safety_node = None
    if render:
        renders = [RenderNode(bus, clock, sim.model, "cam_e2h", "/image_e2h"),
                   RenderNode(bus, clock, sim.model, "cam_eih", "/image_eih")]
        if safety:
            # 安全哨兵：颜色分割检测黄色标识物（详见 safety_node.py）
            safety_node = SafetyNode(bus, clock, sim, "/image_e2h", "cam_e2h")
        if vision:
            # 视觉定位（两路均输出目标物位姿，经标定板→目标物偏差补偿）：
            #   e2h：场景标定板（id=0）贴车插座旁 → /ee_pose_vision 输出
            #        车插座世界位姿（任务状态机视觉伺服消费此话题）
            #   eih：枪尾码板（id=1）x 偏置避开枪体自遮挡 → /target_fine
            #        输出枪位姿（相机系）
            # marker_size = 板宽 0.10 × 0.7407（PNG 图案占比，见 XML 注释）
            renders += [VisionNode(bus, clock, "/image_e2h",
                                   "/ee_pose_vision", "e2h",
                                   sim=sim, camera="cam_e2h",
                                   marker_size=0.074, marker_id=0,
                                   marker_body="scene_marker",
                                   target_body="car_socket",
                                   face_offset=+0.003),
                        VisionNode(bus, clock, "/image_eih",
                                   "/target_fine", "eih",
                                   sim=sim, camera="cam_eih",
                                   marker_size=0.074, marker_id=1,
                                   marker_body="gun_marker",
                                   target_body="charging_gun_1",
                                   face_offset=-0.003)]

    viewer_node = ViewerNode(bus, clock, sim) if viewer else None
    # overlays：vision 开启时相机窗口画 ArUco 检测框（观察视觉链路）
    overlays = ({"/image_e2h": "/ee_pose_vision",
                 "/image_eih": "/target_fine"} if vision else None)
    cam_show_node = (CamShowNode(bus, clock,
                                 [("/image_e2h", "cam_e2h"),
                                  ("/image_eih", "cam_eih")],
                                 overlays=overlays)
                     if (cam_show and render) else None)

    ex = Executor(clock, dt=0.05)             # 50Hz 控制拍
    for n in (task, ctrl, ik, grasp, pose, sim,          # 主循环顺序
              *renders, viewer_node, cam_show_node):
        if n is not None:
            ex.add(n)
    if safety_node is not None:
        ex.add(safety_node)                   # threaded：只启动 worker

    # viewer/cam_show 供上层做退出判定（未启用时为 None）
    handles = {"task": task, "ctrl": ctrl, "sim": sim, "ik": ik,
               "grasp": grasp, "pose": pose, "executor": ex,
               "viewer": viewer_node, "cam_show": cam_show_node}
    return ex, handles
