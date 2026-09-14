"""SafetyNode：安全哨兵（threaded，每帧必检，fail-safe）。

与功能视觉（VisionNode）的三个反差（对话第 6 轮决策）：
  1. 失败语义：图像中断 / 检测超时 → 按侵入处理（绝不回退 GT）
  2. 帧策略：每帧必检（漏检窗口 = 危险窗口），用轻量模型保延迟
  3. 响应路径：快通道 /safety_state.latest 由 ArmController 每拍读；
     任务挂起走慢通道（ChargingAction 订阅本话题）
两级分区：减速区（限速继续）→ 停止区（冻结 · 保持夹持）。

检测实现（特定颜色目标物分割，非画面变化）：
  依据是"黄色标识物（intruder 胶囊）出现在画面中并靠近"，
  而非背景差分——画面任何其他变化（含机械臂运动、物体被移动）
  都不触发安全响应：
  1. RGB→HSV → 亮黄色 inRange 分割（场景唯一色：枪/插座红、臂灰、
     地板蓝灰、pan 绿，无颜色冲突）
  2. 形态学闭运算填高光空洞 + 连通域面积过滤 → 目标
  3. 目标底部像素（脚）反投影射线与地面 z=0 求交 → 世界坐标
  4. d = 距基座水平距离 − 工作半径 → zone 判定
  注意：启动后 SafetyNode 默认不创建（launch safety=False），
  run_stack --safety 显式开启。

fail-safe 恢复语义：检测消失 ≠ 安全（可能被工作区遮挡），
STOP 有最短保持时长 + 恢复需连续无侵入帧（去抖）。

真机迁移期替换 _detect_intruders / _min_distance 为
YOLO-nano 人体检测 + 3D 定位（接口不变，见 TODO 注释）。
"""
from __future__ import annotations
import threading
import time as pytime
from enum import Enum

import numpy as np
import mujoco

from rclike import Node


class Zone(Enum):
    NORMAL = "normal"
    SLOW = "slow"       # 减速区：臂限速（如 25%）继续作业
    STOP = "stop"       # 停止区：冻结运动 · 保持夹持 · 等待确认


class SafetyNode(Node):
    threaded = True

    def __init__(self, bus, clock, sim, image_topic: str,
                 camera: str = "cam_e2h",
                 slow_radius: float = 0.8, stop_radius: float = 0.4,
                 image_timeout: float = 0.2):
        super().__init__("safety_node", bus, clock)
        self.declare_parameter("slow_radius", slow_radius, "减速区阈值 (m)")
        self.declare_parameter("stop_radius", stop_radius, "停止区阈值 (m)")
        self.declare_parameter("image_timeout", image_timeout, "图像超时即失明 (s)")
        self.declare_parameter("workspace_radius", 1.3,
                               "机器人工作区圆柱半径 (m)：含臂展+枪全程轨迹")
        self.declare_parameter("ground_inset", 0.35,
                               "反投影保守补偿 (m)：连通域底边为胶囊朝相机"
                               "前缘像素，射线延长到地面系统性偏远 ~0.2-0.35m，"
                               "直接减去以免触发偏晚（不保守方向）")
        self.declare_parameter("hue_lo", 10, "目标色 H 下界（OpenCV 0-180）")
        self.declare_parameter("hue_hi", 30, "目标色 H 上界（OpenCV 0-180）")
        self.declare_parameter("sat_lo", 80, "目标色饱和度下界 (0-255)")
        self.declare_parameter("val_lo", 80, "目标色明度下界 (0-255)")
        self.declare_parameter("min_area", 300, "前景连通域最小面积 (px)")
        # fail-safe 恢复语义：检测消失 ≠ 安全（可能被工作区遮挡）
        self.declare_parameter("stop_hold_s", 2.0,
                               "STOP 最短保持时长 (s)：防遮挡漏检即刻放行")
        self.declare_parameter("clear_frames", 5,
                               "恢复 NORMAL 需连续无侵入帧数（防边缘抖动）")
        self._img = None
        self._shape = None                      # 最近帧分辨率（内参按此取）
        self._intruders: list = []              # 最新检测的侵入者（像素域）
        self._seen_first = False               # 启动宽限：未收到首帧前不视为失明
        self._last_alive_wall = None             # 最近一帧的墙钟时刻（流中断判定）
        self._stop_evt = threading.Event()
        self._stop_since = 0.0                 # 本次 STOP 进入时刻（墙钟）
        self._clear_n = 0                      # 连续无侵入帧计数（恢复去抖）
        self._pub = self.create_publisher("/safety_state")
        self.zone = Zone.NORMAL
        self.create_subscription(image_topic, self._on_img)

        # ---- 相机内/外参 + 基座（构造期主线程读一次，e2h 固定相机不变）----
        model, data = sim.model, sim.data
        mujoco.mj_forward(model, data)        # 确保 cam_xpos/xmat 有效
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise ValueError(f"相机 '{camera}' 不存在")
        self._cam_pos = data.cam_xpos[cid].copy()
        # cam_xmat：世界←相机旋转（行主序），列 = 相机轴在世界系的分量
        self._cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        self._fovy = model.cam_fovy[cid]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._base = data.xpos[bid].copy() if bid >= 0 \
            else np.zeros(3)

    # ---------- 检测（颜色分割 + 连通域） ----------
    def _on_img(self, img, stamp):
        self._seen_first = True
        self._last_alive_wall = pytime.time()
        self._img = (img, stamp)

    def _K(self, h, w):
        """针孔内参（MuJoCo 相机方形像素，fx=fy 由 fovy 推出）。"""
        f = (h / 2) / np.tan(np.radians(self._fovy) / 2)
        return f, f, w / 2, h / 2

    def _detect_intruders(self, img) -> list:
        """HSV 亮黄色分割 → 目标连通域。

        只响应"特定颜色标识物"（intruder 胶囊）；画面其他变化
        （臂运动、物体移动、光照波动）不在分割范围内，不触发。
        返回 [{"bottom": (u, v), "area": px}, ...]（v 最大行 = 脚部像素）。
        TODO 真机迁移：YOLO-nano 人体检测（返回值结构不变）。
        """
        import cv2                              # 懒加载
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lo = (self.get_parameter("hue_lo"),
              self.get_parameter("sat_lo"),
              self.get_parameter("val_lo"))
        hi = (self.get_parameter("hue_hi"), 255, 255)
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        # 闭运算填目标内部高光空洞；开运算去孤立噪点
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        min_area = self.get_parameter("min_area")
        for i in range(1, n):                 # 0 = 背景
            x, y, w_, h_, area = stats[i]
            if area < min_area:
                continue
            out.append({"bottom": (x + w_ // 2, y + h_ - 1), "area": int(area)})
        return out

    def _min_distance(self, intruders) -> float:
        """侵入者到工作区边界的最小距离（世界系，m）。

        底部像素射线与地面 z=0 求交 → 距基座水平距离 − 工作半径 − 保守补偿
        （连通域底边是目标朝相机的前缘像素，反投影偏远；见 ground_inset）。
        TODO 真机迁移：检测框 + 深度图 / 地面平面拟合。
        """
        if not intruders or self._shape is None:
            return float("inf")
        h, w = self._shape
        fx, fy, cx, cy = self._K(h, w)
        o, R3 = self._cam_pos, self._cam_R
        best = float("inf")
        R_ws = self.get_parameter("workspace_radius")
        inset = self.get_parameter("ground_inset")
        for it in intruders:
            u, v = it["bottom"]
            # MuJoCo 相机看向 -z：射线 z 分量为负
            d_cam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
            d = R3 @ d_cam
            if abs(d[2]) < 1e-9:
                continue                       # 射线近水平，不与地面相交
            t = -o[2] / d[2]
            if t <= 0:
                continue
            p = o + t * d                     # 地面交点
            dist = np.hypot(p[0] - self._base[0], p[1] - self._base[1])
            best = min(best, dist - R_ws - inset)
        return best

    # ---------- worker（心跳 + fail-safe + 检测） ----------
    def start(self):
        threading.Thread(target=self._loop, daemon=True,
                         name=self.name).start()

    def _loop(self):
        while not self._stop_evt.is_set():
            # 心跳：每循环发布当前 zone——ArmController 的 watchdog 依赖
            # safety_state 周期性刷新，仅在 zone 变化时发布会导致误判超时
            self._pub.publish({"zone": self.zone.value, "reason": "heartbeat"},
                              stamp=self.clock.now)
            snap = self._img
            if snap is None:
                pytime.sleep(0.01)              # 等待下一帧（或启动首帧）
                continue
            # fail-safe：收到过首帧后墙钟停更 = 图像流中断 = 哨兵失明 = 停止。
            # 判定用墙钟：渲染周期是墙钟现象，与仿真时钟速率（可快于实时）解耦
            if (self._seen_first and self._last_alive_wall is not None and
                    pytime.time() - self._last_alive_wall >
                    self.get_parameter("image_timeout")):
                self._set_zone(Zone.STOP, "image timeout")
                pytime.sleep(0.01)
                continue
            img, _ = snap
            self._shape = img.shape[:2]
            self._intruders = self._detect_intruders(img)
            d = self._min_distance(self._intruders)
            now = pytime.monotonic()
            in_stop_hold = (self.zone is Zone.STOP and
                            now - self._stop_since <
                            self.get_parameter("stop_hold_s"))
            if d < self.get_parameter("stop_radius"):
                self._clear_n = 0
                if self.zone is not Zone.STOP:
                    self._stop_since = now      # 仅在切入 STOP 时计时
                self._set_zone(Zone.STOP, f"intruder {d:.2f}m")
            elif d < self.get_parameter("slow_radius"):
                self._clear_n = 0
                if not in_stop_hold:            # hold 期内不降级
                    self._set_zone(Zone.SLOW, f"intruder {d:.2f}m")
            else:
                if self.zone is Zone.NORMAL:
                    pass                        # 已安全，无需动作
                elif in_stop_hold:
                    pass                        # hold 期：保持 STOP
                elif self._clear_n < self.get_parameter("clear_frames"):
                    self._clear_n += 1          # 去抖：等连续干净帧
                else:
                    self._clear_n = 0
                    self._set_zone(Zone.NORMAL, "")
            self._img = None
            pytime.sleep(0.01)               # 每帧必检：跟随图像流节奏

    def _set_zone(self, zone: Zone, reason: str):
        if zone is not self.zone:
            self.zone = zone
            (self.log.warn if zone is not Zone.NORMAL else self.log.info)(
                f"zone -> {zone.value} ({reason})")
            self._pub.publish({"zone": zone.value, "reason": reason},
                              stamp=self.clock.now)

    def shutdown(self):
        self._stop_evt.set()
