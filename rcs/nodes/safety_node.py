"""SafetyNode：安全哨兵（threaded，每帧必检，fail-safe）。

与功能视觉（VisionNode）的三个反差（对话第 6 轮决策）：
  1. 失败语义：图像中断 / 检测超时 → 按侵入处理（绝不回退 GT）
  2. 帧策略：每帧必检（漏检窗口 = 危险窗口），用轻量模型保延迟
  3. 响应路径：快通道 /safety_state.latest 由 ArmController 每拍读；
     任务挂起走慢通道（ChargingAction 订阅本话题）
两级分区：减速区（限速继续）→ 停止区（冻结 · 保持夹持）。
"""
from __future__ import annotations
import threading
import time as pytime
from enum import Enum

from rclike import Node


class Zone(Enum):
    NORMAL = "normal"
    SLOW = "slow"       # 减速区：臂限速（如 25%）继续作业
    STOP = "stop"       # 停止区：冻结运动 · 保持夹持 · 等待确认


class SafetyNode(Node):
    threaded = True

    def __init__(self, bus, clock, image_topic: str,
                 slow_radius: float = 0.8, stop_radius: float = 0.4,
                 image_timeout: float = 0.2):
        super().__init__("safety_node", bus, clock)
        self.declare_parameter("slow_radius", slow_radius, "减速区阈值 (m)")
        self.declare_parameter("stop_radius", stop_radius, "停止区阈值 (m)")
        self.declare_parameter("image_timeout", image_timeout, "图像超时即失明 (s)")
        self._img = None
        self._seen_first = False               # 启动宽限：未收到首帧前不视为失明
        self._last_alive_wall = None             # 最近一帧的墙钟时刻（流中断判定）
        self._stop_evt = threading.Event()
        self._pub = self.create_publisher("/safety_state")
        self.zone = Zone.NORMAL
        self.create_subscription(image_topic, self._on_img)

    def _on_img(self, img, stamp):
        self._seen_first = True
        self._last_alive_wall = pytime.time()
        self._img = (img, stamp)

    def _detect_intruders(self, img) -> list:
        """TODO: 轻量检测（YOLO-nano 级）+ 人体分割 → 侵入者轮廓（世界系）。
        膨胀补偿：轮廓按运动速度外扩（人 1.5m/s × 检测周期 + 裕量），
        等价于对"下一刻可能在哪里"做保守估计。"""
        return []

    def _min_distance(self, intruders) -> float:
        """TODO: 侵入点到工作区凸包 / 最近臂体的最小距离。"""
        return float("inf")

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
            d = self._min_distance(self._detect_intruders(img))
            if d < self.get_parameter("stop_radius"):
                self._set_zone(Zone.STOP, f"intruder {d:.2f}m")
            elif d < self.get_parameter("slow_radius"):
                self._set_zone(Zone.SLOW, f"intruder {d:.2f}m")
            else:
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
