"""PoseNode：视觉位姿聚合与新鲜度管理（主循环）。

数据源（本轮语义更新）：
  /ee_pose_vision（e2h）—— 标定板旁目标物（车插座）的世界位姿，
                            经 VisionNode 标定板→目标物偏差补偿
  /target_fine（eih）    —— 枪尾码板补偿后的枪位姿（相机系）

消费语义（fail-safe，与安全链路方向相反）：
  视觉新鲜（< freshness 窗口）→ 供视觉伺服；过期/缺失 → 下游回退 GT
  （仿真期 GT 恒可用；真机上应改为报错或冻结上一次值）
freshness 默认 0.3s：视觉 10fps + worker 处理节奏，最新结果滞后
可达 ~0.2s，窗口过窄（旧值 0.15s）会间歇性误判过期回退 GT。
"""
from __future__ import annotations

from rclike import Node


class PoseNode(Node):
    def __init__(self, bus, clock, sim):
        super().__init__("pose_node", bus, clock)
        self.sim = sim
        self.declare_parameter("freshness_s", 0.3, "视觉数据新鲜度窗口 (s)")
        self._vis_target = None      # (msg, stamp)：e2h 目标物世界位姿
        self._vis_gun = None         # (msg, stamp)：eih 枪位姿（相机系）
        self.create_subscription("/ee_pose_vision", self._on_vis_target)
        self.create_subscription("/target_fine", self._on_vis_gun)

    # ---- 视觉结果回调（只存引用，立即返回）----
    def _on_vis_target(self, msg, stamp):
        self._vis_target = (msg, stamp)

    def _on_vis_gun(self, msg, stamp):
        self._vis_gun = (msg, stamp)

    # ---- 任务状态机的查询入口 ----
    def get_target_pose(self):
        """e2h 目标物（车插座）世界位姿。新鲜返回 (pos, rot)，否则 None。"""
        return self._fresh(self._vis_target)

    def get_gun_pose(self):
        """eih 枪位姿（相机系）。新鲜返回 (pos, rot)，否则 None。"""
        return self._fresh(self._vis_gun)

    def _fresh(self, vis):
        if vis and (self.clock.now - vis[1]) < self.get_parameter("freshness_s"):
            return vis[0]["target_pose"]
        return None
