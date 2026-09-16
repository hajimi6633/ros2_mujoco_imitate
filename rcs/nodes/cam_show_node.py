"""CamShowNode：相机画面窗口——OpenCV imshow（主线程 GUI）。

订阅渲染节点发布的图像话题，按固定刷新率弹窗显示相机画面。
与主窗口（ViewerNode）相互独立开关：只想看主窗口时不开本节点。

视觉检测框叠加（本轮新增）：overlays 指定 {图像话题: 视觉输出话题}，
显示时读取对应 VisionNode 最新检测结果，在画面上画 ArUco 角点框 +
id 标注——视觉链路可直接观察（此前相机窗口看不出检测是否工作）。

注意：OpenCV 的 imshow 必须在主线程调用（Windows GUI 限制），
本节点为主循环节点，显示经 create_timer 节流（默认 ~15fps）。
"""
from __future__ import annotations

from rclike import Node


class CamShowNode(Node):
    def __init__(self, bus, clock, windows: list,
                 overlays: dict | None = None,
                 refresh: float = 0.066):
        """windows: [(image_topic, 窗口标题), ...]
        overlays: {image_topic: vision_out_topic}——对应窗口画检测框"""
        super().__init__("cam_show", bus, clock)
        import cv2                                # 懒加载：仅开启相机窗口时需要
        self.cv2 = cv2
        self._windows = list(windows)
        self._overlays = dict(overlays or {})
        self._shown: set = set()                  # 已创建过的窗口标题
        self._closed: set = set()                 # 被用户点 X 关闭的窗口
        self.create_timer(refresh, self._show)

    def _show(self):
        for topic, title in self._windows:
            if title in self._closed:
                continue                          # 用户已关闭：不再重建
            s = self.latest(topic)
            if s is None:
                continue
            # imshow 会自动重建被关闭的窗口；已显示过的窗口先查可见性，
            # 用户点 X 关闭后标记跳过（首次显示前不可查，否则误判）
            if title in self._shown and not self._visible(title):
                self._closed.add(title)
                self.log.info(f"相机窗口 '{title}' 已被关闭，停止刷新")
                continue
            self._shown.add(title)
            # MuJoCo 渲染为 RGB，OpenCV 显示需转 BGR
            img = self.cv2.cvtColor(s[0], self.cv2.COLOR_RGB2BGR)
            img = self._draw_overlay(topic, img)
            self.cv2.imshow(title, img)
        try:
            self.cv2.waitKey(1)
        except Exception:
            pass              # 全部窗口销毁后 QT 后端 waitKey 也可能抛错

    def _draw_overlay(self, image_topic, img):
        """有视觉检出时画角点框 + id（无检出原样返回）。"""
        vtopic = self._overlays.get(image_topic)
        if vtopic is None:
            return img
        s = self.latest(vtopic)
        if s is None or "overlay" not in s[0]:
            return img
        import numpy as np
        ov = s[0]["overlay"]
        corners = ov["corners"]
        # 视觉结果与当前帧存在 ~0.1s 时差，框可能轻微滞后（观察用途可接受）
        quad = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        self.cv2.polylines(img, [quad], True, (0, 255, 0), 2)
        u, v = int(corners[0][0][0]), int(corners[0][0][1])
        self.cv2.putText(img, f'id={ov["marker_id"]}', (u, v - 6),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    def _visible(self, title) -> bool:
        """窗口是否可见；QT 后端在全部窗口销毁后查询会抛错，视为已关闭。"""
        try:
            return bool(self.cv2.getWindowProperty(
                title, self.cv2.WND_PROP_VISIBLE))
        except Exception:
            return False

    # ---------- 窗口状态查询（run_stack 退出判定用） ----------
    def any_open(self):
        """是否仍有打开的相机窗口（已显示且未被用户关闭）。"""
        return bool(self._shown - self._closed)

    def ever_shown(self):
        """是否显示过至少一帧（区分"尚未出图"与"渲染禁用"）。"""
        return bool(self._shown)

    def shutdown(self):
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass
