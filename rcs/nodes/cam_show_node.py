"""CamShowNode：相机画面窗口——OpenCV imshow（主线程 GUI）。

订阅渲染节点发布的图像话题，按固定刷新率弹窗显示相机画面。
与主窗口（ViewerNode）相互独立开关：只想看主窗口时不开本节点。

注意：OpenCV 的 imshow 必须在主线程调用（Windows GUI 限制），
本节点为主循环节点，显示经 create_timer 节流（默认 ~15fps）。
"""
from __future__ import annotations

from rclike import Node


class CamShowNode(Node):
    def __init__(self, bus, clock, windows: list,
                 refresh: float = 0.066):
        """windows: [(image_topic, 窗口标题), ...]"""
        super().__init__("cam_show", bus, clock)
        import cv2                                # 懒加载：仅开启相机窗口时需要
        self.cv2 = cv2
        self._windows = list(windows)
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
            if title in self._shown and \
                    not self.cv2.getWindowProperty(
                        title, self.cv2.WND_PROP_VISIBLE):
                self._closed.add(title)
                self.log.info(f"相机窗口 '{title}' 已被关闭，停止刷新")
                continue
            self._shown.add(title)
            # MuJoCo 渲染为 RGB，OpenCV 显示需转 BGR
            self.cv2.imshow(title,
                            self.cv2.cvtColor(s[0], self.cv2.COLOR_RGB2BGR))
        self.cv2.waitKey(1)

    def shutdown(self):
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass
