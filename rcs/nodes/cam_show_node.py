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
        self.create_timer(refresh, self._show)

    def _show(self):
        for topic, title in self._windows:
            s = self.latest(topic)
            if s is not None:
                # MuJoCo 渲染为 RGB，OpenCV 显示需转 BGR
                self.cv2.imshow(title,
                                self.cv2.cvtColor(s[0], self.cv2.COLOR_RGB2BGR))
        self.cv2.waitKey(1)

    def shutdown(self):
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass
