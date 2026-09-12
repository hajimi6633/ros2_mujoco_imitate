"""ViewerNode：主窗口——MuJoCo 自带 passive viewer。

launch_passive 自带独立渲染线程（GL 在其内部正确初始化），
本节点每控制拍调用 viewer.sync() 同步主 MjData 即可。
需要交互显示环境（本地桌面）；无头环境勿开启（构造即失败）。
"""
from __future__ import annotations

from rclike import Node


class ViewerNode(Node):
    def __init__(self, bus, clock, sim):
        super().__init__("viewer", bus, clock)
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    def on_tick(self):
        """每拍同步仿真状态到窗口（sync 轻量，50Hz 无压力）。"""
        if self._viewer.is_running():
            self._viewer.sync()

    def shutdown(self):
        try:
            self._viewer.close()
        except Exception:
            pass
