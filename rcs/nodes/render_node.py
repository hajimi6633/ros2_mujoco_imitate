"""RenderNode：相机渲染节点（主线程，按 fps 节流）。

Windows 下 worker 线程内初始化 GL 会失败（gladLoadGL error），
故渲染在主线程执行（观察用途 10Hz 的开销可接受）：
  - 快照模式不变：独立 MjData，只消费 /state_snapshot，不碰主 MjData
  - 按仿真时钟节流（fps 参数），渲染不占满每个控制拍
  - GL 初始化失败（无显示环境）自动降级禁用并告警，不阻塞整个栈；
    此时 SafetyNode 收不到图像（启动宽限语义下保持放行，日志可查）

后续模型复杂、渲染耗时挤占控制环时，可再引入独立渲染进程恢复旁路化。
"""
from __future__ import annotations

import mujoco

from rclike import Node


class RenderNode(Node):
    # 类级标志：GL 初始化失败一次后全进程不再尝试——失败后重试会触发
    # mujoco 内部 C++ 静态初始化递归崩溃（__cxa_guard_acquire）
    _gl_broken = False

    def __init__(self, bus, clock, model, camera_name: str,
                 out_topic: str, fps: float = 10.0):
        super().__init__(f"render_{camera_name}", bus, clock)
        self.model = model                       # 只读共享
        self.rdata = mujoco.MjData(model)        # 渲染专用 data（隔离）
        self.camera = camera_name
        self._period = 1.0 / fps
        self._last_t = None
        self._pub = self.create_publisher(out_topic)
        self._snap = None                        # 最新快照槽（原子替换引用）
        self._renderer = None                     # 惰性创建（GL 失败降级）
        self._disabled = False
        self.create_subscription("/state_snapshot", self._on_snap)

    def _on_snap(self, qpos, stamp):
        """快照回调：仅存引用（微秒级）。"""
        self._snap = (qpos, stamp)

    def on_tick(self):
        if self._disabled or RenderNode._gl_broken or self._snap is None:
            if RenderNode._gl_broken and not self._disabled:
                self._disabled = True          # 兄弟节点已判定 GL 不可用
            return
        # 按仿真时钟节流：观察用途无需每个控制拍都渲染
        t = self.clock.now
        if self._last_t is not None and t - self._last_t < self._period - 1e-9:
            return
        self._last_t = t
        # 惰性创建 Renderer；无显示环境失败则降级禁用（栈继续运行）
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model)
            except Exception as e:                # gladLoadGL / 无 GL 库等
                RenderNode._gl_broken = True       # 广播：全进程放弃 GL
                self._disabled = True
                self.log.warn(f"GL 初始化失败，渲染全部禁用: {e}")
                return
        qpos, stamp = self._snap
        self._snap = None                         # 处理最新帧，跳帧不积压
        # 写入自己的 data；派生量重算也在这里（不影响主 MjData）
        self.rdata.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.rdata)
        self._renderer.update_scene(self.rdata, camera=self.camera)
        img = self._renderer.render()            # np.uint8 (H, W, 3)
        self._pub.publish(img, stamp=stamp)      # 时间戳随快照传递

    def shutdown(self):
        if self._renderer is not None:
            self._renderer.close()
