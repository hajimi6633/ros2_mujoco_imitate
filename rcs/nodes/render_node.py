"""RenderNode：相机离屏渲染节点（显式 MjrContext 管线，主线程）。

渲染管线（按用户要求显式使用 MjrContext，而非 mujoco.Renderer 封装）：
  GLContext（GLFW 隐藏窗口 / EGL，按 MUJOCO_GL 选择，全进程共享一个）
    → mujoco.MjrContext（每相机一个，离屏帧缓冲）
    → mjv_updateScene → mjr_render → mjr_readPixels → RGB 图像话题

线程模型（Windows gladLoadGL error 的根因即线程）：
  - GL 上下文必须在主线程创建/使用（GLFW 限制），本节点为主循环节点
  - 快照模式不变：独立 MjData，只消费 /state_snapshot，不碰主 MjData
  - 按仿真时钟节流（fps 参数），渲染不占满每个控制拍
  - GL 初始化失败（无显示环境）自动降级禁用并告警，不阻塞整个栈；
    此时 SafetyNode 收不到图像（启动宽限语义下保持放行，日志可查）

显示与数据流解耦：本节点只产图像话题；主窗口见 viewer_node（launch_passive），
相机画面窗口见 cam_show_node（OpenCV），两者独立开关。
"""
from __future__ import annotations

import numpy as np
import mujoco

from rclike import Node


class RenderNode(Node):
    # 类级共享 GL 上下文：两个相机复用同一个隐藏窗口上下文（惰性创建）
    _gl = None
    # 类级标志：GL 初始化失败一次后全进程不再尝试——失败后重试会触发
    # mujoco 内部 C++ 静态初始化递归崩溃（__cxa_guard_acquire）
    _gl_broken = False
    _active = 0                                 # 活跃实例数，用于共享上下文释放

    def __init__(self, bus, clock, model, camera_name: str,
                 out_topic: str, fps: float = 10.0,
                 width: int = 640, height: int = 480):
        super().__init__(f"render_{camera_name}", bus, clock)
        self.model = model                       # 只读共享
        self.rdata = mujoco.MjData(model)        # 渲染专用 data（隔离）
        self.camera = camera_name
        self._period = 1.0 / fps
        self._last_t = None
        self._pub = self.create_publisher(out_topic)
        self._snap = None                        # 最新快照槽（原子替换引用）
        self._ctx = None                         # MjrContext 惰性创建（GL 失败降级）
        self._disabled = False
        # 显式渲染四件套：选项 / 场景 / 相机（FIXED 指向模型相机）/ 视口矩形
        self._vopt = mujoco.MjvOption()
        self._scene = mujoco.MjvScene(model, 10000)   # maxgeom 同 Renderer 默认
        self._vcam = mujoco.MjvCamera()
        self._vcam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA.value, camera_name)
        self._vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # 分辨率不得超出模型离屏帧缓冲（默认 640x480），超出会被裁剪
        self._rect = mujoco.MjrRect(
            0, 0,
            min(width, model.vis.global_.offwidth),
            min(height, model.vis.global_.offheight))
        self._img = np.empty((self._rect.height, self._rect.width, 3),
                             dtype=np.uint8)
        self.create_subscription("/state_snapshot", self._on_snap)
        # GL 上下文与 MjrContext 必须在构造期创建（而非惰性到第一拍）：
        # --viewer 时 launch_passive 的后台线程会持续运行 GLFW 事件循环，
        # GLFW 非线程安全，第一拍才创建会与 viewer 线程并发 init/create_window，
        # 部分平台（Windows/WGL）直接失败 → 渲染禁用 → 相机窗口永不出现。
        # 构造期创建可确保先于 viewer 线程启动（launch 组装顺序保证）。
        # GL 初始化失败（无 GL 环境）仍自动降级禁用并告警，不阻塞整个栈。
        if not RenderNode._gl_broken:
            try:
                self._ensure_gl()
                self._ctx = mujoco.MjrContext(
                    self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            except Exception as e:            # gladLoadGL / 无 GL 库等
                RenderNode._gl_broken = True   # 广播：全进程放弃 GL
                RenderNode._gl = None
                self._ctx = None
                self._disabled = True
                self.log.warn(f"GL 初始化失败，渲染全部禁用: {e}")
        else:
            self._disabled = True             # 兄弟节点已判定 GL 不可用
        RenderNode._active += 1

    @classmethod
    def _ensure_gl(cls):
        """确保共享 GL 上下文存在且 current（主线程调用；重复调用无害）。"""
        if cls._gl is None:
            from mujoco.rendering.classic import gl_context
            cls._gl = gl_context.GLContext(640, 480)
        cls._gl.make_current()

    def _on_snap(self, qpos, stamp):
        """快照回调：仅存引用（微秒级）。"""
        self._snap = (qpos, stamp)

    def on_tick(self):
        if self._disabled or self._snap is None:
            return
        # 按仿真时钟节流：观察用途无需每个控制拍都渲染
        t = self.clock.now
        if self._last_t is not None and t - self._last_t < self._period - 1e-9:
            return
        self._last_t = t
        qpos, stamp = self._snap
        self._snap = None                         # 处理最新帧，跳帧不积压
        # 写入自己的 data；派生量重算也在这里（不影响主 MjData）
        self.rdata.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.rdata)
        # 显式 MjrContext 渲染：场景更新 → 离屏渲染 → 读回像素
        self._ensure_gl()
        mujoco.mjv_updateScene(self.model, self.rdata, self._vopt,
                               None, self._vcam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self._scene)
        mujoco.mjr_render(self._rect, self._scene, self._ctx)
        mujoco.mjr_readPixels(self._img, None, self._rect, self._ctx)
        # GL 原点在左下需垂直翻转；copy 断开与复用缓冲的引用
        self._pub.publish(self._img[::-1].copy(), stamp=stamp)

    def shutdown(self):
        try:
            if self._ctx is not None:
                if RenderNode._gl is not None:
                    RenderNode._gl.make_current()
                self._ctx.free()
        except Exception:
            pass                                  # 退出路径不做清理失败处理
        finally:
            self._ctx = None
            # 最后一个实例释放共享 GL 上下文
            RenderNode._active -= 1
            if RenderNode._active <= 0 and RenderNode._gl is not None:
                try:
                    RenderNode._gl.free()
                except Exception:
                    pass
                RenderNode._gl = None
