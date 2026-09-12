"""RenderNode：相机离屏渲染节点（threaded worker，独立渲染线程）。

渲染管线（按用户要求显式使用 MjrContext，而非 mujoco.Renderer 封装）：
  GLContext（GLFW 隐藏窗口 / EGL，按 MUJOCO_GL 选择，每实例一个）
    → mujoco.MjrContext（每相机一个，离屏帧缓冲）
    → mjv_updateScene → mjr_render → mjr_readPixels → RGB 图像话题

线程模型（渲染移出主循环，主线程只做物理与控制）：
  - GL 上下文线程绑定，一个 context 同一时刻只能 current 在一个线程：
    每实例独立 GLContext（构造期在主线程 init/create_window——launch
    组装顺序保证先于 viewer 线程，避免 GLFW init 竞态）；
    make_current / MjrContext / 渲染循环全程锁定在本实例 worker 线程
  - 离屏渲染含同步回读（mjr_readPixels），开销大（软件渲染下可达
    80ms+/帧），放主线程会把主循环拖到个位数 Hz——故为 threaded 旁路节点
  - 数据交接走 rclike 标准模式：主线程 publish 快照 → 回调写槽 +
    Event 唤醒；worker 消费最新帧（跳帧不积压），图像 topic 的订阅
    回调（Safety/Vision 只写引用）与 latest 读取均为线程安全
  - 节流按墙钟（fps 参数）：与 SafetyNode 的墙钟超时判定语义一致
  - GL 初始化失败（无显示环境）自动降级禁用并告警，不阻塞整个栈；
    此时 SafetyNode 收不到图像（启动宽限语义下保持放行，日志可查）

显示与数据流解耦：本节点只产图像话题；主窗口见 viewer_node（launch_passive），
相机画面窗口见 cam_show_node（OpenCV），两者独立开关。
"""
from __future__ import annotations

import threading
import time as pytime

import numpy as np
import mujoco

from rclike import Node


class RenderNode(Node):
    threaded = True                          # 旁路渲染线程，不进主循环

    # 类级标志：GL 初始化失败一次后全进程不再尝试——失败后重试会触发
    # mujoco 内部 C++ 静态初始化递归崩溃（__cxa_guard_acquire）
    _gl_broken = False

    def __init__(self, bus, clock, model, camera_name: str,
                 out_topic: str, fps: float = 10.0,
                 width: int = 640, height: int = 480):
        super().__init__(f"render_{camera_name}", bus, clock)
        self.model = model                       # 只读共享
        self.rdata = mujoco.MjData(model)        # 渲染专用 data（隔离，worker 私有）
        self.camera = camera_name
        self._period = 1.0 / fps                 # 墙钟节流周期（worker 用）
        self._pub = self.create_publisher(out_topic)
        self._snap = None                        # 最新快照槽（原子替换引用）
        self._snap_evt = threading.Event()       # 新快照通知（主线程 set）
        self._stop_evt = threading.Event()
        self._ctx = None                         # MjrContext（worker 线程内创建）
        self._disabled = False
        self._thread = None
        self._gl = None                          # 实例私有 GL 上下文（线程绑定）
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
        # GL 上下文在主线程构造期创建（不 make_current）：
        # GLFW init/create_window 非线程安全，须在 viewer 线程启动前完成
        # （launch 组装顺序：renders 先于 viewer_node add）；
        # context 随后由本实例 worker 线程独占使用。
        # 每实例独立 context：一个 GL context 同一时刻只能绑定一个线程。
        # GL 初始化失败（无 GL 环境）自动降级禁用并告警，不阻塞整个栈。
        if not RenderNode._gl_broken:
            try:
                from mujoco.rendering.classic import gl_context
                self._gl = gl_context.GLContext(640, 480)
            except Exception as e:            # glfw init / 无 GL 库等
                RenderNode._gl_broken = True   # 广播：全进程放弃 GL
                self._gl = None
                self._disabled = True
                self.log.warn(f"GL 上下文创建失败，渲染全部禁用: {e}")
        else:
            self._disabled = True             # 兄弟节点已判定 GL 不可用

    def _on_snap(self, qpos, stamp):
        """快照回调（主线程同步调用）：仅存引用 + 唤醒 worker。"""
        self._snap = (qpos, stamp)
        self._snap_evt.set()

    # ---------- 生命周期（threaded 节点） ----------
    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name=self.name)
        self._thread.start()

    def _worker(self):
        """渲染线程：GL 绑定 → 循环渲染最新快照 → 退出前释放 MjrContext。"""
        # MjrContext 惰性创建（必须在本线程 make_current 后）；
        # gladLoadGL / EGL 无设备等失败则降级禁用（栈继续运行）
        try:
            self._gl.make_current()
            self._ctx = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        except Exception as e:
            RenderNode._gl_broken = True       # 广播：全进程放弃 GL
            self._ctx = None
            self._disabled = True
            self.log.warn(f"GL 初始化失败，渲染全部禁用: {e}")
            return
        while not self._stop_evt.is_set():
            # 等新快照（至多等一个周期，兼作节流上限）
            self._snap_evt.wait(timeout=self._period)
            self._snap_evt.clear()
            if self._disabled or self._snap is None:
                continue
            # 墙钟节流：观察用途无需灌满（下游 SafetyNode 超时判定同为墙钟）
            now = pytime.monotonic()
            if now - getattr(self, "_last_wall", 0.0) < self._period - 1e-3:
                continue
            self._last_wall = now
            qpos, stamp = self._snap
            self._snap = None                 # 处理最新帧，跳帧不积压
            # 写入自己的 data；派生量重算也在这里（不影响主 MjData）
            self.rdata.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.rdata)
            # 显式 MjrContext 渲染：场景更新 → 离屏渲染 → 读回像素
            mujoco.mjv_updateScene(self.model, self.rdata, self._vopt,
                                   None, self._vcam,
                                   mujoco.mjtCatBit.mjCAT_ALL.value, self._scene)
            mujoco.mjr_render(self._rect, self._scene, self._ctx)
            mujoco.mjr_readPixels(self._img, None, self._rect, self._ctx)
            # GL 原点在左下需垂直翻转；copy 断开与复用缓冲的引用
            self._pub.publish(self._img[::-1].copy(), stamp=stamp)
        # 线程退出前释放 MjrContext（GL 资源须在绑定线程释放）
        try:
            if self._ctx is not None:
                self._gl.make_current()
                self._ctx.free()
        except Exception:
            pass
        finally:
            self._ctx = None

    def shutdown(self):
        """主线程调用：通知退出 → join（worker 自释 MjrContext）→ 释放 GL。"""
        self._stop_evt.set()
        self._snap_evt.set()                 # 唤醒等待中的 worker 以尽快退出
        if self._thread is not None:
            self._thread.join(timeout=2.0)   # 等 worker 退出（含渲染中一帧）
        if self._gl is not None:             # destroy_window：创建线程调用
            try:
                self._gl.free()
            except Exception:
                pass
            self._gl = None
