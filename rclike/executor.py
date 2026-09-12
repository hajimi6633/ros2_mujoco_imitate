"""单线程执行器：唯一的主循环。

  - 顺序确定性 = 仿真可复现（同 seed 同结果，无锁无竞态）
  - 节点执行顺序 = add 顺序 = 数据依赖顺序（任务 → 控制 → 物理）
  - 定时器统一调度：不同节点可按不同频率运行（控制 50Hz、任务状态机每拍）
  - threaded 节点不进主循环：add 时只启动其 worker 线程
"""
from __future__ import annotations
from time import perf_counter, sleep

from rclike.core import SimClock
from rclike.node import Node


class Executor:
    def __init__(self, clock: SimClock, dt: float = 0.05):
        self.clock = clock
        self.dt = dt                  # 控制拍周期（= n_substeps × timestep）
        self._nodes: list[Node] = []
        self._running = False

    def add(self, node: Node) -> Node:
        node.on_configure()
        node.on_activate()
        if node.threaded:
            node.start()              # 旁路 worker 线程，不占主循环
        else:
            self._nodes.append(node)
        return node

    def spin_once(self):
        """一个控制拍：主循环节点 on_tick → 到期定时器 → 推进仿真时钟。"""
        for node in self._nodes:
            node.on_tick()
        for node in self._nodes:
            self._fire_timers(node)
        self.clock.advance(self.dt)

    def _fire_timers(self, node: Node):
        t = self.clock.now
        for tm in node._timers:
            if tm["last"] is None or (t - tm["last"]) >= tm["period"] - 1e-9:
                tm["last"] = t
                tm["cb"]()

    def spin(self, n_ticks: int | None = None,
             stop_when=None, realtime: bool = False):
        """阻塞运行。n_ticks=None 表示无限运行（Ctrl+C 结束）。

        stop_when：每拍末调用的回调，返回 True 提前结束（正常走 shutdown）。
        realtime：按 dt 墙钟节拍运行（仿真 1x 实时）。
          默认 False 全速（CI/无头）；带显示窗口时应开启，
          否则主循环全速会让仿真时间狂飙（动作快进十几倍）。
          单拍超期时不追赶（如实呈现实时率 < 1x），落后过多重新对齐基准。
        """
        self._running = True
        i = 0
        next_t = perf_counter() + self.dt if realtime else None
        try:
            while self._running and (n_ticks is None or i < n_ticks):
                self.spin_once()
                i += 1
                if stop_when is not None and stop_when():
                    break
                if next_t is not None:
                    delay = next_t - perf_counter()
                    if delay > 0:
                        sleep(delay)
                    next_t += self.dt
                    # 落后超 1 拍：丢拍重新对齐，避免睡眠风暴连击
                    if next_t < perf_counter() - self.dt:
                        next_t = perf_counter() + self.dt
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        for node in self._nodes:
            node.on_deactivate()
            node.on_cleanup()
            node.shutdown()
        self._running = False
