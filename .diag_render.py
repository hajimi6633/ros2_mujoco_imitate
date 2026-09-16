"""最小复现：1 vs 2 个 RenderNode 并发出图对比（EGL 双 context 嫌疑）。

A: 两个 RenderNode（复刻 launch 的双相机结构）
B: 单个 RenderNode
各 spin 40 ticks，每 10 tick 打印图像话题 stamp，看渲染是否持续出图。
"""
import os
import platform

if (platform.system() == "Linux" and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")):
    os.environ["MUJOCO_GL"] = "egl"

import time
from rclike import SimClock, Bus, Executor
from rcs.nodes.sim_node import SimNode
from rcs.nodes.render_node import RenderNode


def run(n_render: int, ticks: int = 600):
    clock, bus = SimClock(), Bus()
    sim = SimNode(bus, clock, "models/scene_table.xml")
    cams = [("cam_e2h", "/image_e2h"), ("cam_eih", "/image_eih")][:n_render]
    renders = [RenderNode(bus, clock, sim.model, c, t) for c, t in cams]
    ex = Executor(clock, dt=0.05)
    ex.add(sim)
    for r in renders:
        ex.add(r)
    t0 = time.monotonic()
    stamps = []
    for i in range(ticks):
        ex.spin_once()
        if i % 100 == 99:
            lt = [bus.topic(t).latest for _, t in cams]
            s = [round(x[1], 2) if x is not None else None for x in lt]
            stamps.append((i + 1, round(time.monotonic() - t0, 1), s))
    wall = time.monotonic() - t0
    for i, w, s in stamps:
        print(f"  tick={i} 墙钟{w}s 图像stamp={s}")
    print(f"  {n_render} 个 RenderNode: {ticks} ticks 墙钟 {wall:.1f}s"
          f"（{ticks/wall:.0f} ticks/s）\n")
    for r in renders:
        r.shutdown()


print("=== B: 单 RenderNode ===")
run(1)
print("=== A: 双 RenderNode（复刻 launch） ===")
run(2)
