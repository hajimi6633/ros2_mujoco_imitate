"""运行入口：组装任务栈并运行。

用法:
  python -m scripts.run_stack --ticks 400              # 默认任务（无头裸跑）
  python -m scripts.run_stack --task charging --ticks 2500
  python -m scripts.run_stack --ticks 4000 --render    # 含渲染/安全节点

任务可观测性：
  --task 参数写入 goal；运行中看阶段日志（[1a] 完成 …）；
  进度经 feedback 话题（<task>/feedback）每拍发布；结束看 result。
"""
from __future__ import annotations
import argparse

# 已迁移的任务：名字 -> 组装器（后续 reach / pick_place 迁入后在此登记）
TASKS = {
    "charging": {
        "desc": "充电枪抓取-插拔 6 阶段任务",
        "build": "build_charging_stack",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="charging",
                    help=f"要执行的任务：{ {k: v['desc'] for k, v in TASKS.items()} }")
    ap.add_argument("--scene", default="models/scene_table.xml")
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--render", action="store_true",
                    help="创建渲染 + 安全节点（需 GL 环境）")
    ap.add_argument("--vision", action="store_true",
                    help="启用视觉节点（需先实现 VisionNode._detect）")
    args = ap.parse_args()

    import rcs.launch as launch
    build = getattr(launch, TASKS[args.task]["build"])

    ex, h = build(args.scene, vision=args.vision, render=args.render)
    # 任务名写入 goal：出现在"接受 goal"日志里，执行全程可追溯
    h["task"].send_goal({"task": args.task})
    print(f"========== 任务: {args.task}（{TASKS[args.task]['desc']}）"
          f" | 拍数: {args.ticks} | 渲染: {'开' if args.render else '关'} ==========")
    ex.spin(n_ticks=args.ticks)


if __name__ == "__main__":
    main()
