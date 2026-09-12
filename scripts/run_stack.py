"""运行入口：组装任务栈并运行。

用法:
  python -m scripts.run_stack --ticks 2500                     # 无显示（CI/无头）
  python -m scripts.run_stack --viewer                         # 只看主窗口
  python -m scripts.run_stack --viewer --cams                  # 主窗口 + 相机窗口
  python -m scripts.run_stack --cams                           # 只看相机窗口
  python -m scripts.run_stack --no-render --ticks 2500         # 关闭渲染数据流

显示说明（三者独立开关）：
  --viewer 主窗口 = MuJoCo 自带 passive viewer（launch_passive，需桌面环境）
  --cams  相机窗口 = cam_e2h / cam_eih 两路画面（OpenCV 窗口，需 --render）
  --render 渲染数据流 = 显式 MjrContext 离屏渲染管线，默认开；
          无 GL 环境自动降级（渲染节点禁用并告警，栈继续运行）
任务到达终态（成功/中止/取消）后自动退出并关闭全部窗口（--ticks 为拍数上限）。
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
    ap.add_argument("--render", action=argparse.BooleanOptionalAction,
                    default=True, help="渲染数据流（相机图像 + 安全哨兵）")
    ap.add_argument("--viewer", action="store_true",
                    help="主窗口：MuJoCo passive viewer（需桌面环境）")
    ap.add_argument("--cams", action="store_true",
                    help="相机画面窗口（OpenCV，需 --render）")
    args = ap.parse_args()

    import rcs.launch as launch
    build = getattr(launch, TASKS[args.task]["build"])

    ex, h = build(args.scene, vision=False, render=args.render,
                  viewer=args.viewer, cam_show=args.cams)
    # 任务名写入 goal：出现在"接受 goal"日志里，执行全程可追溯
    h["task"].send_goal({"task": args.task})
    modes = " ".join(filter(None, [
        f"渲染={'开' if args.render else '关'}",
        "主窗口" if args.viewer else "",
        "相机窗口" if args.cams else ""]))
    print(f"========== 任务: {args.task}（{TASKS[args.task]['desc']}）"
          f" | 拍数: {args.ticks} | {modes} ==========")
    # 任务到达终态（成功/中止/取消）即结束：自动走 shutdown 关闭全部窗口，
    # 否则 spin 会空转剩余拍数，相机窗口被 imshow 不断重建（关不掉）
    from rclike import GoalState
    action = h["task"].action

    def task_done():
        return action.handle is not None and action.handle.state in (
            GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED)

    ex.spin(n_ticks=args.ticks, stop_when=task_done)


if __name__ == "__main__":
    main()
