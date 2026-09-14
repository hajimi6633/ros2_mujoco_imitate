"""运行入口：组装任务栈并运行。

用法:
  python -m scripts.run_stack --ticks 2500                     # 无显示（CI/无头）
  python -m scripts.run_stack --viewer                         # 只看主窗口
  python -m scripts.run_stack --viewer --cams                  # 主窗口 + 相机窗口
  python -m scripts.run_stack --cams                           # 只看相机窗口
  python -m scripts.run_stack --no-render --ticks 2500         # 关闭渲染数据流
  python -m scripts.run_stack --cams --vision                  # 相机窗口 + ArUco 视觉定位
  python -m scripts.run_stack --cams --safety                  # 相机窗口 + 安全哨兵

显示说明（独立开关，按需组合）：
  --viewer 主窗口 = MuJoCo 自带 passive viewer（launch_passive，需桌面环境）
  --cams  相机窗口 = cam_e2h / cam_eih 两路画面（OpenCV 窗口，需 --render）
  --vision 视觉定位 = ArUco + PnP（e2h 场景码板 / eih 枪尾码板，需 --render）
  --safety 安全哨兵 = 黄色标识物颜色检测（默认关，需 --render）
  --render 渲染数据流 = 显式 MjrContext 离屏渲染管线（渲染线程，供
          相机窗口/视觉/安全哨兵消费）；缺省自动推断：任一图像
          消费者（--cams/--vision/--safety）开启则开，否则关；
          无 GL 环境自动降级（渲染节点禁用并告警，栈继续运行）
退出语义：
  有显示窗口（--viewer / --cams）→ 不限拍数且按 dt 实时节拍（1x 实时），
  任务结束后窗口保持打开，手动关闭全部显示窗口（或 Ctrl+C）后退出；
  无显示窗口 → 全速运行，任务终态即退出，--ticks 为拍数上限（CI/无头）。
"""
from __future__ import annotations
import argparse
import os


def _gl_backend():
    """Linux 无显示环境时离屏渲染切 EGL（须先于任何 gl_context import 生效）。

    MuJoCo 的 gl_context 按环境变量 MUJOCO_GL 在模块级选定后端，
    且合法性校验与平台绑定（Linux: egl/glx/osmesa，Windows: wgl，
    macOS: cgl）——egl 仅 Linux 合法，跨平台设置会直接 RuntimeError。
    本机桌面（Linux 有 DISPLAY / Windows / macOS 默认后端可用）不动；
    已显式设置 MUJOCO_GL 的环境也不动。
    """
    import platform
    if (platform.system() == "Linux"
            and not os.environ.get("MUJOCO_GL")
            and not os.environ.get("DISPLAY")):
        os.environ["MUJOCO_GL"] = "egl"

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
                    default=None,
                    help="渲染数据流（相机图像）；缺省自动：任一图像消费者"
                         "（--cams/--vision/--safety）开启则开，否则关")
    ap.add_argument("--viewer", action="store_true",
                    help="主窗口：MuJoCo passive viewer（需桌面环境）")
    ap.add_argument("--cams", action="store_true",
                    help="相机画面窗口（OpenCV，需 --render）")
    ap.add_argument("--vision", action="store_true",
                    help="视觉定位节点（ArUco + PnP，需 --render）")
    ap.add_argument("--safety", action="store_true",
                    help="安全哨兵（黄色标识物颜色检测，默认关）")
    args = ap.parse_args()

    _gl_backend()                       # 先定 GL 后端再 import 栈（渲染节点）
    import rcs.launch as launch
    build = getattr(launch, TASKS[args.task]["build"])

    # 渲染数据流自动推断：相机窗口/视觉/安全哨兵都消费图像话题；
    # 仅 --viewer 时关闭——离屏渲染含同步回读，约占主循环 90%+，
    # 主窗口画面更新率 = 主循环频率，会被拖到个位数 Hz
    render = args.render if args.render is not None else \
        (args.cams or args.vision or args.safety)
    ex, h = build(args.scene, vision=args.vision, render=render,
                  viewer=args.viewer, cam_show=args.cams,
                  safety=args.safety)
    # 任务名写入 goal：出现在"接受 goal"日志里，执行全程可追溯
    h["task"].send_goal({"task": args.task})
    viewer, cams = h["viewer"], h["cam_show"]
    modes = " ".join(filter(None, [
        f"渲染={'开' if render else '关'}",
        "主窗口" if viewer else "",
        "相机窗口" if cams else "",
        "视觉" if args.vision else "",
        "安全哨兵" if args.safety else ""]))
    tail = ("1x 实时 · 关闭全部显示窗口后退出" if (viewer or cams)
            else f"拍数: {args.ticks}")
    print(f"========== 任务: {args.task}（{TASKS[args.task]['desc']}）"
          f" | {tail} | {modes} ==========")
    # 有显示窗口：不限拍数且按 dt 实时节拍（仿真以 1x 实时推进，
    # 否则主循环全速会让动作快进十几倍）；用户关窗后退出；
    # 无显示窗口：全速，任务终态即退出，--ticks 为拍数上限
    ex.spin(n_ticks=None if (viewer or cams) else args.ticks,
            stop_when=exit_when(h["task"].action, viewer, cams),
            realtime=bool(viewer or cams))


def exit_when(action, viewer, cams):
    """构造 spin 的 stop_when 退出判定。

    无显示窗口 → 任务终态（成功/中止/取消）即退出；
    有显示窗口 → 等用户关闭全部窗口（viewer + 相机）后退出。
    相机首帧未出且任务未终态时继续等（启动窗口期，防误退）。
    """
    from rclike import GoalState
    terminal = (GoalState.SUCCEEDED, GoalState.ABORTED, GoalState.CANCELED)

    def task_terminal():
        return action.handle is not None and action.handle.state in terminal

    def stop():
        if viewer is None and cams is None:
            return task_terminal()
        if viewer is not None and viewer.is_open():
            return False
        if cams is not None:
            if cams.any_open():
                return False
            if not cams.ever_shown() and not task_terminal():
                return False      # 首帧未出：正常启动窗口期，继续等
        return True

    return stop


if __name__ == "__main__":
    main()
