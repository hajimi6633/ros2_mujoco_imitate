"""测试全局配置。

Linux 无显示环境（CI）下 MuJoCo 离屏渲染切 EGL 后端。
必须在任何模块 import gl_context 之前设置（其后端分支在模块级缓存，
事后改环境变量无效）；conftest 加载先于所有测试模块 import，正好。
注意 egl 仅 Linux 合法（Windows/macOS 会 RuntimeError），本机桌面
（有 DISPLAY / 其他平台默认后端）不动；已显式设置的不受影响。
"""
import os
import platform

if (platform.system() == "Linux"
        and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")):
    os.environ["MUJOCO_GL"] = "egl"
