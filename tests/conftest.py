"""测试全局配置。

MUJOCO_GL=egl：无显示环境（CI）下 MuJoCo 离屏渲染用 EGL 后端。
必须在任何模块 import gl_context 之前设置（其后端分支在模块级缓存，
事后改环境变量无效）；conftest 加载先于所有测试模块 import，正好。
已显式设置 MUJOCO_GL 的环境不受影响（setdefault）。
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
