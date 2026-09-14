"""VisionNode：通用视觉 worker（threaded），按角色参数化。

双相机分工（对话第 5 轮）：
  e2h —— 眼在手外：粗定位 / 末端位姿估计 / 障碍物（跳帧 · 最新值语义）
  eih —— 眼在手上：目标精定位（相机位姿由 FK 给出，只解相机系下目标位姿）

检测实现（仿真验证期方案：ArUco + PnP，接口按实施顺序第 1 步）：
  - 图像 → ArUco 检测（DICT_4X4_50）→ solvePnP → marker 位姿
  - eih：输出相机系 {"target_fine": (pos, rot)}（真机迁移只换内参）
  - e2h：经固定相机外参转世界系 {"ee_pose": (pos, rot)}
    （marker 贴在末端即末端估计；贴在目标即粗定位——输出键按 role 固定）
  - 内参由 MuJoCo 相机 fovy 推出（方形像素针孔）；e2h 外参构造期读一次
  算法替换期（第 2 步）：YOLO 检测 + FoundationPose / ICP（仿真 GT 做监督），
  真机迁移期（第 3 步）：手眼标定（AX=XB）+ 外参标定——均只替换 _detect。

注意：当前 launch 的 vision 开关默认关（run_stack 写死 False），
本节点算法已实现并通过合成图像单测（tests/test_vision_aruco.py），
接线启用待 ArUco 码贴入场景（mesh 纹理）后打开。
"""
from __future__ import annotations
import threading
import time as pytime

import numpy as np

from rclike import Node


class VisionNode(Node):
    threaded = True

    def __init__(self, bus, clock, image_topic: str, out_topic: str,
                 role: str = "e2h", fps: float = 10.0,
                 sim=None, camera: str | None = None,
                 marker_size: float = 0.05):
        super().__init__(f"vision_{role}", bus, clock)
        assert role in ("e2h", "eih"), "role 必须是 e2h / eih"
        self.role = role
        self._period = 1.0 / fps
        self._marker_size = marker_size       # ArUco 实物边长 (m)
        self._pub = self.create_publisher(out_topic)
        self._img = None                     # 最新图像槽（原子替换引用）
        self._stop_evt = threading.Event()
        # ---- 相机模型（内参由 fovy 推出；e2h 另读固定外参）----
        self._K = None                       # (fx, fy, cx, cy)
        self._cam_pos = self._cam_R = None   # e2h 世界系外参
        if sim is not None and camera is not None:
            self._init_camera(sim, camera)
        self.create_subscription(image_topic, self._on_img)

    def _init_camera(self, sim, camera: str):
        """构造期（主线程）从模型读内参；e2h 另存固定外参。"""
        import mujoco
        model, data = sim.model, sim.data
        mujoco.mj_forward(model, data)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise ValueError(f"相机 '{camera}' 不存在")
        fovy = model.cam_fovy[cid]
        h, w = 480, 640                      # 渲染分辨率（RenderNode 默认）
        f = (h / 2) / np.tan(np.radians(fovy) / 2)
        self._K = (f, f, w / 2, h / 2)
        if self.role == "e2h":
            self._cam_pos = data.cam_xpos[cid].copy()
            self._cam_R = data.cam_xmat[cid].reshape(3, 3).copy()

    def _on_img(self, img, stamp):
        self._img = (img, stamp)

    def _detect(self, img) -> dict:
        """ArUco 检测 + PnP → 按 role 输出位姿。

        e2h: {"ee_pose": (pos_world, rot_world)}   marker 贴末端时
        eih: {"target_fine": (pos_cam, rot_cam)}
        无检测 / 无内参 → {}（下游按缺数据语义处理，不猜测）。
        """
        import cv2
        if self._K is None:
            return {}
        fx, fy, cx, cy = self._K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        corners, ids, _ = det.detectMarkers(img)
        if ids is None or len(ids) == 0:
            return {}
        # 单 marker PnP：角点顺序 TL,TR,BR,BL（ArUco 惯例）→ marker 系 3D 点
        s = self._marker_size
        objp = np.array([[-s/2, s/2, 0], [s/2, s/2, 0],
                         [s/2, -s/2, 0], [-s/2, -s/2, 0]], dtype=float)
        ok, rvec, tvec = cv2.solvePnP(objp, corners[0], K, None)
        if not ok:
            return {}
        rot_cam, _ = cv2.Rodrigues(rvec)      # marker → 相机系旋转
        pos_cam = tvec.ravel()
        if self.role == "eih":
            return {"target_fine": (pos_cam, rot_cam)}
        # e2h：相机系 → 世界系（固定外参）
        pos_w = self._cam_R @ pos_cam + self._cam_pos
        rot_w = self._cam_R @ rot_cam
        return {"ee_pose": (pos_w, rot_w)}

    def start(self):
        threading.Thread(target=self._loop, daemon=True,
                         name=self.name).start()

    def _loop(self):
        """worker：纯计算（只碰 numpy 图像，不碰 MjData），永远处理最新帧。"""
        while not self._stop_evt.is_set():
            snap = self._img
            if snap is None:
                pytime.sleep(0.001)
                continue
            img, stamp = snap
            result = self._detect(img)
            self._pub.publish(result, stamp=stamp)
            self._img = None                 # 跳帧不积压：旧帧直接丢弃
            pytime.sleep(self._period)

    def shutdown(self):
        self._stop_evt.set()
