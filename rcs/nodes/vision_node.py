"""VisionNode：通用视觉 worker（threaded），按角色参数化。

双相机分工（对话第 5 轮）：
  e2h —— 眼在手外：粗定位 / 末端位姿估计 / 障碍物（跳帧 · 最新值语义）
  eih —— 眼在手上：目标精定位（相机位姿由 FK 给出，只解相机系下目标位姿）

检测实现（仿真验证期方案：ArUco + PnP，接口按实施顺序第 1 步）：
  - 图像 → ArUco 检测（DICT_4X4_50，按 marker_id 过滤目标码）→
    solvePnP → 码面位姿 → **标定板→目标物偏差补偿** → 目标物位姿
  - eih：输出相机系 {"target_pose": (pos, rot)}（真机迁移只换内参）
  - e2h：经固定相机外参转世界系 {"target_pose": (pos, rot)}
  - 偏差补偿（本轮新增）：标定板贴在目标物旁（板心与目标物刚性固定），
    构造期从模型读两 body 的初始位姿推导 T_target_in_face（检测面系下
    目标物的位姿），运行时 target = face ∘ T_target_in_face——
    仿真里精确已知；真机对应手眼标定产物，均只替换 _detect
  - 内参由 MuJoCo 相机 fovy 推出（方形像素针孔）；e2h 外参构造期读一次
  - 检测面偏移：PnP 原点在图案面中心，body 原点在板中心——
    face_offset = ±板半厚（e2h 俯视 +z 面 / eih 看 -z 面，实测两种
    朝向下 PnP 旋转均与 marker body 旋转一致，位置差面偏移）
  - 周期日志（本轮新增）：worker 每 log_period 秒汇总检出率与最新
    目标位姿——视觉链路可观测（此前检测静默，无法判断是否工作）
  - overlay（本轮新增）：输出带角点像素坐标，供 CamShowNode 画检测框
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
                 marker_size: float = 0.05, marker_id: int | None = None,
                 marker_body: str | None = None,
                 target_body: str | None = None,
                 face_offset: float = 0.0,
                 log_period: float = 5.0):
        super().__init__(f"vision_{role}", bus, clock)
        assert role in ("e2h", "eih"), "role 必须是 e2h / eih"
        self.role = role
        self._period = 1.0 / fps
        self._marker_size = marker_size       # ArUco 实物边长 (m)
        self._marker_id = marker_id           # 指定 id（None=任意，取第一个）
        self._face_offset = face_offset       # 图案面沿 marker body +z 偏移
        self._log_period = log_period         # 周期日志间隔 (s)
        self._pub = self.create_publisher(out_topic)
        self._img = None                     # 最新图像槽（原子替换引用）
        self._stop_evt = threading.Event()
        # ---- 周期日志统计 ----
        self._n_try = 0
        self._n_hit = 0
        self._t_log = pytime.monotonic()
        self._last_hit_info = None
        # ---- 相机模型（内参由 fovy 推出；e2h 另读固定外参）----
        self._K = None                       # (fx, fy, cx, cy)
        self._cam_pos = self._cam_R = None   # e2h 世界系外参
        # ---- 标定板→目标物 偏差（构造期从模型推导）----
        self._t_off_pos = None               # 目标物在检测面系下的位置
        self._t_off_rot = None               # 目标物在检测面系下的旋转
        if sim is not None and camera is not None:
            self._init_camera(sim, camera)
        if sim is not None and marker_body and target_body:
            self._init_target_offset(sim, marker_body, target_body)
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

    def _init_target_offset(self, sim, marker_body: str, target_body: str):
        """构造期推导"标定板→目标物"偏差（用户需求：板在目标物旁）。

        标定板与目标物刚性固连（板贴目标物旁），从模型初始位姿读：
          T_target_in_face = T_face⁻¹ ∘ T_target
        其中 T_face = marker body 位姿沿 +z 平移 face_offset（图案面中心）。
        运行时：target = 检测面位姿 ∘ T_target_in_face。
        """
        import mujoco
        model, data = sim.model, sim.data
        mujoco.mj_forward(model, data)       # 保证派生量就绪
        bm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
        bt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if bm < 0 or bt < 0:
            raise ValueError(
                f"marker/target body 不存在: {marker_body}/{target_body}")
        p_m = data.xpos[bm].copy()
        R_m = data.xmat[bm].reshape(3, 3).copy()
        p_face = p_m + R_m[:, 2] * self._face_offset
        p_t = data.xpos[bt].copy()
        R_t = data.xmat[bt].reshape(3, 3).copy()
        self._t_off_pos = R_m.T @ (p_t - p_face)
        self._t_off_rot = R_m.T @ R_t
        self.log.info(
            f"标定板→目标物偏差: |Δp|={np.linalg.norm(self._t_off_pos):.3f}m "
            f"(marker={marker_body} -> target={target_body})")

    def _on_img(self, img, stamp):
        self._img = (img, stamp)

    def _detect(self, img) -> dict:
        """ArUco 检测 + PnP + 目标物偏差补偿。

        坐标系约定：OpenCV solvePnP 输出（x 右 / y 下 / z 前）需转
        MuJoCo 相机系（x 右 / y 上 / z 后）——绕 x 翻 180°（M=diag(1,-1,-1)，
        M²=I）：p_mj = M·p_cv，R_mj = M·R_cv。渲染图像按 MuJoCo 相机生成，
        不转则位置差一个镜像翻转（实测误差达米级）。

        返回 {"target_pose": (pos, rot), "overlay": {...}}；无检测 / 无内参
        → {}（下游按缺数据语义处理，不猜测）。
        """
        import cv2
        if self._K is None:
            return {}
        fx, fy, cx, cy = self._K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        # 角点细化 APEX（AprilTag 风格亚像素）：MuJoCo 纹理线性滤波使
        # 黑白边缘轻微模糊，默认角点（边缘直线拟合）系统性内缩 ~1.6%
        # → PnP 深度偏大 ~2%（实测 0.34-1.5m 正对场景）；APEX 恢复到
        # <0.5%（0.34/0.5/1.0m 实测 0.1%/0.3%/0.3%）
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), dp)
        corners, ids, _ = det.detectMarkers(img)
        if ids is None or len(ids) == 0:
            return {}
        # 按 id 取目标码（多码同帧时明确指定，避免取错目标）
        if self._marker_id is not None:
            hits = [k for k, mid in enumerate(ids.ravel())
                    if mid == self._marker_id]
            if not hits:
                return {}
            k = hits[0]
        else:
            k = 0
        # 单 marker PnP：角点顺序 TL,TR,BR,BL（ArUco 惯例）→ marker 系 3D 点
        s = self._marker_size
        objp = np.array([[-s/2, s/2, 0], [s/2, s/2, 0],
                         [s/2, -s/2, 0], [-s/2, -s/2, 0]], dtype=float)
        ok, rvec, tvec = cv2.solvePnP(objp, corners[k], K, None)
        if not ok:
            return {}
        rot_cv, _ = cv2.Rodrigues(rvec)      # 检测面 → OpenCV 相机系
        M = np.diag([1.0, -1.0, -1.0])       # OpenCV → MuJoCo 相机系
        rot_face = M @ rot_cv                # 检测面 → MuJoCo 相机系
        pos_face = M @ tvec.ravel()
        # ---- 标定板→目标物偏差补偿（无 target_body 时恒等）----
        if self._t_off_pos is not None:
            pos = pos_face + rot_face @ self._t_off_pos
            rot = rot_face @ self._t_off_rot
        else:
            pos, rot = pos_face, rot_face
        if self.role == "eih":
            out = (pos, rot)
        else:                                # e2h：相机系 → 世界系（固定外参）
            out = (self._cam_R @ pos + self._cam_pos,
                   self._cam_R @ rot)
        overlay = {"corners": np.asarray(corners[k], dtype=float).tolist(),
                   "marker_id": int(ids.ravel()[k])}
        return {"target_pose": out, "overlay": overlay}

    def start(self):
        threading.Thread(target=self._loop, daemon=True,
                         name=self.name).start()

    def _loop(self):
        """worker：纯计算（只碰 numpy 图像，不碰 MjData），永远处理最新帧。

        异常保护：检测异常不再炸线程（此前 _detect 抛异常会静默杀死
        worker，视觉输出停更且无提示）——记录日志后继续。
        """
        while not self._stop_evt.is_set():
            snap = self._img
            if snap is None:
                pytime.sleep(0.001)
                continue
            img, stamp = snap
            try:
                result = self._detect(img)
            except Exception as e:
                self.log.error(f"检测异常: {e!r}")
                result = {}
            self._n_try += 1
            if result:
                self._n_hit += 1
                self._last_hit_info = result.get("target_pose")
                self._pub.publish(result, stamp=stamp)
            self._img = None                 # 跳帧不积压：旧帧直接丢弃
            self._periodic_log()
            pytime.sleep(self._period)

    def _periodic_log(self):
        """周期汇总：检出率 + 最新目标位姿（视觉链路可观测性）。"""
        now = pytime.monotonic()
        if now - self._t_log < self._log_period:
            return
        rate = self._n_hit / self._n_try * 100 if self._n_try else 0.0
        if self._last_hit_info is not None:
            p = self._last_hit_info[0]
            tip = (f"检出 {self._n_hit}/{self._n_try} 帧 ({rate:.0f}%)，"
                   f"目标 pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
        else:
            tip = f"检出 {self._n_hit}/{self._n_try} 帧 ({rate:.0f}%)"
        self.log.info(tip)
        self._n_try = self._n_hit = 0
        self._t_log = now

    def shutdown(self):
        self._stop_evt.set()
