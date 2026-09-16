"""对保存的 eih/e2h 帧跑 ArUco 检测，量化码板可见性/尺寸/位置。"""
import glob
import cv2
import numpy as np

det = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

for f in sorted(glob.glob("diag_frames/*eih*.png")):
    img = cv2.imread(f)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        # 找枪体红色区域（枪 rgba 0.9,0.2,0.2 → BGR 51,51,229）
        red = cv2.inRange(img, (0, 0, 150), (120, 120, 255))
        n_red = int(red.sum() / 255)
        ys, xs = np.nonzero(red)
        pos = (f"红枪像素中心=({xs.mean():.0f},{ys.mean():.0f})"
               if n_red > 100 else "无枪色")
        print(f"{f.split('/')[-1]}: 无码板检出 | {pos} 红像素={n_red}")
    else:
        for c, i in zip(corners, ids.ravel()):
            c0 = c[0]
            edges = [np.linalg.norm(c0[j] - c0[(j + 1) % 4]) for j in range(4)]
            print(f"{f.split('/')[-1]}: 检出 id={i} 边长={np.mean(edges):.0f}px "
                  f"中心=({c0[:,0].mean():.0f},{c0[:,1].mean():.0f}) "
                  f"画面640x480")
