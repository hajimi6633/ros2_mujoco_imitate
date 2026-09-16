"""针对 1a_eih.png 的码板检测失败分析：rejected 列表 + 裁剪放大。"""
import cv2
import numpy as np

img = cv2.imread("diag_frames2/1a_eih.png")
dp = cv2.aruco.DetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
det = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), dp)
corners, ids, rejected = det.detectMarkers(img)
print("ids =", None if ids is None else ids.ravel())
print(f"rejected 数 = {len(rejected) if rejected is not None else 0}")
if rejected:
    for i, r in enumerate(rejected[:5]):
        c = r[0]
        print(f"  rejected[{i}] 中心=({c[:,0].mean():.0f},{c[:,1].mean():.0f}) "
              f"边长={np.mean([np.linalg.norm(c[j]-c[(j+1)%4]) for j in range(4)]):.0f}px")

# 几何预测码板区域 u 52-214, v 147-309：裁剪 ×2 放大保存
crop = img[140:320, 40:230]
cv2.imwrite("crop_marker.png", cv2.resize(crop, None, fx=2, fy=2))
# 灰度直方图（对比度检查）
g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
print(f"码板区灰度: min={g.min()} max={g.max()} 双峰? "
      f"P10={np.percentile(g,10):.0f} P90={np.percentile(g,90):.0f}")
# 在裁剪图上再检测
c2, i2, r2 = det.detectMarkers(crop)
print("裁剪区 ids =", None if i2 is None else i2.ravel(),
      f"rejected={len(r2) if r2 is not None else 0}")
