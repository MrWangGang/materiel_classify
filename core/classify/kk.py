import cv2
import math

# --- 焦距法计算得到的比例尺 ---
# 拍摄距离 Z = 97.0 mm
# 像素焦距 f_pixels ≈ 3610.53 像素 (基于物理焦距 6.86 mm 和 1.9 微米像素间距)
MM_PER_PIXEL = 0.02686 # 毫米/像素

# --- 图像加载 ---
# 假设 '1.png' 就在脚本当前目录下
image_path = '1.png'
img = cv2.imread(image_path)

if img is None:
    print(f"错误：无法加载图像文件。请检查文件路径: {image_path}")
    # 注意：如果加载失败，程序将在此处停止，不会运行后续代码
    exit()

height, width = img.shape[:2]

# --- 1. 确定目标两点的像素坐标 ---
# ！！！请根据您实际的图片分辨率来调整这两个坐标 ！！！
# 这里的坐标是基于图片中央的两个小点，进行的一个示例估算：
# 假设两点在中央且水平相距 50 像素（如果您图片的分辨率很大，这个距离可能需要更大）

# 示例估算坐标 (以图像中心为基准)
center_x = width // 2
center_y = height // 2

# 假设两个点水平分布，各距中心 25 像素
target_pt1_pixels = (center_x - 25, center_y)
target_pt2_pixels = (center_x + 25, center_y)

# --- 2. 计算目标两点之间的像素距离 ---
def calculate_pixel_distance(p1, p2):
    """计算两点 (x1, y1) 和 (x2, y2) 之间的欧几里得像素距离"""
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist

target_length_pixels = calculate_pixel_distance(target_pt1_pixels, target_pt2_pixels)

# --- 3. 计算实际距离 ---
target_real_length_mm = target_length_pixels * MM_PER_PIXEL
target_real_length_cm = target_real_length_mm / 10.0

# --- 结果打印 ---
print(f"--- 基于焦距法（iPhone 13 Pro Max, 9.7cm）的测量结果 ---")
print(f"使用的比例尺: {MM_PER_PIXEL:.5f} 毫米/像素")
print(f"图像尺寸: {width}x{height} 像素")
print("-" * 30)

print(f"目标两点（估算）的像素坐标: {target_pt1_pixels} 和 {target_pt2_pixels}")
print(f"目标两点像素距离: {target_length_pixels:.2f} 像素")
print("-" * 30)
print(f"**实际距离 (毫米): {target_real_length_mm:.2f} mm**")
print(f"**实际距离 (厘米): {target_real_length_cm:.2f} cm**")

# 可选：在图像上标记两点 (需要显示图像，如果您的环境不支持，请注释掉)
# cv2.circle(img, target_pt1_pixels, 5, (0, 0, 255), -1)
# cv2.circle(img, target_pt2_pixels, 5, (0, 0, 255), -1)
# cv2.line(img, target_pt1_pixels, target_pt2_pixels, (255, 0, 0), 2)
# cv2.imshow('Measured Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()