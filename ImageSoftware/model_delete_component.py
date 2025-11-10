import easyocr
import cv2
import numpy as np
import re
from typing import Tuple, Optional, Any

def process_image_by_x_cutoff(image_path: str,reader: easyocr.Reader) -> Tuple[Optional[np.ndarray], bool]:
    """
    加载图片，使用 EasyOCR 查找包含 'x' 且 'x' 前一个字符为数字或 'I' 的文本。
    从所有符合条件的文本框中，选择底边 Y 坐标最高（最远离顶部/最靠近底部）的一个作为截止线，
    **但仅考虑底边 Y 坐标位于图片中心点上方的文本框。**
    保留图片上半部分并保持透明度。

    Args:
        image_path (str): 待处理的图片文件路径。

    Returns:
        Tuple[Optional[np.ndarray], bool]:
            第一个元素是处理后的图片 (np.ndarray) 或 None (如果加载失败)。
            第二个元素是布尔值，指示是否找到了截止线并进行了裁剪（True 为处理成功，False 为未处理）。
    """
    # 2. 加载原始图片，保留 Alpha 通道（透明度）
    # 使用 cv2.IMREAD_UNCHANGED 尝试读取所有通道，包括 Alpha 通道
    img_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img_original is None:
        print(f"错误：无法加载图片 {image_path}。请检查文件路径。")
        return None, False

    # 统一图片为 4 通道 (BGRA)，便于处理透明度
    if img_original.ndim == 3:
        if img_original.shape[2] == 3:
            # 3 通道图片 (BGR)，添加不透明 Alpha 通道
            print("警告：原始图片为 3 通道，已添加完全不透明的 Alpha 通道。")
            img_bgra = cv2.cvtColor(img_original, cv2.COLOR_BGR2BGRA)
        elif img_original.shape[2] == 4:
            # 已经是 4 通道 (BGRA)
            img_bgra = img_original
        else:
            print("错误：图片通道数非 3 或 4。")
            return img_original, False
    elif img_original.ndim == 2:
        # 灰度图，转换为 BGRA
        print("警告：原始图片为灰度图，已转换为 BGRA 并添加完全不透明的 Alpha 通道。")
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    else:
        print(f"错误：图片维度异常: {img_original.ndim}")
        return img_original, False

    h, w, _ = img_bgra.shape
    # 计算图片中心线的 Y 坐标
    center_y = h / 2.0
    print(f"图片高度: {h}，中心线 Y 坐标: {center_y:.1f}")


    # 初始化变量：
    # 追踪符合条件的最高 Y 坐标（最远离顶部/最靠近底部），初始设置为 0
    highest_y_max_cutoff = 0
    found_match_above_center = False # 用于标记是否找到了中心点上方的有效匹配
    best_match_text = ""

    # 正则表达式：匹配前面是数字(0-9)或大写字母I，后面跟着小写x (不区分大小写)
    pattern = re.compile(r'(?:[0-9]|I)x', re.IGNORECASE)

    # 3. 调用 readtext() 方法
    try:
        # EasyOCR 不支持直接传入 numpy 数组，需要传入路径
        results = reader.readtext(image_path)
    except Exception as e:
        print(f"错误：EasyOCR readtext 失败: {e}")
        return img_bgra, False

    print("EasyOCR识别结果：")
    for (bbox, text, confidence) in results:
        # 将坐标转换为整数
        bbox = np.array(bbox).astype(int)
        print(f"文本: {text}, 坐标: {bbox.tolist()}, 置信度: {confidence:.2f}")

        # 检查文本是否符合 '数字x' 或 'Ix' 的模式
        if pattern.search(text):

            # 提取边界框的 Y 坐标，找到最大值 (即文本框的底边 Y 坐标)
            y_coords = [p[1] for p in bbox]
            current_y_max = max(y_coords)

            print(f"  --> 符合正则条件！当前文本底边 Y 坐标: {current_y_max}")

            # **【新增逻辑】**：判断文本框的底边是否在图片中心线 (h/2) *上方*。
            # 只有底边 Y 坐标严格小于中心线 Y 坐标才算在上方。
            if current_y_max < center_y:
                found_match_above_center = True
                print("  --> 位于中心线**上方**，视为有效截止线候选。")

                # 比较并更新最高的截止线坐标 (越大越靠近图片底部)
                if current_y_max > highest_y_max_cutoff:
                    highest_y_max_cutoff = current_y_max
                    best_match_text = text # 记录导致最高截止线的文本
            else:
                print(f"  --> 底边 Y 坐标 {current_y_max:.1f} >= 中心线 {center_y:.1f}，**忽略**。")


    # 4. 确定最终截止线并处理图片
    if not found_match_above_center:
        # 未找到符合条件的文本 *且* 位于中心点上方，**不进行处理**，返回原始图片和 False
        print("\n未找到符合 '数字x' 或 'Ix' 条件且位于中心点上方的文本。返回原始图片，处理状态：False。")
        return img_bgra, False
    else:
        # 找到了位于中心点上方的截止线，执行裁剪处理

        # 确保截止线坐标不会超出图片高度
        global_y_max = min(h, highest_y_max_cutoff)

        # 截止线 Y 坐标（像素索引）。+1 是为了包含 highest_y_max_cutoff 所在的行
        y_cutoff = min(h, global_y_max + 1)

        print(f"\n导致最高截止线（最远离顶部）的文本: {best_match_text}")
        print(f"确定的水平截止线（最高底边 Y 坐标 + 1）: {y_cutoff}")

        # 5. 创建一个完全透明的背景图像作为画布
        processed_img = np.zeros((h, w, 4), dtype=np.uint8)

        # 6. 复制保留区域：保留从图片顶部（0行）到 y_cutoff 行（不含）的所有像素
        # 保留 [0:y_cutoff] 范围
        processed_img[0:y_cutoff, 0:w] = img_bgra[0:y_cutoff, 0:w]

        print("图片处理完成，处理状态：True。")
        return processed_img, True