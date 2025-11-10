import cv2
import numpy as np
from typing import Tuple, Optional, Any

def process_image_for_streamlit(image_path: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    自动检测图像中的绿色和红色线条，并对每个轮廓进行独立处理。
    处理结果将叠加到原图上，并根据新逻辑调整透明度。
    此版本不显示图像，而是返回处理后的图像数组。

    Args:
        image_path (str): 待处理的图像文件路径。

    Returns:
        Tuple[Optional[np.ndarray], bool]:
            第一个元素是处理后的图像 (np.ndarray) 或 None (如果处理失败)。
            第二个元素是布尔值，指示处理是否成功（True 为处理成功，False 为未处理）。
    """
    # 颜色映射字典
    color_map = {
        'green': (0, 255, 0),  # BGR
        'red': (0, 0, 255),    # BGR
    }

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, False

    # 检查图像是否包含 alpha 通道
    has_alpha = img.shape[2] == 4
    if not has_alpha:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    def get_processed_mask(image, bgr_color):
        target_bgr_array = np.uint8([[bgr_color]])
        hsv_color = cv2.cvtColor(target_bgr_array, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_color[0][0].astype(int)

        h_min, h_max, s_min, s_max, v_min, v_max = [max(0, h - 10), min(179, h + 10),
                                                    max(0, s - 50), min(255, s + 50),
                                                    max(0, v - 50), min(255, v + 50)]

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if h_min < 0:
            mask1 = cv2.inRange(hsv_img, np.array([h_min + 180, s_min, v_min]), np.array([179, s_max, v_max]))
            mask2 = cv2.inRange(hsv_img, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max]))
            line_mask = cv2.bitwise_or(mask1, mask2)
        else:
            line_mask = cv2.inRange(hsv_img, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

        if not np.any(line_mask):
            return None, False

        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return processed_mask, True

    found_color = None
    processed_mask = None

    for color_name, bgr_value in color_map.items():
        temp_mask, is_found = get_processed_mask(img, bgr_value)
        if is_found:
            processed_mask = temp_mask
            found_color = color_name
            break

    if processed_mask is None:
        return img, False

    contours_external, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_result = np.zeros_like(img)

    for i, external_contour in enumerate(contours_external):
        x, y, w, h = cv2.boundingRect(external_contour)
        roi_processed_mask = processed_mask[y:y+h, x:x+w]
        contours_comp, hierarchy_comp = cv2.findContours(roi_processed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        internal_holes_count = 0
        if hierarchy_comp is not None:
            hierarchy_flat = hierarchy_comp[0]
            for j in range(len(contours_comp)):
                if hierarchy_flat[j][3] != -1:
                    internal_holes_count += 1

        temp_filled_bgra = np.zeros_like(filled_result)

        if internal_holes_count > 1:
            filled_mask_bgr = cv2.cvtColor(roi_processed_mask.copy(), cv2.COLOR_GRAY2BGR)
            if hierarchy_comp is not None:
                for j in range(len(contours_comp)):
                    parent_idx = hierarchy_comp[0][j][3]
                    if parent_idx != -1:
                        M = cv2.moments(contours_comp[j])
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            h_roi, w_roi = filled_mask_bgr.shape[:2]
                            flood_mask = np.zeros((h_roi + 2, w_roi + 2), np.uint8)
                            cv2.floodFill(filled_mask_bgr, flood_mask, (cx, cy), (255, 0, 0))

            blue_part_mask = cv2.inRange(filled_mask_bgr, (255, 0, 0), (255, 0, 0))
            temp_filled_bgra[y:y+h, x:x+w][blue_part_mask > 0] = (0, 0, 0, 255)

            external_filled_result_roi = np.zeros_like(roi_processed_mask, dtype=np.uint8)
            cv2.drawContours(external_filled_result_roi, [external_contour], -1, (255), thickness=cv2.FILLED, offset=(-x, -y))
            white_part_mask = cv2.inRange(cv2.cvtColor(external_filled_result_roi, cv2.COLOR_GRAY2BGR), (255, 255, 255), (255, 255, 255))
            white_part_mask[blue_part_mask > 0] = 0
            temp_filled_bgra[y:y+h, x:x+w][white_part_mask > 0] = (255, 255, 255, 255)
        else:
            cv2.drawContours(temp_filled_bgra, [external_contour], -1, (255, 255, 255, 255), thickness=cv2.FILLED)

        filled_result[(temp_filled_bgra[:,:,3] > 0)] = temp_filled_bgra[(temp_filled_bgra[:,:,3] > 0)]

    final_image = img.copy()

    final_alpha = np.zeros_like(final_image[:, :, 3])
    foreground_mask = final_image[:, :, 3] > 0
    roi_mask = filled_result[:, :, 3] > 0

    final_alpha[foreground_mask] = 80
    final_alpha[roi_mask] = 255
    final_image[:, :, 3] = final_alpha

    return final_image, True
