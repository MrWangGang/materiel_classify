# pdf_processor2.py

import cv2
import numpy as np
import easyocr
import os
import re
import fitz
import io
# 新增：从 typing 导入 Callable 和 Optional，用于类型注解
from typing import Optional, List, Dict, Tuple, Any, Callable

# ======================================================================
# 核心配置 (常量)
# ======================================================================
# 目标文本模式 (例如：1X, 2X, IX, iX)
TARGET_REGEX_PATTERN = r'\b(?:[0-9]+[xX]|[Ii][xX])\b'

# 调试颜色配置 (BGR 格式)
DEBUG_LINE_COLOR = (0, 0, 255)       # 红色：用于主标签框 (细线/粗实线) 和连线 (细线)
DEBUG_BBOX_COLOR = (0, 255, 255)     # 黄色：用于所有候选轮廓 (细线模拟虚线)
# 子图框选颜色
SUBFIGURE_BOX_COLOR = (0, 0, 255)
SUBFIGURE_LABEL_COLOR = (255, 255, 255)
SUBFIGURE_LABEL_BG_COLOR = (0, 0, 255)

# 轮廓颜色列表 (BGR 格式)
CONTOUR_COLORS = [
    (255, 255, 0),  # 青色
    (0, 255, 255),  # 黄色
    (255, 0, 0),    # 蓝色
    (0, 255, 0),    # 绿色
    (0, 0, 255),    # 红色
]

# 形态学操作核
MORPH_KERNEL_3X3 = np.ones((1, 1), np.uint8)
MORPH_KERNEL_5X5 = np.ones((2, 2), np.uint8)

# OCR分析配置
MIN_OCR_CONFIDENCE = 0.1

# ======================================================================
# 辅助函数
# ======================================================================

_OCR_READER_INSTANCE: Optional[easyocr.Reader] = None

def initialize_ocr_reader(languages: List[str]) -> easyocr.Reader:
    """初始化或获取 EasyOCR 读取器实例。"""
    global _OCR_READER_INSTANCE
    if _OCR_READER_INSTANCE is None:
        print("正在加载 EasyOCR 模型...")
        try:
            _OCR_READER_INSTANCE = easyocr.Reader(languages, gpu=True)
            print("EasyOCR 模型加载完成 (使用 GPU)。")
        except Exception:
            _OCR_READER_INSTANCE = easyocr.Reader(languages, gpu=False)
            print("EasyOCR 模型加载完成 (使用 CPU)。")
    return _OCR_READER_INSTANCE

def get_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """计算边界框的中心点。"""
    return (box[0] + box[2] // 2, box[1] + box[3] // 2)

def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """计算两点之间的欧氏距离。"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_bbox_min_max(bbox_coords: Any) -> Tuple[int, int, int, int]:
    """从坐标点列表计算外接矩形 (xmin, ymin, xmax, ymax)。"""
    if not isinstance(bbox_coords, list) or not bbox_coords: return 0, 0, 0, 0
    x_min = int(min(point[0] for point in bbox_coords))
    y_min = int(min(point[1] for point in bbox_coords))
    x_max = int(max(point[0] for point in bbox_coords))
    y_max = int(max(point[1] for point in bbox_coords))
    return x_min, y_min, x_max, y_max

def check_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], min_overlap_ratio: float = 0.8) -> bool:
    """检查一个框是否与另一个框重叠达到一定比例。"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_overlap * y_overlap
    contour_area = w1 * h1
    if contour_area == 0: return False
    return intersection_area / contour_area >= min_overlap_ratio

def render_pdf_page_to_image(doc: fitz.Document, page_num: int, dpi: int) -> Optional[np.ndarray]:
    """将 PDF 页面渲染为 OpenCV 图像。"""
    try:
        page = doc.load_page(page_num)
        zoom_factor = dpi / 72.0
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=matrix)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"致命错误：渲染 PDF 页面 {page_num+1} 失败。详情：{e}")
        return None

# ======================================================================
# 核心处理函数
# ======================================================================

def process_rendered_image(img: np.ndarray, page_num_1based: int,
                           reader: easyocr.Reader, regex: re.Pattern, morph_kernel_size: int) -> List[Dict[str, Any]]:
    """从渲染的页面图像中提取包含目标文本的子图区域。"""
    page_num_str = str(page_num_1based)
    extracted_figures_data: List[Dict[str, Any]] = []

    results: List[Tuple[Any, str, float]] = reader.readtext(img, detail=1)
    all_target_texts: List[Dict[str, Any]] = []

    for item in results:
        if len(item) != 3: continue
        bbox_coords, text, _ = item
        x_min, y_min, x_max, y_max = get_bbox_min_max(bbox_coords)
        if regex.search(text):
            all_target_texts.append({'center': ((x_min + x_max) / 2, (y_min + y_max) / 2)})

    if not all_target_texts: return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closing = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp_figure_list: List[Dict[str, Any]] = []
    for target in all_target_texts:
        text_center_x, text_center_y = target['center']
        matched_contour = None
        for contour in contours:
            if cv2.contourArea(contour) < 1000: continue
            if cv2.pointPolygonTest(contour, (int(text_center_x), int(text_center_y)), False) >= 0:
                matched_contour = contour
                break
        if matched_contour is not None:
            cx, cy, cw, ch = cv2.boundingRect(matched_contour)
            current_crop_box = ((cx, cy), (cx + cw, cy + ch))
            if current_crop_box not in [item.get('crop_box') for item in temp_figure_list]:
                foreground_crop = img[cy:cy + ch, cx:cx + cw]
                if foreground_crop.size > 0:
                    temp_figure_list.append({
                        'image_array': foreground_crop, 'page_num_1based': page_num_1based,
                        'crop_box': current_crop_box, 'center_y': cy + ch // 2, 'center_x': cx + cw // 2
                    })

    temp_figure_list.sort(key=lambda item: (item['center_x'], item['center_y']))

    for i, data in enumerate(temp_figure_list):
        extracted_figures_data.append({
            'image_array': data['image_array'],
            'page_num_1based': data['page_num_1based'],
            'sub_figure_index': i + 1
        })
    print(f" -> 页面 {page_num_str} 完成子图提取，找到 {len(extracted_figures_data)} 个子图。")
    return extracted_figures_data

def extract_part_contours_from_pdf(
        pdf_file: str, page_range: List[int], target_dpi: int, ocr_languages: List[str],
        morph_kernel_size: int, min_contour_area: int, padding: int,
        output_debug_images: bool = False,
        # 新增：接收一个可选的回调函数，用于向GUI报告进度
        progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, bytes]:
    """
    主函数，从PDF中提取、匹配零件轮廓。
    返回一个字典，键为文件名，值为 PNG 格式的字节流。
    """
    if not os.path.exists(pdf_file):
        print(f"错误: 找不到 PDF 文件 '{pdf_file}'。")
        return {}

    contour_fill_mode = cv2.FILLED
    total_output_data: Dict[str, bytes] = {}
    IX_PATTERN = re.compile(r'([Ii])([xX])')

    try:
        doc = fitz.open(pdf_file)
        reader = initialize_ocr_reader(ocr_languages)
        target_regex = re.compile(TARGET_REGEX_PATTERN, re.IGNORECASE)
    except Exception as e:
        print(f"初始化失败: {e}")
        return {}

    start_page, end_page = page_range
    total_pdf_pages = len(doc)
    end_page = min(end_page, total_pdf_pages)

    # 新增：计算要处理的总页数，用于进度回调
    total_pages_to_process = end_page - start_page + 1

    for i, page_num_1based in enumerate(range(start_page, end_page + 1)):
        # 新增：调用进度回调函数
        if progress_callback:
            # i 是当前处理页面的 0-based 索引
            progress_callback(i, total_pages_to_process)

        page_num_0based = page_num_1based - 1
        print(f"\n--- 正在处理页面 {page_num_1based} ---")
        rendered_img = render_pdf_page_to_image(doc, page_num_0based, target_dpi)
        if rendered_img is None: continue

        extracted_figures_data = process_rendered_image(
            rendered_img, page_num_1based, reader, target_regex, morph_kernel_size)

        if not extracted_figures_data:
            print(" -> 未找到符合条件的子图。")
            continue

        for data_item in extracted_figures_data:
            img = data_item['image_array']
            page_num = data_item['page_num_1based']
            sub_index = data_item['sub_figure_index']
            if img is None or img.size == 0: continue
            H, W, _ = img.shape

            debug_img = img.copy() if output_debug_images else None

            ocr_results = reader.readtext(img, detail=1)
            main_labels, ocr_boxes = [], []

            for j, item in enumerate(ocr_results):
                if len(item) != 3 or item[2] < MIN_OCR_CONFIDENCE: continue
                bbox_coords, text, _ = item
                x_min, y_min, x_max, y_max = get_bbox_min_max(bbox_coords)
                box = (x_min, y_min, x_max - x_min, y_max - y_min)
                ocr_boxes.append(box)
                if target_regex.fullmatch(text.strip()):
                    main_labels.append({'id': j, 'text': text.strip(), 'box': box})

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, MORPH_KERNEL_5X5)
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidate_contours = [{'id': j, 'contour': cnt, 'box': cv2.boundingRect(cnt)}
                                  for j, cnt in enumerate(contours)
                                  if cv2.contourArea(cnt) > min_contour_area and not any(check_overlap(cv2.boundingRect(cnt), ocr_box) for ocr_box in ocr_boxes)]

            if debug_img is not None:
                for c_data in candidate_contours:
                    cv2.drawContours(debug_img, [c_data['contour']], 0, DEBUG_BBOX_COLOR, 1)
                for label_data in main_labels:
                    x, y, w, h = label_data['box']
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), DEBUG_LINE_COLOR, 1)

            final_matches, available_contour_ids = [], {c['id'] for c in candidate_contours}
            contours_by_id = {c['id']: c for c in candidate_contours}
            main_labels.sort(key=lambda lbl: lbl['box'][0])

            for label_data in main_labels:
                label_center = get_center(label_data['box'])
                eligible_contours = [contours_by_id[cid] for cid in available_contour_ids if contours_by_id[cid]['box'][0] < label_center[0]]
                if not eligible_contours: continue

                best_contour = min(eligible_contours, key=lambda c: calculate_distance((c['box'][0], c['box'][1] + c['box'][3]), label_center))
                dist = calculate_distance((best_contour['box'][0], best_contour['box'][1] + best_contour['box'][3]), label_center)
                final_matches.append({'contour_data': best_contour, 'text_data': label_data})
                available_contour_ids.remove(best_contour['id'])
                print(f"   -> 文本 '{label_data['text']}' 成功匹配到轮廓 C{best_contour['id']} (距离: {dist:.2f})")

            filtered_matches = []
            for match in final_matches:
                contour_box, label_box = match['contour_data']['box'], match['text_data']['box']
                contour_right_x = contour_box[0] + contour_box[2]
                label_center_x = label_box[0] + label_box[2] / 2
                if label_center_x > contour_right_x:
                    print(f"   -> 过滤: 文本 '{match['text_data']['text']}' 在轮廓 C{match['contour_data']['id']} 的右侧，已忽略。")
                else:
                    filtered_matches.append(match)

            filtered_matches.sort(key=lambda m: m['contour_data']['box'][0])
            for part_idx, pair in enumerate(filtered_matches, 1):
                contour, box, text, t_box = pair['contour_data']['contour'], pair['contour_data']['box'], pair['text_data']['text'], pair['text_data']['box']
                x, y, w, h = box
                x_p, y_p, x2_p, y2_p = max(0, x - padding), max(0, y - padding), min(W, x + w + padding), min(H, y + h + padding)
                if x2_p - x_p <= 0 or y2_p - y_p <= 0: continue

                extracted_img_bgra = np.zeros((y2_p - y_p, x2_p - x_p, 4), dtype=np.uint8)
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, contour_fill_mode)
                cropped_mask = mask[y_p:y2_p, x_p:x2_p]
                cropped_original = img[y_p:y2_p, x_p:x2_p]

                extracted_img_bgra[:, :, :3] = cropped_original
                extracted_img_bgra[:, :, 3] = cropped_mask

                _, buffer = cv2.imencode('.png', extracted_img_bgra)
                cleaned_text = IX_PATTERN.sub(r'1\2', text)
                filename = f"轮廓图/page{page_num}_sub{sub_index}_part{part_idx}_{cleaned_text}.png"
                total_output_data[filename] = buffer.tobytes()

                if debug_img is not None:
                    contour_color = CONTOUR_COLORS[(part_idx - 1) % len(CONTOUR_COLORS)]
                    cv2.drawContours(debug_img, [contour], 0, contour_color, 3)
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), contour_color, 2)
                    cv2.rectangle(debug_img, (t_box[0], t_box[1]), (t_box[0] + t_box[2], t_box[1] + t_box[3]), DEBUG_LINE_COLOR, 3)
                    cv2.line(debug_img, (box[0], box[1] + box[3]), get_center(t_box), DEBUG_LINE_COLOR, 1)

            if debug_img is not None:
                _, debug_buffer = cv2.imencode('.png', debug_img)
                debug_filename = f"调试图/debug_page{page_num}_sub{sub_index}.png"
                total_output_data[debug_filename] = debug_buffer.tobytes()
                print(f" -> 已收集调试图: {debug_filename}")

    doc.close()
    print(f"\n--- 处理完成！共收集了 {len(total_output_data)} 个文件字节流。---")
    return total_output_data