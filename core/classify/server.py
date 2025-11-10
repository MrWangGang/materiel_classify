import os
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure
import segmentation_models_pytorch as smp
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 配置常量 ---
IMG_SIZE_SEG = (256, 256)
IMG_SIZE_CLS = 224

# 注意：请确保 'model/' 目录和以下文件路径在运行 Flask 的环境中存在！
SEG_MODEL_PATH = "model/segment/best_metric_model.pth"
CLS_MODEL_PATH = "model/classification/best_classify_model.pth"

CLASS_NAMES = [
    '右前车门上铰链加强板总成+30136B2410218001',
    '右后门上铰链加强板总成+30136B2410239001',
    '左前车门上铰链加强板总成+30136B2410215001',
    '左后门上铰链加强板总成+30136B2410242001'
]

# --- 全局模型和设备初始化 ---
seg_model = None
cls_model = None
device = None

def load_models():
    """
    加载分割模型和分类模型到内存，只执行一次。
    """
    global seg_model, cls_model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在加载模型到设备: {device} ---")

    if not os.path.exists(SEG_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        print("错误：找不到模型文件。请检查路径:")
        print(f"分割模型路径: {SEG_MODEL_PATH}")
        print(f"分类模型路径: {CLS_MODEL_PATH}")
        return None, None

    try:
        # 3. 加载分割模型 (Unet/ResNet18)
        seg_model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
        seg_model.to(device)
        seg_model.eval()

        # 4. 加载分类模型 (ViT-B/16)
        cls_model = models.swin_v2_b(weights=None)
        num_ftrs = cls_model.head.in_features
        cls_model.head = nn.Linear(num_ftrs, len(CLASS_NAMES))
        cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=device))
        cls_model.to(device)
        cls_model.eval()

        print("--- 模型加载成功。 ---")
        return seg_model, cls_model

    except Exception as e:
        print(f"模型加载过程中发生错误: {e}")
        return None, None

# --- 核心处理函数 ---

def pil_to_base64(pil_image: Image.Image) -> str | None:
    """将 PIL Image 对象转换为 Base64 编码的字符串。"""
    if pil_image is None:
        return None
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_seg_mask(image: Image.Image, threshold: float):
    """根据分割模型获取物料掩膜. (需要 256x256 输入)"""
    global seg_model

    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE_SEG),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = seg_model(img_tensor)

    output_mask = (torch.sigmoid(output) > threshold).squeeze().cpu().numpy().astype(np.uint8)
    output_mask_pil = Image.fromarray(output_mask * 255).resize(original_size, Image.NEAREST)
    return np.array(output_mask_pil) > 0

def get_largest_roi_and_bbox(image: Image.Image, mask: np.ndarray):
    """从掩膜中提取面积最大的物料区域 (ROI)."""
    labels_mask = measure.label(mask)
    regions = measure.regionprops(labels_mask)
    if not regions:
        return None, None

    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox

    cropped_size = (maxc - minc, maxr - minr)

    processed_image = Image.new("RGBA", cropped_size)
    cropped_original_image = image.crop((minc, minr, maxc, maxr))

    region_mask_bbox = (labels_mask[minr:maxr, minc:maxc] == largest_region.label)
    region_mask_pil = Image.fromarray(region_mask_bbox.astype(np.uint8) * 255)

    processed_image.paste(cropped_original_image, (0, 0), region_mask_pil)

    return processed_image, (minr, minc, maxr, maxc)

def classify_roi(roi_image: Image.Image):
    """
    对提取的 ROI 进行物料分类识别，包含保持比例缩放和居中填充步骤。
    返回: 预测结果 和 标准化后的 PIL 图像。
    """
    global cls_model

    # --- 1. 保持比例缩放并居中填充到 IMG_SIZE_CLS (224x224) 的黑色背景上 ---
    final_size = IMG_SIZE_CLS

    original_width, original_height = roi_image.size

    scale = min(final_size / original_width, final_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = roi_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    final_image_224 = Image.new("RGB", (final_size, final_size), (0, 0, 0)) # <--- 224x224 图像
    x_offset = (final_size - new_width) // 2
    y_offset = (final_size - new_height) // 2

    final_image_224.paste(resized_image, (x_offset, y_offset), resized_image)

    # 2. 对标准化后的图像进行 PyTorch Transform 和分类
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    roi_tensor = transform(final_image_224).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cls_model(roi_tensor)

    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    return predicted_class, confidence, final_image_224 # <--- 返回 224x224 图像

# --- Flask 应用设置 ---

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 在应用启动时加载模型
seg_model, cls_model = load_models()

@app.route('/', methods=['GET'])
def home():
    """简单的健康检查接口."""
    if seg_model and cls_model:
        return jsonify({"status": "模型服务已就绪", "device": str(device)}), 200
    else:
        return jsonify({"status": "错误：模型加载失败", "details": "请检查服务器日志中的文件路径错误。"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    预测接口。
    入参： multipart/form-data，包含一个文件字段 'file'。
    返回： 包含识别结果和三张 Base64 编码图片的 JSON。
    """
    if seg_model is None or cls_model is None:
        return jsonify({"error": "模型初始化失败。请检查服务器日志。"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "请求中没有文件部分。请以上传文件键名 'file' 上传图片。"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有选择文件。"}), 400

    if file:
        try:
            image_stream = io.BytesIO(file.read())
            original_image = Image.open(image_stream).convert("RGB")

            # --- 核心处理流程 ---
            # 1. 分割
            seg_threshold = 0.7
            seg_mask_np = get_seg_mask(original_image, threshold=seg_threshold)

            mask_image_pil = Image.fromarray(seg_mask_np.astype(np.uint8) * 255)

            # 2. ROI 提取
            roi_image, bbox = get_largest_roi_and_bbox(original_image, seg_mask_np)

            # 3. 检查是否提取到 ROI
            if roi_image is None:
                return jsonify({
                    "error": "分割成功但未检测到有效目标 (ROI)。",
                    "details": "分割区域在应用置信度阈值后可能太小或过于分散。",
                    "images": {
                        "original": pil_to_base64(original_image),
                        "mask": pil_to_base64(mask_image_pil),
                        "roi": None
                    }
                }), 400

            # 4. 分类 (同时获取 224x224 标准化图像)
            predicted_label_idx, confidence, final_image_224 = classify_roi(roi_image) # <--- 接收 224 图像
            predicted_class = CLASS_NAMES[predicted_label_idx]

            # 5. 可视化：在原图上绘制 BBOX
            draw_image = original_image.copy()
            draw = ImageDraw.Draw(draw_image)
            draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline="red", width=3)

            # 6. 编码所有图片为 Base64
            original_b64 = pil_to_base64(draw_image)
            mask_b64 = pil_to_base64(mask_image_pil)
            roi_b64 = pil_to_base64(final_image_224) # <--- 使用 224x224 图像作为 ROI 输出

            # 7. 返回包含图片和结果的 JSON
            return jsonify({
                "material_id": predicted_class,
                "confidence": float(f"{confidence:.4f}"),
                "result_message": "识别完成。",
                "bbox": [int(b) for b in bbox] if bbox else None,
                "images": {
                    "original": original_b64,
                    "mask": mask_b64,
                    "roi": roi_b64 # 现在是 224x224 图像
                }
            }), 200

        except Exception as e:
            print(f"预测因异常失败: {e}")
            return jsonify({"error": f"处理过程中发生内部服务器错误: {str(e)}"}), 500

if __name__ == '__main__':
    if seg_model is None or cls_model is None:
        print("应用启动失败：模型加载错误。")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)