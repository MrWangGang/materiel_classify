import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure
import segmentation_models_pytorch as smp

IMG_SIZE_SEG = (256, 256)
IMG_SIZE_CLS = 224

SEG_MODEL_PATH = "model/segment/best_metric_model.pth"
CLS_MODEL_PATH = "model/classification/vit_b_16_best.pth"

CLASS_NAMES = ['右前车门上铰链加强板总成+30136B2410218001', '右后门上铰链加强板总成+30136B2410239001', '左前车门上铰链加强板总成+30136B2410215001', '左后门上铰链加强板总成+30136B2410242001']

st.set_page_config(
    page_title="基于分割与分类的端到端物料识别",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("基于分割与分类的端到端物料识别")
st.markdown("上传一张图片，应用将自动完成：**物料分割** → **面积最大ROI提取** → **物料识别**。")
st.markdown("---")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(SEG_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        st.error(f"错误：找不到模型文件。请确保以下路径存在：")
        st.code(f"分割模型: {SEG_MODEL_PATH}")
        st.code(f"分类模型: {CLS_MODEL_PATH}")
        return None, None, None
    seg_model_loaded = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    seg_model_loaded.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
    seg_model_loaded.eval()
    cls_model_loaded = models.vit_b_16(weights=None)
    num_ftrs = cls_model_loaded.heads.head.in_features
    cls_model_loaded.heads.head = nn.Linear(num_ftrs, len(CLASS_NAMES))
    cls_model_loaded.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=device))
    cls_model_loaded.eval()
    return seg_model_loaded, cls_model_loaded, device

seg_model, cls_model, device = load_models()

if seg_model is None or cls_model is None:
    st.stop()

def get_seg_mask(image: Image.Image, threshold: float):
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
    labels_mask = measure.label(mask)
    regions = measure.regionprops(labels_mask)
    if not regions:
        return None, None
    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox
    cropped_size = (maxc - minc, maxr - minr)
    processed_image = Image.new("RGB", cropped_size, (0, 0, 0))
    cropped_original_image = image.crop((minc, minr, maxc, maxr))
    region_mask = (labels_mask[minr:maxr, minc:maxc] == largest_region.label)
    region_mask_pil = Image.fromarray(region_mask.astype(np.uint8) * 255)
    processed_image.paste(cropped_original_image, (0, 0), region_mask_pil)
    return processed_image, (minr, minc, maxr, maxc)

def classify_roi(roi_image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    roi_tensor = transform(roi_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cls_model(roi_tensor)
    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    return predicted_class, confidence

st.subheader("参数设置")
seg_threshold = st.slider("分割模型置信度阈值", 0.0, 1.0, 0.7, 0.05)
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("正在处理图片..."):
        original_image = Image.open(uploaded_file).convert("RGB")
        seg_mask_np = get_seg_mask(original_image, threshold=seg_threshold)
        roi_image, bbox = get_largest_roi_and_bbox(original_image, seg_mask_np)
        masked_image = Image.new("RGB", original_image.size)
        masked_image.paste(original_image, (0, 0), Image.fromarray(seg_mask_np.astype(np.uint8) * 255))
        if bbox:
            draw = ImageDraw.Draw(masked_image)
            draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline="red", width=3)
        if roi_image is not None:
            predicted_label_idx, confidence = classify_roi(roi_image)
            predicted_class = CLASS_NAMES[predicted_label_idx]
        else:
            predicted_class = "未检测到ROI"
            confidence = 0.0

    st.markdown("---")

    st.subheader("数据处理流程可视化")
    cols = st.columns([1, 1, 1, 1, 1.5])
    with cols[0]:
        st.image(original_image, caption="1. 原图", width='stretch')
    with cols[1]:
        st.image(seg_mask_np.astype(np.uint8) * 255, caption=f"2. 掩膜 (阈值: {seg_threshold})", width='stretch')
    with cols[2]:
        st.image(masked_image, caption="3. 掩膜映射到原图 (带包围盒)", width='stretch')
    with cols[3]:
        if roi_image is not None:
            st.image(roi_image, caption="4. 裁剪的ROI", width='stretch')
        else:
            st.markdown("<p style='text-align: center; color: red;'>未检测到ROI</p>", unsafe_allow_html=True)
            st.empty()
    with cols[4]:
        st.subheader("最终识别结果")
        if roi_image is not None:
            st.markdown(f"**预测类别**: `{predicted_class}`")
            st.markdown(f"**置信度**: `{confidence:.4f}`")
        else:
            st.warning("由于未能检测到ROI，无法进行识别。")

    st.markdown("---")

else:
    st.info("请上传一张图片以开始。")