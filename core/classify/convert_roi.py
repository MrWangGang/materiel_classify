import os
import glob
from PIL import Image
import numpy as np
from skimage import measure
from tqdm import tqdm
import math

# --- 1. 定义源数据目录、输出目录 和 最终图像尺寸 ---
SOURCE_DATA_DIR = "datasets"
OUTPUT_DATA_DIR = "data"  # 更改输出目录以区分
FINAL_SIZE = 224  # <--- 定义最终输出图像的固定尺寸 (例如 256x256)

print("--- 数据集预处理脚本（固定尺寸输出） ---")
print(f"将从 '{SOURCE_DATA_DIR}' 读取数据")
print(f"处理后的数据将保存到 '{OUTPUT_DATA_DIR}'，最终尺寸: {FINAL_SIZE}x{FINAL_SIZE}")
print("-" * 35)

# --- 2. 核心处理函数 (已修改以输出固定尺寸) ---
def process_image_and_mask(img_path, mask_path, final_size):
    """
    加载图像和掩码，找到最大的前景物体，将其裁剪出来，
    保持比例缩放，并粘贴到固定尺寸的黑色背景上。

    返回:
        PIL.Image: 固定尺寸 (final_size x final_size) 的处理后的图像。
        None: 如果在掩码中没有找到任何前景物体。
    """
    try:
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if image.size != mask.size:
            # 使用最近邻插值调整掩码大小
            mask = mask.resize(image.size, Image.NEAREST)

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np_binary = (mask_np > 0).astype(np.uint8)

        labels_mask = measure.label(mask_np_binary)
        regions = measure.regionprops(labels_mask)

        if regions:
            # 找到面积最大的前景
            largest_region = max(regions, key=lambda r: r.area)

            # 获取最大前景的包围盒
            minr, minc, maxr, maxc = largest_region.bbox

            # --- A. 裁剪出原始图像中对应包围盒的区域 ---
            cropped_original_image = image.crop((minc, minr, maxc, maxr))

            # 创建一个只包含最大前景的掩码
            region_mask = (labels_mask[minr:maxr, minc:maxc] == largest_region.label)
            region_mask_pil = Image.fromarray(region_mask.astype(np.uint8) * 255)

            # 创建一个新的图像，只包含物体前景，背景透明
            cropped_with_alpha = Image.new("RGBA", cropped_original_image.size)
            # 使用前景掩码作为透明度通道
            cropped_with_alpha.paste(cropped_original_image, (0, 0), region_mask_pil)


            # --- B. 缩放 (保持比例) ---
            original_width, original_height = cropped_with_alpha.size

            # 计算缩放比例，以适应 final_size 的最大边
            scale = min(final_size / original_width, final_size / original_height)

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # 保持比例缩放，使用高质量的双线性插值
            resized_image = cropped_with_alpha.resize((new_width, new_height), Image.Resampling.LANCZOS)


            # --- C. 填充到固定尺寸 ---
            # 创建一个 final_size x final_size 的黑色背景图像
            final_image = Image.new("RGB", (final_size, final_size), (0, 0, 0))

            # 计算粘贴位置 (居中)
            x_offset = (final_size - new_width) // 2
            y_offset = (final_size - new_height) // 2

            # 将缩放后的图像粘贴到黑色背景上 (使用 alpha 通道确保只粘贴前景)
            final_image.paste(resized_image, (x_offset, y_offset), resized_image)

            return final_image
        else:
            # 如果没有前景，返回 None
            return None
    except Exception as e:
        # 使用 math.ceil 是为了防止浮点数导致的问题，这里先注释掉
        print(f"\n处理文件 {img_path} 时出错: {e}")
        return None

# --- 3. 构建新数据集的主函数 (不变) ---
def create_cropped_dataset(source_dir, output_dir, final_size):
    """
    遍历源目录中的 train 和 test 集，处理每张图片并保存到输出目录。
    """
    # 确保根输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 'train' 和 'test' 两个子集
    for split in ['train', 'test']:
        print(f"\n--- 正在处理 '{split}' 数据集 ---")

        # 定义当前子集的图片和掩码目录
        images_base_dir = os.path.join(source_dir, split, 'images')
        masks_base_dir = os.path.join(source_dir, split, 'masks')
        output_base_dir = os.path.join(output_dir, split)

        if not os.path.exists(images_base_dir):
            print(f"目录 {images_base_dir} 不存在，跳过。")
            continue

        # 获取所有类别的文件夹
        class_dirs = [d for d in os.listdir(images_base_dir) if os.path.isdir(os.path.join(images_base_dir, d))]

        for class_name in class_dirs:
            class_image_dir = os.path.join(images_base_dir, class_name)
            class_mask_dir = os.path.join(masks_base_dir, class_name)
            class_output_dir = os.path.join(output_base_dir, class_name)

            # 创建对应的输出类别文件夹
            os.makedirs(class_output_dir, exist_ok=True)

            # 使用 os.path.join 确保路径正确
            image_files = glob.glob(os.path.join(class_image_dir, '*.jpg'))

            print(f"处理类别 '{class_name}' ({len(image_files)} 张图片)...")

            for img_path in tqdm(image_files, desc=class_name):
                # 构建对应的掩码路径和输出路径
                file_name = os.path.basename(img_path)
                mask_path = os.path.join(class_mask_dir, file_name)
                output_path = os.path.join(class_output_dir, file_name)

                # 检查掩码文件是否存在
                if not os.path.exists(mask_path):
                    continue

                # 调用核心函数处理图片，传入 final_size
                cropped_image = process_image_and_mask(img_path, mask_path, FINAL_SIZE)

                # 如果成功处理，则保存
                if cropped_image:
                    cropped_image.save(output_path)
                else:
                    print(f"警告: 在 {mask_path} 中未找到前景，跳过图片 {img_path}")

    print("\n--- 所有数据处理完成！ ---")
    print(f"处理后的数据集已保存在: {output_dir}")

# --- 4. 运行脚本 ---
if __name__ == "__main__":
    create_cropped_dataset(SOURCE_DATA_DIR, OUTPUT_DATA_DIR, FINAL_SIZE)