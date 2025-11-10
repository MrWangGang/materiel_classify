import os
import json
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from tqdm import tqdm

# 辅助函数：将 RLE 格式解码为二进制掩码
def rle_to_binary_mask(rle, height, width):
    """
    将 RLE（游程编码）对象解码为二进制掩码。
    返回的 NumPy 数组中，前景像素为 1，背景为 0。
    """
    # 使用 pycocotools 提供的解码函数
    mask = coco_mask.decode(rle)
    return mask

def find_image_path(root_dir, image_filename):
    """
    递归地在根目录及其所有子目录中查找指定的图片文件，并返回其完整路径。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if image_filename in filenames:
            return os.path.join(dirpath, image_filename)
    return None

def create_masks_from_coco(images_dir, output_dir):
    """
    根据 COCO 格式的 JSON 标注文件，为图像生成对应的掩码文件，并保存到指定目录。
    该函数支持处理多边形和游程编码（RLE）两种分割格式。

    参数：
        images_dir (str): 包含原始图像和 COCO JSON 文件的根目录路径。
        output_dir (str): 保存生成的掩码的根目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    total_images_in_jsons = 0
    success_count = 0
    fail_count = 0

    json_files = [f for f in os.listdir(images_dir) if f.endswith('.json')]

    for filename in tqdm(json_files, desc="处理标注文件"):
        json_path = os.path.join(images_dir, filename)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"\n警告: 读取JSON文件 {json_path} 时出错：{e}。跳过。")
            continue

        if 'images' not in data or 'annotations' not in data:
            print(f"\n警告: {filename} 中缺少 'images' 或 'annotations' 键。跳过。")
            continue

        total_images_in_jsons += len(data['images'])

        for image_info in data['images']:
            img_name = image_info['file_name']
            img_id = image_info['id']
            img_width = image_info['width']
            img_height = image_info['height']

            original_img_path = find_image_path(images_dir, img_name)

            if not original_img_path:
                print(f"\n警告: 原始图像 {img_name} 未在 {images_dir} 及其子目录中找到。跳过。")
                fail_count += 1
                continue

            relative_path = os.path.relpath(original_img_path, images_dir)
            output_mask_path = os.path.join(output_dir, relative_path)

            # 创建一个用于最终掩码的 NumPy 数组，初始为全黑
            final_mask_np = np.zeros((img_height, img_width), dtype=np.uint8)

            # 遍历所有标注信息，并绘制到 final_mask_np 上
            for annotation in data['annotations']:
                if annotation['image_id'] == img_id:
                    if 'segmentation' in annotation and annotation['segmentation']:
                        segmentation = annotation['segmentation']

                        if isinstance(segmentation, dict) and 'counts' in segmentation:
                            try:
                                rle = {
                                    "counts": segmentation['counts'],
                                    "size": [img_height, img_width]
                                }
                                # 解码 RLE，得到 0/1 的 NumPy 数组
                                binary_mask_np = rle_to_binary_mask(rle, img_height, img_width)
                                # 将 1 转换为 255，并与最终掩码进行最大值合并
                                final_mask_np = np.maximum(final_mask_np, binary_mask_np * 255)
                            except Exception as e:
                                print(f"\n警告: 标注 {annotation['id']} 的RLE数据解码失败。错误：{e}。跳过。")
                                continue

                        elif isinstance(segmentation, list):
                            # 创建一个临时 PIL 图像来绘制多边形
                            temp_pil_mask = Image.new('L', (img_width, img_height), 0)
                            draw = ImageDraw.Draw(temp_pil_mask)

                            for segment in segmentation:
                                try:
                                    polygon_points = [(int(p), int(q)) for p, q in zip(segment[0::2], segment[1::2])]
                                    if polygon_points:
                                        draw.polygon(polygon_points, fill=255)
                                except (ValueError, IndexError) as e:
                                    print(f"\n警告: 标注 {annotation['id']} 的多边形数据格式不正确。错误：{e}")
                                    continue

                            # 将绘制好的多边形掩码转换为 NumPy 数组，并与最终掩码合并
                            final_mask_np = np.maximum(final_mask_np, np.array(temp_pil_mask))

            # 将最终的 NumPy 掩码数组转换为 PIL 图像，并保存
            final_mask_image = Image.fromarray(final_mask_np)
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            final_mask_image.save(output_mask_path, format='JPEG', quality=95)
            success_count += 1

    # ---
    # 打印最终统计结果
    # ---
    print("\n" + "-"*30)
    print("生成任务完成！")
    print(f"总计 JSON 文件中包含的图片数量：{total_images_in_jsons}")
    print(f"成功生成掩码的图片数量：{success_count}")
    print(f"失败的图片数量：{fail_count}")
    print("-" * 30 + "\n")

# ---
# 脚本入口
# ---
if __name__ == '__main__':
    images_directory = './datasets/test/images'
    masks_directory = './datasets/test/masks'
    create_masks_from_coco(images_directory, masks_directory)