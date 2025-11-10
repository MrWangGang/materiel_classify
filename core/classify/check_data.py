import os
from PIL import Image

def is_foreground_present(image_path):
    """
    检查给定的 mask 图像文件是否包含非黑色（非零）像素。
    返回 True 如果有前景，否则返回 False。
    """
    try:
        with Image.open(image_path) as img:
            # 转换为灰度图以便于检查，即使原始图像是RGB格式
            gray_img = img.convert('L')
            if gray_img.getbbox():
                return True
            else:
                return False
    except IOError:
        print(f"警告: 无法打开图像文件 {image_path}，已跳过。")
        return False

def find_no_foreground_masks(base_path):
    """
    递归检查指定路径下所有 mask 文件，并返回没有前景的文件的列表。
    """
    no_foreground_masks = []
    mask_dir = os.path.join(base_path, 'masks')

    if not os.path.isdir(mask_dir):
        print(f"警告: 路径 {mask_dir} 不存在，跳过检查。")
        return no_foreground_masks

    print(f"正在递归检查 {mask_dir} 中的所有 masks...")

    file_count = 0
    for dirpath, _, filenames in os.walk(mask_dir):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_count += 1
                file_path = os.path.join(dirpath, filename)
                if not is_foreground_present(file_path):
                    # 不再在遍历时打印，只将路径添加到列表中
                    no_foreground_masks.append(file_path)

    print(f"总共检查了 {file_count} 个 mask 文件。")
    return no_foreground_masks

if __name__ == "__main__":
    # 定义你的数据集根目录
    DATASET_ROOT = './datasets'

    print("--- 开始查找没有前景的训练集 masks ---")
    no_fg_train = find_no_foreground_masks(os.path.join(DATASET_ROOT, 'train'))

    print("\n" + "="*50)

    print("--- 开始查找没有前景的测试集 masks ---")
    no_fg_test = find_no_foreground_masks(os.path.join(DATASET_ROOT, 'test'))

    print("\n" + "="*50)

    print("--- 检查结果汇总：没有前景的 masks 列表 ---")

    # 打印训练集的结果
    if no_fg_train:
        print("\n训练集中没有前景的 mask 文件：")
        for path in no_fg_train:
            print(path)
    else:
        print("\n训练集中的所有 mask 都包含前景。")

    # 打印测试集的结果
    if no_fg_test:
        print("\n测试集中没有前景的 mask 文件：")
        for path in no_fg_test:
            print(path)
    else:
        print("\n测试集中的所有 mask 都包含前景。")