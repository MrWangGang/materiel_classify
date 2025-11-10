import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# <--- 修改点 1: 导入 Swin Transformer V2 Base 权重 --->
from torchvision.models import Swin_V2_B_Weights
from tqdm import tqdm
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# --- 0. 函数和类的定义 (无需修改) ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PreprocessedDataset(Dataset):
    """为预处理后的数据创建的自定义Dataset"""
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.file_list[idx]
        image = Image.open(data["image_path"]).convert("RGB")
        label = data["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':

    # 设置随机种子
    SEED = 42
    set_seed(SEED)
    print(f"随机种子已设置为: {SEED}")

    # --- 1. 定义数据路径和超参数 ---
    data_dir = "data"
    # Swin V2 B 标准输入尺寸是 256 或 224。我们保持 224。
    IMG_SIZE = 224
    num_epochs = 50
    batch_size = 16
    # 保持 AdamW 推荐的低学习率
    lr = 5e-5

    # --- 2. 准备数据集文件列表和分类标签 (无需修改) ---
    print("--- 准备数据集文件列表和分类标签 ---")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"错误：预处理后的数据集目录 '{data_dir}' 不存在。请先运行数据预处理脚本。")
        exit()

    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_classes)}
    num_classes = len(train_classes)

    train_val_files = []
    for cls_name in train_classes:
        class_dir = os.path.join(train_dir, cls_name)
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
        for img_path in image_paths:
            train_val_files.append({"image_path": img_path, "label": class_to_idx[cls_name]})

    train_data, val_data = train_test_split(
        train_val_files,
        test_size=0.25,
        random_state=SEED,
        stratify=[d['label'] for d in train_val_files]
    )

    test_data = []
    for cls_name in train_classes:
        class_dir = os.path.join(test_dir, cls_name)
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
        for img_path in image_paths:
            test_data.append({"image_path": img_path, "label": class_to_idx[cls_name]})

    print(f"训练集图像数量: {len(train_data)}")
    print(f"验证集图像数量: {len(val_data)}")
    print(f"测试集图像数量: {len(test_data)}")
    print(f"分类类别: {train_classes}")
    print(f"类别数量: {num_classes}")

    # --- 3. 创建数据增强和DataLoader (无需修改) ---
    print("--- 创建数据增强和DataLoader ---")

    # 数据增强保持不变
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.2),

        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = PreprocessedDataset(train_data, transform=train_transforms)
    val_ds = PreprocessedDataset(val_data, transform=val_test_transforms)
    test_ds = PreprocessedDataset(test_data, transform=val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- 4. 定义并加载预训练的 Swin Transformer V2 B 模型 ---
    print("--- 定义并加载预训练的 Swin Transformer V2 B 模型 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # 加载 Swin V2 B 模型并替换分类头
    model = models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)

    # 替换分类头 (Swin 的分类头在 'head' 属性中)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)

    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    # 保持 AdamW 优化器和学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # --- 5. 训练、验证和测试循环 (修改了保存逻辑) ---
    print("--- 开始训练 ---")

    # <--- 修改点 A: 将保存模型的指标改回验证集损失，并初始化为无穷大 --->
    best_loss_val = float('inf')

    # 为新模型创建一个新目录
    model_dir = "model/classification"
    report_dir = "report/classification"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print(f"最佳模型将保存至 {model_dir}")
    print(f"报告和图表将保存至 {report_dir}")

    train_losses, val_losses, test_losses = [], [], []
    val_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"训练中 Epoch {epoch+1}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)
        scheduler.step()

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="验证中"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_ds)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        test_loss, correct_test, total_test = 0.0, 0, 0
        all_test_preds, all_test_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="测试中"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_ds)
        test_accuracy = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"训练集 Loss: {epoch_loss:.4f}")
        print(f"验证集 Loss: {val_loss:.4f}, 验证集准确率: {val_accuracy:.4f}")
        print(f"测试集 Loss: {test_loss:.4f}, 测试集准确率: {test_accuracy:.4f}")

        print("\n--- 本轮验证集分类报告 ---")
        val_report_str_current = classification_report(all_val_labels, all_val_preds, target_names=train_classes, zero_division=0)
        print(val_report_str_current)

        print("\n--- 本轮测试集分类报告 ---")
        test_report_str_current = classification_report(all_test_labels, all_test_preds, target_names=train_classes, zero_division=0)
        print(test_report_str_current)

        # <--- 修改点 B: 将保存模型的条件改为 val_loss (最小化) --->
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            # 更新模型文件名
            torch.save(model.state_dict(), os.path.join(model_dir, "best_classify_model.pth"))
            print("保存了新的最佳模型（基于最低验证集损失）！")

            # 确保最佳报告也基于最低验证集损失的模型保存
            with open(os.path.join(report_dir, "best_classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(f"Best validation loss: {best_loss_val:.4f}\n\n")
                f.write(f"Corresponding test accuracy: {test_accuracy:.4f}\n\n")
                f.write("--- 验证集分类报告 ---\n")
                f.write(val_report_str_current)
                f.write("\n\n--- 测试集分类报告 ---\n")
                f.write(test_report_str_current)

    print("\n--- 训练完成！ ---\n")

    # --- 6. 绘制并保存训练指标图表 (无需修改) ---
    print("--- 绘制并保存训练指标图表 ---")
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'training_curves.png'))
    # plt.show() # 如果不需要在终端环境显示图表，可以注释掉这行
    print(f"图表已保存至 {report_dir} 目录。")
    print(f"最佳模型已保存至 {model_dir} 目录。")