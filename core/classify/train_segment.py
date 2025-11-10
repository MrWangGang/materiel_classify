import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# --- 1. 定义数据路径和超参数 ---
data_dir = "datasets"
IMG_SIZE = (256, 256)
num_epochs = 50
batch_size = 4
lr=1e-4

# --- 2. 准备数据集文件列表 ---
print("--- 准备数据集文件列表 ---")
train_images = sorted(glob.glob(os.path.join(data_dir, "train", "images", "*", "*.jpg")))
train_masks = sorted(glob.glob(os.path.join(data_dir, "train", "masks", "*", "*.jpg")))
test_images = sorted(glob.glob(os.path.join(data_dir, "test", "images", "*", "*.jpg")))
test_masks = sorted(glob.glob(os.path.join(data_dir, "test", "masks", "*", "*.jpg")))

train_files = [{"image": img, "mask": mask} for img, mask in zip(train_images, train_masks)]
test_files = [{"image": img, "mask": mask} for img, mask in zip(test_images, test_masks)]

val_size = int(len(train_files) * 0.2)
train_data = train_files[:-val_size]
val_data = train_files[-val_size:]

print(f"训练集图像数量: {len(train_data)}")
print(f"验证集图像数量: {len(val_data)}")
print(f"测试集图像数量: {len(test_files)}")

# --- 3. 创建自定义 Dataset 和通用 Transforms ---
print("--- 创建自定义 Dataset 和 Transforms ---")
class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.file_list[idx]
        image_path = data["image"]
        mask_path = data["mask"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 确保掩码是二值化的
        mask = (np.array(mask) > 0).astype(np.uint8)
        mask = Image.fromarray(mask * 255)

        if self.transform:
            image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}

class ImageMaskTransform:
    """对图像和掩码同时进行的数据增强"""
    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

    def __call__(self, image, mask):
        # 调整大小
        resize = transforms.Resize(self.img_size)
        image = resize(image)
        mask = resize(mask)

        # 训练时应用数据增强
        if self.is_train:
            # 应用随机旋转
            angle = random.uniform(-15, 15)
            image = transforms.functional.rotate(image, angle, expand=False)
            mask = transforms.functional.rotate(mask, angle, expand=False)

            # 应用随机色彩抖动
            image = self.color_jitter(image)

            # 应用高斯模糊
            if random.random() > 0.5:
                # kernel_size 必须是奇数，这里使用3x3
                image = transforms.functional.gaussian_blur(image, kernel_size=3, sigma=(0.1, 2.0))

        # 转换为 Tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # 确保掩码是二值化的
        mask = (mask > 0.5).float()

        return image, mask

train_transform = ImageMaskTransform(IMG_SIZE, is_train=True)
val_transform = ImageMaskTransform(IMG_SIZE, is_train=False)
test_transform = ImageMaskTransform(IMG_SIZE, is_train=False)

train_ds = CustomDataset(train_data, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_ds = CustomDataset(val_data, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=1)
test_ds = CustomDataset(test_files, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=1)

# --- 4. 定义预训练的 U-Net 模型 ---
print("--- 定义预训练的 U-Net 模型 ---")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"当前设备: {device}")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)

loss_function = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def dice_score(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# --- 5. 训练、验证和测试循环 ---
print("--- 开始训练 ---")
best_dice_val = 0.0
train_losses = []
val_losses = []
test_losses = []
val_dices = []
test_dices = []
model_dir = "model/segment"
report_dir = "report/segment"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
print(f"最佳模型将保存至 {model_dir}")
print(f"图表将保存至 {report_dir}")

for epoch in range(num_epochs):
    print("-" * 30)
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练阶段
    model.train()
    epoch_loss = 0
    with tqdm(train_loader, desc=f"训练中 Epoch {epoch+1}") as pbar:
        for batch_data in pbar:
            inputs, masks = batch_data["image"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # 验证和测试阶段
    model.eval()
    with torch.no_grad():
        val_dice_sum = 0
        val_loss_sum = 0
        with tqdm(val_loader, desc=f"验证中 Epoch {epoch+1}") as pbar_val:
            for val_data in pbar_val:
                val_inputs, val_masks = val_data["image"].to(device), val_data["mask"].to(device)
                val_outputs = model(val_inputs)
                val_loss = loss_function(val_outputs, val_masks)
                val_loss_sum += val_loss.item()
                val_dice_sum += dice_score(val_outputs, val_masks).item()

        test_dice_sum = 0
        test_loss_sum = 0
        with tqdm(test_loader, desc=f"测试中 Epoch {epoch+1}") as pbar_test:
            for test_data in pbar_test:
                test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
                test_outputs = model(test_inputs)
                test_loss = loss_function(test_outputs, test_masks)
                test_loss_sum += test_loss.item()
                test_dice_sum += dice_score(test_outputs, test_masks).item()

    mean_dice_val = val_dice_sum / len(val_loader)
    mean_val_loss = val_loss_sum / len(val_loader)
    val_losses.append(mean_val_loss)
    val_dices.append(mean_dice_val)

    mean_dice_test = test_dice_sum / len(test_loader)
    mean_test_loss = test_loss_sum / len(test_loader)
    test_losses.append(mean_test_loss)
    test_dices.append(mean_dice_test)

    print(f"训练集平均 Loss: {epoch_loss:.4f}")
    print(f"验证集平均 Loss: {mean_val_loss:.4f}, 平均 Dice: {mean_dice_val:.4f}")
    print(f"测试集平均 Loss: {mean_test_loss:.4f}, 平均 Dice: {mean_dice_test:.4f}")

    if mean_dice_val > best_dice_val:
        best_dice_val = mean_dice_val
        torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
        print("保存了新的最佳模型！")

print("---")
print("训练完成！")

# --- 6. 绘制并保存训练指标图表 ---
print("--- 绘制并保存训练指标图表 ---")

epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.title('Training and Evaluation Losses Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(report_dir, 'losses.png'))

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, val_dices, label='Validation Dice')
plt.plot(epochs_range, test_dices, label='Test Dice')
plt.title('Validation and Test Dice Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(report_dir, 'dice_scores.png'))

print("图表已保存至 ./report/segment 目录。")
print("最佳模型已保存至 ./model/segment 目录。")