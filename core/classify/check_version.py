import torch

# 打印 PyTorch 版本
print(torch.__version__)

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    print("GPU 可用！")
    # 打印可用的 GPU 数量
    print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
    # 打印当前选择的 GPU 名称
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}") # 假设我们看第一个 GPU (索引 0)
else:
    print("GPU 不可用，正在使用 CPU。")