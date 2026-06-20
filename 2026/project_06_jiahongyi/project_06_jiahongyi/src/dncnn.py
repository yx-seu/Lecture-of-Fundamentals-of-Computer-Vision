import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# =====================================================
# DnCNN 模型定义
# =====================================================

class DnCNN(nn.Module):
    """DnCNN for grayscale image denoising.

    Architecture (17-layer, BN-merged):
        Conv(in=channels, out=64, 3x3) + ReLU
        15 x (Conv(in=64, out=64, 3x3) + ReLU)
        Conv(in=64, out=channels, 3x3)

    Residual learning: output = input - predicted_noise
    """

    def __init__(self, channels=1):
        super(DnCNN, self).__init__()
        self.channels = channels
        depth = 17
        n_channels = 64

        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(channels, n_channels, 3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers: 15 x (Conv + ReLU)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Last layer: Conv only, no activation
        layers.append(nn.Conv2d(n_channels, channels, 3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise


# =====================================================
# 加载预训练模型
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DnCNN(channels=1)

checkpoint = torch.load("dncnn_pretrained.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

# =====================================================
# 数据集目录
# =====================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

sub_dirs = ["train", "val", "test"]

# =====================================================
# 开始处理
# =====================================================

for sub_dir in sub_dirs:

    folder_path = os.path.join(base_dir, sub_dir)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    print(f"\nProcessing {sub_dir}...")

    count = 0

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(".jpg"):
            continue

        # 防止重复处理
        if "_dncnn" in file_name:
            continue

        img_path = os.path.join(folder_path, file_name)

        try:

            # 读取灰度图
            img = Image.open(img_path).convert("L")

            img_np = np.array(img).astype(np.float32) / 255.0

            input_tensor = (
                torch.tensor(img_np)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            with torch.no_grad():

                output_tensor = model(input_tensor)

            output_np = (
                output_tensor
                .squeeze()
                .cpu()
                .numpy()
            )

            output_np = np.clip(output_np, 0, 1)

            output_img = (output_np * 255).astype(np.uint8)

            save_name = file_name.replace(
                ".jpg",
                "_dncnn.jpg"
            )

            save_path = os.path.join(
                folder_path,
                save_name
            )

            cv2.imwrite(save_path, output_img)

            count += 1

        except Exception as e:

            print(f"Error processing {file_name}")
            print(e)

    print(f"Finished {sub_dir}: {count} images")

print("\nAll done!")