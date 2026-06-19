import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything.dpt import DepthAnything
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet
)

device = "cpu"

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    }
}

depth_model = DepthAnything(
    model_configs["vits"]
)

checkpoint = torch.load(
    "./models/depth_anything_vits14.pth",
    map_location=device
)

depth_model.load_state_dict(checkpoint)

depth_model.to(device)
depth_model.eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    PrepareForNet()
])


def depth_estimator(image):
    """
    image : PIL.Image

    return:
    {
        "depth": PIL.Image
    }
    """

    # PIL -> numpy
    image = np.array(image)

    h, w = image.shape[:2]

    # RGB [0,1]
    image = image.astype(np.float32) / 255.0

    transformed = transform(
        {
            "image": image
        }
    )

    input_tensor = torch.from_numpy(
        transformed["image"]
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        depth = depth_model(input_tensor)

        depth = F.interpolate(
            depth[:, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )[0, 0]

    depth = depth.cpu().numpy()

    # normalize -> 0~255
    depth = (
        depth - depth.min()
    ) / (
        depth.max() - depth.min()
    )

    depth = (depth * 255).astype(np.uint8)

    depth_image = Image.fromarray(depth)

    return {
        "depth": depth_image
    }


def process_depth_map(raw_depth_input, output_size=512):
    """
    全兼容版深度图后处理函数，支持所有transformers版本的返回格式
    """
    # ====================== 第一层：智能解析输入，提取纯深度数据 ======================
    depth_data = raw_depth_input

    # 1. 处理列表格式（批量返回的情况）
    if isinstance(depth_data, list):
        depth_data = depth_data[0]

    # 2. 处理字典格式（pipeline原始返回）
    if isinstance(depth_data, dict):
        # 优先取PIL格式的可视化深度图，其次取原始张量
        if "depth" in depth_data:
            depth_data = depth_data["depth"]
        elif "predicted_depth" in depth_data:
            depth_data = depth_data["predicted_depth"]
        else:
            raise ValueError(f"无法从输入中提取深度数据，可用字段：{list(depth_data.keys())}")

    # 3. 处理PyTorch张量格式
    if isinstance(depth_data, torch.Tensor):
        depth_data = depth_data.squeeze().cpu().numpy()

    # 4. 处理PIL图像格式，统一转为灰度numpy数组
    if isinstance(depth_data, Image.Image):
        depth_data = np.array(depth_data.convert("L"))

    # ====================== 第二层：数据类型标准化 ======================
    depth_np = np.array(depth_data)

    # 兜底处理object类型，强制转为数值数组
    if depth_np.dtype == object:
        depth_np = np.array(depth_np.tolist(), dtype=np.float32)
    else:
        depth_np = depth_np.astype(np.float32)

    # 统一归一化到0-255
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_max > depth_min:
        depth_np = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
    depth_cv = depth_np.astype(np.uint8)

    # ====================== 第三层：原有后处理逻辑 ======================
    # 标准化尺寸
    depth_cv = cv2.resize(depth_cv, (output_size, output_size), interpolation=cv2.INTER_AREA)
    
    # 全局对比度拉伸
    depth_normalized = cv2.normalize(depth_cv, None, 0, 255, cv2.NORM_MINMAX)
    
    # 高斯平滑去噪
    depth_smoothed = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
    
    # 直方图均衡化增强层次
    depth_equalized = cv2.equalizeHist(depth_smoothed)
    
    # 转回三通道RGB格式，匹配ControlNet输入要求
    return Image.fromarray(cv2.cvtColor(depth_equalized, cv2.COLOR_GRAY2RGB))