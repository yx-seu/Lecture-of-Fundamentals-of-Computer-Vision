import cv2
import numpy as np
from PIL import Image

def standardize_image(img, target_size=512):
    """
    通用图像标准化预处理：统一尺寸、色彩空间、像素范围
    输入：OpenCV BGR图像 或 PIL图像
    输出：标准化后的RGB图像（PIL格式）和OpenCV BGR图像
    """
    # 统一转换为OpenCV BGR格式
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 保持比例缩放并居中裁剪到正方形
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = int(target_size * h / w), target_size
    else:
        new_h, new_w = target_size, int(target_size * w / h)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 居中裁剪
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    img_cropped = img_resized[start_y:start_y+target_size, start_x:start_x+target_size]
    
    # 转换为PIL RGB格式（SD输入要求）
    img_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    
    return img_pil, img_cropped

def save_control_image(img, save_path):
    """保存控制图，自动处理格式转换"""
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        Image.fromarray(img).save(save_path)
    else:
        img.save(save_path)