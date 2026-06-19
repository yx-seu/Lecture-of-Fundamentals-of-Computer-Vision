import os
import cv2
import numpy as np
from PIL import Image
from basic_preprocess import standardize_image,save_control_image


def preprocess_sketch_for_generation(img, output_size=512, filter_strength=1):
    """
    手绘草图专用预处理：去除涂改、杂线、背景纹理，提取核心结构
    参数：
        img: 输入手绘草图（手机拍摄/平板手绘）
        output_size: 输出尺寸
        filter_strength: 杂线过滤强度，1-3之间
    返回：
        标准化白底黑线条稿（PIL格式）
    """
    # 复用通用标准化
    _, img_cv = standardize_image(img, output_size)
    
    # 1. 转灰度图
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 光照校正：消除拍摄草图的阴影、光照不均
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 3. 自适应二值化：提取草稿线条
    binary = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=10
    )
    
    # 4. 过滤小杂点与涂改痕迹（核心步骤）
    # 连通域分析：去除面积过小的杂点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 10 * filter_strength  # 最小保留面积
    
    clean_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_binary[labels == i] = 255
    
    # 5. 去除细长辅助线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_OPEN, kernel, iterations=filter_strength)
    
    # 6. 线条增强
    kernel = np.ones((2, 2), np.uint8)
    clean_binary = cv2.dilate(clean_binary, kernel, iterations=1)
    
    # 7. 反色为白底黑线
    final_sketch = 255 - clean_binary
    
    return Image.fromarray(cv2.cvtColor(final_sketch, cv2.COLOR_GRAY2RGB))
    
    
# ---------------------- 2. 手绘草图转成品一键预处理 ----------------------
def sketch_generation_preprocess(image_path, output_size=512, save_dir="./sketch_control"):
    os.makedirs(save_dir, exist_ok=True)
    
    original = Image.open(image_path).convert("RGB")
    processed_sketch = preprocess_sketch_for_generation(original, output_size)
    
    save_control_image(processed_sketch, f"{save_dir}/processed_sketch.png")
    print("✓ 手绘草图预处理完成")
    return processed_sketch