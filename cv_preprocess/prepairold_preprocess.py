import os
import cv2
import numpy as np
from PIL import Image
from basic_preprocess import standardize_image,save_control_image
from depth_preprocess import depth_estimator,process_depth_map

def preprocess_old_photo(img, output_size=512, scratch_removal=True):
    """
    老照片修复专用预处理：去除划痕、斑点、增强对比度
    参数：
        img: 输入老照片（扫描/拍摄）
        output_size: 输出尺寸
        scratch_removal: 是否开启划痕去除
    返回：
        修复后的老照片 + 提取的清晰边缘图（PIL格式）
    """
    # 复用通用标准化
    _, img_cv = standardize_image(img, output_size)
    
    # 1. 转灰度图（老照片多为黑白）
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 对比度增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 去除斑点与小划痕（中值滤波）
    denoised = cv2.medianBlur(enhanced, 3)
    
    # 4. 去除长划痕（核心步骤）
    if scratch_removal:
        # 检测线性划痕
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        scratches = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
        
        # 阈值分割划痕
        _, scratch_mask = cv2.threshold(scratches, 20, 255, cv2.THRESH_BINARY)
        
        # 膨胀划痕区域
        kernel = np.ones((2, 2), np.uint8)
        scratch_mask = cv2.dilate(scratch_mask, kernel, iterations=1)
        
        # 用周围像素修复划痕
        repaired = cv2.inpaint(denoised, scratch_mask, 3, cv2.INPAINT_TELEA)
    else:
        repaired = denoised
    
    # 5. 锐化增强细节
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(repaired, -1, kernel)
    
    # 6. 提取清晰边缘图（用于ControlNet控制）
    edges = cv2.Canny(sharpened, threshold1=50, threshold2=150)
    edges = 255 - edges  # 转为白底黑线
    
    # 转换为PIL格式
    repaired_pil = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
    edges_pil = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    
    return repaired_pil, edges_pil
	
	# ---------------------- 3. 老照片修复一键预处理 ----------------------
def old_photo_restoration_preprocess(image_path, output_size=512, save_dir="./old_photo_control"):
    os.makedirs(save_dir, exist_ok=True)
    
    original = Image.open(image_path).convert("RGB")
    repaired_photo, edges_map = preprocess_old_photo(original, output_size)

    raw_depth = depth_estimator(repaired_photo)
    processed_depth = process_depth_map(raw_depth, output_size)

    save_control_image(repaired_photo, f"{save_dir}/repaired_photo.jpg")
    save_control_image(edges_map, f"{save_dir}/edges_map.png")
    save_control_image(processed_depth, f"{save_dir}/depth_map.png")
    print("✓ 老照片修复预处理完成")
    return repaired_photo, edges_map , processed_depth