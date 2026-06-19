import cv2
import numpy as np
from PIL import Image
from basic_preprocess import standardize_image,save_control_image

def preprocess_lineart_for_coloring(img, output_size=512, line_thickness=2):
    """
    线稿上色专用预处理：修复扫描/手绘线稿的常见问题
    返回：[标准化白底黑线线稿, 对应Canny硬边缘图]（分别对应两个ControlNet）
    """
    # 复用通用标准化
    _, img_cv = standardize_image(img, output_size)
    
    # 1. 转灰度图
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 背景校正：消除扫描线稿的明暗不均、纸张泛黄
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    corrected = cv2.divide(gray, background, scale=255)
    
    # 3. 全局二值化：分离线条与背景
    _, binary = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. 去除杂点与孤立噪点
    kernel_small = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 5. 断线补全：修复线稿中的断裂线条
    kernel_medium = np.ones((3, 3), np.uint8)
    binary_closed = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # 6. 线条粗细统一
    # 骨架提取
    size = np.size(binary_closed)
    skel = np.zeros(binary_closed.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv2.erode(binary_closed, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_closed, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_closed = eroded.copy()
        zeros = size - cv2.countNonZero(binary_closed)
        if zeros == size:
            done = True
    
    # 膨胀到目标粗细
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        skel = cv2.dilate(skel, kernel, iterations=1)
    
    # 7. 生成第一张图：白底黑线线稿（对应Lineart通用线稿控制）
    final_lineart = 255 - skel
    lineart_pil = Image.fromarray(cv2.cvtColor(final_lineart, cv2.COLOR_GRAY2RGB))
    
    # 8. 新增：生成第二张图：Canny硬边缘图（对应Canny硬边缘控制）
    canny_edges = cv2.Canny(final_lineart, threshold1=50, threshold2=150)
    canny_edges = 255 - canny_edges  # 转为白底黑线
    canny_pil = Image.fromarray(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB))
    
    return lineart_pil, canny_pil
    
def lineart_coloring_preprocess(image_path, output_size=512, save_dir="./lineart_control"):
    """
    线稿上色一键预处理：生成双ControlNet所需的两张控制图
    返回：[处理后的线稿, Canny边缘图]
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    original = Image.open(image_path).convert("RGB")
    processed_lineart, processed_canny = preprocess_lineart_for_coloring(original, output_size)
    
    save_control_image(processed_lineart, f"{save_dir}/1_processed_lineart.png")
    save_control_image(processed_canny, f"{save_dir}/2_processed_canny.png")
    print("✓ 线稿上色双控制图生成完成")
    print(f"  - 1_processed_lineart.png：对应Lineart通用线稿控制")
    print(f"  - 2_processed_canny.png：对应Canny硬边缘控制")
    
    return processed_lineart, processed_canny