import os
os.environ["MEDIAPIPE_DISABLE_TELEMETRY"] = "1"

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision
from basic_preprocess import standardize_image,save_control_image
from depth_preprocess import depth_estimator,process_depth_map
# 正确导入MediaPipe核心类
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

_pose_landmarker = None
_face_landmarker = None

def extract_anime_lineart(img, output_size=512, line_thickness=1):
    """
    提取真人照片的动漫风格干净线稿
    参数：
        img: 输入图像（OpenCV BGR或PIL）
        output_size: 输出尺寸
        line_thickness: 线条粗细，1-3之间调整
    返回：
        白底黑线的动漫线稿（PIL格式）
    """
    # 标准化图像
    _, img_cv = standardize_image(img, output_size)
    
    # 1. 转灰度图
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 双边滤波：保边去噪，去除皮肤纹理、毛孔、光影杂线（核心步骤）
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. 自适应局部二值化：适配不同亮度区域，避免明暗断层
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=5
    )
    
    # 4. 形态学开运算：去除杂点、毛刺
    kernel = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 5. 线条细化（骨架提取）：统一线条粗细，实现动漫风格细线效果
    size = np.size(binary_clean)
    skel = np.zeros(binary_clean.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv2.erode(binary_clean, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_clean, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_clean = eroded.copy()
        zeros = size - cv2.countNonZero(binary_clean)
        if zeros == size:
            done = True
    
    # 6. 可选：调整线条粗细
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        skel = cv2.dilate(skel, kernel, iterations=1)
    
    # 7. 反色成白底黑线（ControlNet标准输入格式）
    lineart = 255 - skel
    
    # 转换为PIL格式
    lineart_pil = Image.fromarray(cv2.cvtColor(lineart, cv2.COLOR_GRAY2RGB))
    
    return lineart_pil
    
    
def generate_openpose_map(image, output_size=(512, 512), detect_face=True):
    """
    生成ControlNet兼容的OpenPose骨骼图（包含面部关键点）
    修复了Windows死锁bug：不使用with语句，不调用close()方法
    """
    global _pose_landmarker, _face_landmarker
    
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    
    h, w = img_np.shape[:2]
    
    # 创建黑色背景
    pose_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 构建模型绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pose_model_path = os.path.join(current_dir, "models", "pose_landmarker_full.task")
    face_model_path = os.path.join(current_dir, "models", "face_landmarker.task")
    
    # 验证模型文件是否存在
    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(f"姿态模型文件不存在: {pose_model_path}")
    
    # 只创建一次PoseLandmarker实例
    if _pose_landmarker is None:
        base_options_pose = BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 不使用with语句，直接创建实例
        _pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
    
    # 检测人体姿态
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result = _pose_landmarker.detect(mp_image)
    
    if result.pose_landmarks:
        for landmarks in result.pose_landmarks:
            # 绘制人体关键点
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(pose_img, (x, y), 4, (255, 255, 255), -1)
            
            # 绘制人体骨骼连接（ControlNet标准）
            pose_connections = [
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
                (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                (29, 31), (30, 32)
            ]
            
            for start_idx, end_idx in pose_connections:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_x, start_y = int(start.x * w), int(start.y * h)
                end_x, end_y = int(end.x * w), int(end.y * h)
                cv2.line(pose_img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
    
    # 检测面部关键点（如果开启且模型存在）
    if detect_face and os.path.exists(face_model_path):
        # 只创建一次FaceLandmarker实例
        if _face_landmarker is None:
            base_options_face = BaseOptions(model_asset_path=face_model_path)
            options_face = vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            _face_landmarker = vision.FaceLandmarker.create_from_options(options_face)
        
        face_result = _face_landmarker.detect(mp_image)
        
        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                # 绘制面部关键点
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(pose_img, (x, y), 2, (255, 255, 255), -1)
    
    # 调整到输出大小
    pose_img = cv2.resize(pose_img, output_size)
    
    # 转换为PIL Image
    return Image.fromarray(pose_img)
        
    
def preprocess_all(image_path, output_size=512, save_dir="./control_maps"):
    """
    一键完成所有预处理，生成三个控制图并保存
    参数：
        image_path: 输入照片路径
        output_size: 输出尺寸
        save_dir: 控制图保存目录
    返回：
        标准化原图、动漫线稿、OpenPose姿态图、优化后深度图
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取原图
    original_img = Image.open(image_path).convert("RGB")
    
    # 1. 标准化原图
    standardized_img, _ = standardize_image(original_img, output_size)
    save_control_image(standardized_img, f"{save_dir}/0_original.jpg")
    print("✓ 原图标准化完成")
    
    # 2. 提取动漫线稿
    lineart = extract_anime_lineart(original_img, output_size)
    save_control_image(lineart, f"{save_dir}/1_lineart_anime.png")
    print("✓ 动漫线稿提取完成")
    
    # 3. 生成OpenPose姿态图
    openpose_map = generate_openpose_map(original_img, (output_size,output_size))
    save_control_image(openpose_map, f"{save_dir}/2_openpose.png")
    print("✓ OpenPose姿态图生成完成")

    raw_depth = depth_estimator(standardized_img)
    processed_depth = process_depth_map(raw_depth, output_size)
    save_control_image(processed_depth, f"{save_dir}/3_depth.png")
    print("✓ 深度图生成与优化完成")
 
    print(f"\n所有控制图已保存到 {save_dir} 目录")
    return standardized_img, lineart, openpose_map , processed_depth