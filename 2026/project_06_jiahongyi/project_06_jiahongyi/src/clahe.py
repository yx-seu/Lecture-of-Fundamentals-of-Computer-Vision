import cv2
import os

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

print("脚本目录:", base_dir)

sub_dirs = ["train", "val", "test"]

# CLAHE参数
clip_limit = 2.0
tile_grid_size = (8, 8)

# 创建CLAHE对象
clahe = cv2.createCLAHE(
    clipLimit=clip_limit,
    tileGridSize=tile_grid_size
)

for sub_dir in sub_dirs:

    folder_path = os.path.join(base_dir, sub_dir)

    print(folder_path)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    count = 0

    for file_name in os.listdir(folder_path):

        # 只处理jpg文件
        if not file_name.lower().endswith(".jpg"):
            continue

        # 避免重复处理已经生成的CLAHE图片
        if "_clahe" in file_name:
            continue

        img_path = os.path.join(folder_path, file_name)

        # 灰度读取
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Cannot read: {img_path}")
            continue

        # CLAHE增强
        img_clahe = clahe.apply(img)

        # 生成新文件名
        new_file_name = file_name.replace(".jpg", "_clahe.jpg")

        save_path = os.path.join(folder_path, new_file_name)

        # 保存
        cv2.imwrite(save_path, img_clahe)

        count += 1

    print(f"Finished {sub_dir}: {count} images processed")

print("\nAll done!")