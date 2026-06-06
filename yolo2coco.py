import os
import json
import cv2
from tqdm import tqdm

# ================= ⚠️ 请根据你的实际情况修改这里 =================
# 1. 数据集根目录 (你的 train 和 val 文件夹所在的上一级目录)
ROOT_PATH = r"C:\date\Visdrone2019"

# 2. 你的类别名称 (顺序必须和 classes.txt 或者你训练 YOLO 时的 yaml 文件一致)
# ⚠️ 注意：VisDrone 的类别通常如下，如果你的不同请修改！
CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# 3. 你想转换哪个集？ (通常需要运行两次：一次填 "train"，一次填 "val")
PHASE = "train"
# PHASE = "val"

# 4. 你的图片和标签文件夹名字 (根据你的实际目录修改)
# 假设结构是: root/images/train 和 root/labels/train
IMG_DIR = os.path.join(ROOT_PATH, "images", PHASE)
TXT_DIR = os.path.join(ROOT_PATH, "labels", PHASE)
SAVE_PATH = os.path.join(ROOT_PATH, "annotations", f"instances_{PHASE}.json")


# ===============================================================

def yolo_to_coco():
    # 如果 annotations 文件夹不存在，创建它
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))

    # 初始化 COCO 字典
    dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 1. 写入类别信息 (Categories)
    for i, cls_name in enumerate(CLASSES):
        dataset["categories"].append({
            "id": i + 1,  # COCO 类别 ID 习惯从 1 开始
            "name": cls_name,
            "supercategory": "object"
        })

    # 获取所有图片
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    annotation_id = 1
    image_id = 1

    print(f"🚀 开始转换 {PHASE} 集，共有 {len(image_files)} 张图片...")

    for img_file in tqdm(image_files):
        # --- A. 处理图片信息 ---
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ 警告：无法读取图片 {img_file}，已跳过")
            continue

        height, width, _ = img.shape

        dataset["images"].append({
            "id": image_id,
            "file_name": img_file,  # 这里的名字必须和文件夹里的文件名完全一致
            "width": width,
            "height": height
        })

        # --- B. 处理对应的 TXT 标签 ---
        txt_name = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(TXT_DIR, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                # 过滤掉空行或格式错误的行
                if len(parts) < 5: continue

                # 读取 YOLO 格式 (归一化)
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # ⚠️ 核心转换逻辑：YOLO(归一化中心点) -> COCO(绝对左上角坐标)
                w_abs = w_norm * width
                h_abs = h_norm * height
                x_min = (x_center * width) - (w_abs / 2)
                y_min = (y_center * height) - (h_abs / 2)

                # 写入标注
                dataset["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id + 1,  # ID + 1 (对应上面的 categories)
                    "bbox": [x_min, y_min, w_abs, h_abs],  # [x, y, w, h]
                    "area": w_abs * h_abs,
                    "iscrowd": 0,
                    "segmentation": []  # 检测任务留空即可
                })
                annotation_id += 1

        image_id += 1

    # --- C. 保存 JSON 文件 ---
    with open(SAVE_PATH, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ 成功！JSON 已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    yolo_to_coco()