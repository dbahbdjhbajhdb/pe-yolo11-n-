import json
import os
import cv2
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# ================= 请修改这两行 =================
# 1. 你的验证集【图片】文件夹路径 (注意是 images 不是 labels)
#    参考你之前的日志，可能是类似 C:\date\Visdrone2019\images\val
VAL_IMG_DIR = r"C:\date\Visdrone2019\images\val2019"

# 2. 你的验证集【标签】文件夹路径 (里面是一堆 .txt)
#    参考你之前的日志，可能是 C:\date\Visdrone2019\labels\val
VAL_LABEL_DIR = r"C:\date\Visdrone2019\labels\val2019"

# 3. YOLO 跑出来的预测结果 (predictions.json)
#    请确保这个文件存在！(检查一下 val3 是不是最新的)
PRED_JSON = r'C:\ultralytics-main\runs\detect\val3\predictions.json'


# ===============================================

def generate_gt_json(img_dir, label_dir, save_name='visdrone_gt.json'):
    print(f"正在从 {label_dir} 生成真值 JSON...")
    dataset = {"images": [], "annotations": [], "categories": []}

    # 定义类别 (VisDrone 默认 10 类)
    categories = ["pedestrian", "people", "bicycle", "car", "van",
                  "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    for i, name in enumerate(categories):
        dataset['categories'].append({"id": i, "name": name, "supercategory": "object"})

    # 扫描所有图片
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

    ann_id = 0
    for img_path in tqdm(img_paths):
        # 读取图片信息
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        file_name = os.path.basename(img_path)
        # image_id 使用文件名中的数字 (去掉后缀)
        # 例如 00001.jpg -> 1 (有些文件名带前缀，这里直接用文件名做 key 也可以，但为了匹配 prediction 建议用纯数字如果可能)
        # 简单起见，我们直接用文件名作为 ID 的哈希或者就在这里硬匹配
        # 但 YOLO 的 prediction.json 通常用的是文件名的 string。
        # 为了保险，我们这里存 file_name，后面用 pycocotools 自动匹配

        img_id = os.path.splitext(file_name)[0]  # ID 是字符串 "00003"

        dataset['images'].append({
            "id": img_id,  # 保持字符串 ID，这就不用担心不匹配了
            "width": w,
            "height": h,
            "file_name": file_name
        })

        # 读取对应标签
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(label_dir, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls_id = int(parts[0])
                    # YOLO格式: x_center y_center w h (归一化)
                    cx, cy, bw, bh = map(float, parts[1:5])

                    # 转回像素坐标 xywh (左上角)
                    abs_w = bw * w
                    abs_h = bh * h
                    abs_x = (cx * w) - (abs_w / 2)
                    abs_y = (cy * h) - (abs_h / 2)

                    dataset['annotations'].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1

    with open(save_name, 'w') as f:
        json.dump(dataset, f)
    print(f"真值文件已生成: {save_name}")
    return save_name


def main():
    # 1. 如果没有真值 JSON，先生成一个
    gt_json = 'visdrone_gt.json'
    if not os.path.exists(gt_json):
        generate_gt_json(VAL_IMG_DIR, VAL_LABEL_DIR, gt_json)

    # 2. 开始评测
    print("加载真值...")
    cocoGt = COCO(gt_json)
    print(f"加载预测值: {PRED_JSON}")

    try:
        cocoDt = cocoGt.loadRes(PRED_JSON)
    except Exception as e:
        print(f"错误: 无法加载预测文件。原因: {e}")
        return

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("\n请查看上面表格中的 'area= small' 行！")


if __name__ == '__main__':
    main()